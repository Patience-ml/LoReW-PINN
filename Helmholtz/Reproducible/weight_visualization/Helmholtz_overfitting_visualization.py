import os
import time

# Set before importing torch so cuBLAS can use deterministic kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from collections import OrderedDict

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.max_open_warning'] = 4

# ==========================================
# 1. Environment setup
# ==========================================
if torch.cuda.is_available():
    print('CUDA available')
    device = torch.device('cuda')
else:
    print('CUDA not available, using CPU')
    device = torch.device('cpu')


def seed_torch(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


seed = 5472
seed_torch(seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def tonp(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    raise TypeError('Unknown type of input, expected torch.Tensor or np.ndarray')


def grad(u, x):
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]


# ==========================================
# 2. Data generation with analytical solution
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(os.path.dirname(file_path), 'figures')  
os.makedirs(FIGURE_DIR, exist_ok=True)

x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
nx, ny = 1001, 1001
xdim, ydim = 4.0, 1.0
ksq = 1.0
N_grid = 101
N_u = 50
N_r = N_grid * N_grid
layers = [2] + 4 * [50] + [1]

x0 = np.linspace(x_min, x_max, nx)[:, None]
y0 = np.linspace(y_min, y_max, ny)[:, None]
X, Y = np.meshgrid(x0.flatten(), y0.flatten())
Exact = (np.sin(xdim * np.pi * X) * np.sin(ydim * np.pi * Y)).T
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

train_x = np.linspace(x_min, x_max, N_grid)[:, None]
train_y = np.linspace(y_min, y_max, N_grid)[:, None]
boundary_x = np.linspace(x_min, x_max, N_u)[:, None]
boundary_y = np.linspace(y_min, y_max, N_u)[:, None]

X_left = np.hstack((np.full((N_u, 1), x_min), boundary_y))
X_right = np.hstack((np.full((N_u, 1), x_max), boundary_y))
X_top = np.hstack((boundary_x, np.full((N_u, 1), y_max)))
X_bottom = np.hstack((boundary_x, np.full((N_u, 1), y_min)))
X_u = np.vstack([X_left, X_top, X_bottom, X_right])
u = np.zeros((X_u.shape[0], 1))

lb = np.array([x_min, y_min], dtype=np.float32)
ub = np.array([x_max, y_max], dtype=np.float32)
X_r_grid, Y_r_grid = np.meshgrid(train_x.flatten(), train_y.flatten())
X_r = np.hstack((X_r_grid.flatten()[:, None], Y_r_grid.flatten()[:, None]))


def source_torch(x, y):
    exact = torch.sin(xdim * np.pi * x) * torch.sin(ydim * np.pi * y)
    force = -((xdim * np.pi) ** 2) * exact - ((ydim * np.pi) ** 2) * exact + ksq * exact
    return force


# ==========================================
# 3. Network definition (Vanilla MLP)
# ==========================================
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh()

        layer_list = list()
        for i in range(self.depth - 1):
            w_layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(w_layer.weight)
            layer_list.append(('layer_%d' % i, w_layer))
            layer_list.append(('activation_%d' % i, self.activation))

        w_layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(w_layer.weight)
        layer_list.append(('layer_%d' % (self.depth - 1), w_layer))

        layer_dict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, x):
        return self.layers(x)


# ==========================================
# 4. Evaluator logic
# ==========================================
class BaseRunner:
    def eval_test_error(self):
        with torch.no_grad():
            self.dnn.eval()
            xx = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
            yy = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
            u_pred = self.net_u(xx, yy)
            u_pred_np = tonp(u_pred)
            sol = np.reshape(u_pred_np, (nx, ny))
            
            l2_rel = np.linalg.norm(self.Exact - sol, 2) / np.linalg.norm(self.Exact, 2)
            self.dnn.train()
            return l2_rel

    def train_and_track(self, max_steps, track_interval=200):
        self.dnn.train()
        history = {'iter': [], 'loss': [], 'test_l2': []}
        for step in range(max_steps):
            self.loss_func()
            
            if self.iter % track_interval == 0:
                test_l2 = self.eval_test_error()
                history['iter'].append(self.iter)
                history['loss'].append(self.loss.item())
                history['test_l2'].append(test_l2)
                
            if (step + 1) % self.step_size == 0:
                self.scheduler.step()
        return history

class SA_PINN_Runner(BaseRunner):
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.bc_weight = 1
        self.Exact = Exact
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        lamr = torch.rand(self.N_r, 1, requires_grad=True).float().to(device) * 1.0
        lamu = torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 1.0
        self.lamr = torch.nn.Parameter(lamr)
        self.lamu = torch.nn.Parameter(lamu)
        self.optimizer2 = torch.optim.Adam([self.lamr, self.lamu], lr=0.005, maximize=True)

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=5e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)
        self.step_size = 1000

    def net_u(self, x, y):
        return self.dnn(torch.cat([x, y], dim=1))

    def net_r(self, x, y):
        u = self.net_u(x, y)
        u_y = grad(u, y)
        u_yy = grad(u_y, y)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_xx + u_yy + ksq * u - source_torch(x, y)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.y_u)
        self.r_pred = self.net_r(self.x_r, self.y_r)

        loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
        loss_u = torch.mean((self.lamu * (self.u_pred - self.u)) ** 2)

        self.loss = loss_r + self.bc_weight * loss_u
        self.loss.backward()
        self.iter += 1

        self.optimizer.step()
        self.optimizer2.step()


class RBA_PINN_Runner(BaseRunner):
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.bc_weight = 1
        self.Exact = Exact
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 0.001
        self.gamma = 0.999
        self.init = 1
        self.rba_eps = 1e-12

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=5e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)
        self.step_size = 1000

    def net_u(self, x, y):
        return self.dnn(torch.cat([x, y], dim=1))

    def net_r(self, x, y):
        u = self.net_u(x, y)
        u_y = grad(u, y)
        u_yy = grad(u_y, y)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_xx + u_yy + ksq * u - source_torch(x, y)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.y_u)
        self.r_pred = self.net_r(self.x_r, self.y_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(self.r_pred).detach()
        r_norm = eta * abs_r_detach / (torch.max(abs_r_detach) + self.rba_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_u = torch.mean((self.u_pred - self.u) ** 2)

        self.loss = loss_r + self.bc_weight * loss_u
        self.loss.backward()
        self.iter += 1

        self.optimizer.step()


class LoReW_PINN_Runner(BaseRunner):
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.bc_weight = 1
        self.Exact = Exact
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 5e-4
        self.gamma = 0.98
        self.init = 1
        self.kernel_h_x = 0.12
        self.kernel_h_y = 0.12
        self.kernel_p = 1.5
        self.kernel_trunc = 3.0
        self.num_kernel_centers_x = 31
        self.num_kernel_centers_y = 31
        self.kernel_eps = 1e-12

        self.x_centers = torch.linspace(x_min, x_max, self.num_kernel_centers_x, device=device).view(1, -1)
        self.y_centers = torch.linspace(y_min, y_max, self.num_kernel_centers_y, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.x_r.detach(), self.y_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=5e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)
        self.step_size = 1000

    def build_kernel_weights(self, x_points, y_points):
        dx = x_points - self.x_centers
        dy = y_points - self.y_centers
        Kx = torch.exp(-0.5 * (dx / self.kernel_h_x) ** 2)
        Ky = torch.exp(-0.5 * (dy / self.kernel_h_y) ** 2)
        if self.kernel_trunc is not None and self.kernel_trunc > 0:
            Kx = Kx * (torch.abs(dx) <= self.kernel_trunc * self.kernel_h_x).float()
            Ky = Ky * (torch.abs(dy) <= self.kernel_trunc * self.kernel_h_y).float()

        K = (Kx[:, :, None] * Ky[:, None, :]).reshape(x_points.shape[0], -1)

        row_sum = K.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            center_x = self.x_centers.T.repeat(1, self.num_kernel_centers_y).reshape(1, -1)
            center_y = self.y_centers.repeat(self.num_kernel_centers_x, 1).reshape(1, -1)
            dist2 = ((x_points[zero_rows] - center_x) / self.kernel_h_x) ** 2
            dist2 = dist2 + ((y_points[zero_rows] - center_y) / self.kernel_h_y) ** 2
            nearest_idx = torch.argmin(dist2, dim=1)
            K[zero_rows] = 0.0
            K[zero_rows, nearest_idx] = 1.0
            row_sum = K.sum(dim=1, keepdim=True)

        return K / (row_sum + self.kernel_eps)

    def kernel_local_scale(self, abs_r_detach):
        center_moment_p = (self.kernel_weights.T @ (abs_r_detach ** self.kernel_p)) / self.kernel_col_mass
        local_moment_p = self.kernel_weights @ center_moment_p
        return (local_moment_p + self.kernel_eps) ** (1.0 / self.kernel_p)

    def net_u(self, x, y):
        return self.dnn(torch.cat([x, y], dim=1))

    def net_r(self, x, y):
        u = self.net_u(x, y)
        u_y = grad(u, y)
        u_yy = grad(u_y, y)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_xx + u_yy + ksq * u - source_torch(x, y)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.y_u)
        self.r_pred = self.net_r(self.x_r, self.y_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(self.r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_u = torch.mean((self.u_pred - self.u) ** 2)

        self.loss = loss_r + self.bc_weight * loss_u
        self.loss.backward()
        self.iter += 1

        self.optimizer.step()


# ==========================================
# 5. Training execution and plotting
# ==========================================
if __name__ == "__main__":
    max_steps = 15000
    track_interval = 200

    # Run SA-PINN
    print("Training SA-PINN and tracking overfitting...")
    sa_runner = SA_PINN_Runner(X_u, u, X_r)
    sa_history = sa_runner.train_and_track(max_steps, track_interval)

    # Run RBA-PINN
    print("Training RBA-PINN and tracking overfitting...")
    rba_runner = RBA_PINN_Runner(X_u, u, X_r)
    rba_history = rba_runner.train_and_track(max_steps, track_interval)

    # Run LoReW-PINN
    print("Training LoReW-PINN and tracking overfitting...")
    lorew_runner = LoReW_PINN_Runner(X_u, u, X_r)
    lorew_history = lorew_runner.train_and_track(max_steps, track_interval)

    # Plotting 1x3 Curve Grid
    print("Generating 1x3 overfitting comparison curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods_data = [
        ("SA-PINN", sa_history),
        ("RBA-PINN", rba_history),
        ("LoReW-PINN (Ours)", lorew_history)
    ]

    for idx, (method_name, hist) in enumerate(methods_data):
        ax1 = axes[idx]
        color1 = 'tab:blue'
        ax1.set_xlabel('Iterations', fontsize=12)
        ax1.set_ylabel('Training Loss', color=color1, fontsize=12)
        ax1.plot(hist['iter'], hist['loss'], color=color1, label='Train Loss', alpha=0.9)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_yscale('log')

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Test Rel. L2 Error', color=color2, fontsize=12)
        ax2.plot(hist['iter'], hist['test_l2'], color=color2, label='Test L2', linestyle='--', alpha=0.9)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        # To align limits roughly for comparison, set identical y-limits for Test L2 across panels
        # but let it auto-scale first
        
        ax1.set_title(method_name, fontsize=14, fontweight='bold')
        
    # Standardize y-axis limits for fair comparison
    # Find global min and max for loss and test l2
    all_loss = sa_history['loss'] + rba_history['loss'] + lorew_history['loss']
    all_l2 = sa_history['test_l2'] + rba_history['test_l2'] + lorew_history['test_l2']
    
    loss_min, loss_max = min(all_loss), max(all_loss)
    l2_min, l2_max = min(all_l2), max(all_l2)
    
    for ax1 in axes:
        ax1.set_ylim([loss_min / 10, loss_max * 10])
        ax2 = ax1.twinx() # This is just getting the reference to the right axis if we had it, but actually we can just iterate again.

    # Re-iterate to set y-limits on twinx axes
    for ax in fig.axes:
        if ax.get_ylabel() == 'Test Rel. L2 Error':
            ax.set_ylim([l2_min / 5, l2_max * 5])

    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, "helmholtz_overfitting_curves.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}!")
