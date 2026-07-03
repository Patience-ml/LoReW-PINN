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
LOG_DIR = os.path.join(file_path, 'logs')
FIGURE_DIR = os.path.join(os.path.dirname(file_path), 'figures')  # Put in Helmholtz/Reproducible/figures
os.makedirs(LOG_DIR, exist_ok=True)
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
# 4. Model Class Definitions (SA, RBA, LoReW)
# ==========================================

class SA_PINN_Runner:
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

    def train_and_collect_weights(self, checkpoints):
        self.dnn.train()
        collected_weights = {}
        max_steps = max(checkpoints)
        for step in range(max_steps):
            self.loss_func()
            if self.iter in checkpoints:
                # Store the weights
                w = tonp(self.lamr).flatten()
                collected_weights[self.iter] = w
            if (step + 1) % self.step_size == 0:
                self.scheduler.step()
        return collected_weights


class RBA_PINN_Runner:
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

    def train_and_collect_weights(self, checkpoints):
        self.dnn.train()
        collected_weights = {}
        max_steps = max(checkpoints)
        for step in range(max_steps):
            self.loss_func()
            if self.iter in checkpoints:
                w = tonp(self.rsum).flatten()
                collected_weights[self.iter] = w
            if (step + 1) % self.step_size == 0:
                self.scheduler.step()
        return collected_weights


class LoReW_PINN_Runner:
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

    def train_and_collect_weights(self, checkpoints):
        self.dnn.train()
        collected_weights = {}
        max_steps = max(checkpoints)
        for step in range(max_steps):
            self.loss_func()
            if self.iter in checkpoints:
                w = tonp(self.rsum).flatten()
                collected_weights[self.iter] = w
            if (step + 1) % self.step_size == 0:
                self.scheduler.step()
        return collected_weights


# ==========================================
# 5. Training execution and plotting
# ==========================================
if __name__ == "__main__":
    checkpoints = [1000, 5000, 15000]

    # Run SA-PINN
    print("Training SA-PINN and collecting weights...")
    sa_runner = SA_PINN_Runner(X_u, u, X_r)
    sa_weights = sa_runner.train_and_collect_weights(checkpoints)

    # Run RBA-PINN
    print("Training RBA-PINN and collecting weights...")
    rba_runner = RBA_PINN_Runner(X_u, u, X_r)
    rba_weights = rba_runner.train_and_collect_weights(checkpoints)

    # Run LoReW-PINN
    print("Training LoReW-PINN and collecting weights...")
    lorew_runner = LoReW_PINN_Runner(X_u, u, X_r)
    lorew_weights = lorew_runner.train_and_collect_weights(checkpoints)

    # Plotting 3x3 Heatmap Grid
    print("Generating 3x3 heatmap grid...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    methods_data = [
        ("SA-PINN", sa_weights),
        ("RBA-PINN", rba_weights),
        ("LoReW-PINN (Ours)", lorew_weights)
    ]

    # Prepare exact solution coordinates for contouring
    x0_plot = np.linspace(x_min, x_max, N_grid)
    y0_plot = np.linspace(y_min, y_max, N_grid)
    X_plot, Y_plot = np.meshgrid(x0_plot, y0_plot)
    Exact_plot = (np.sin(xdim * np.pi * X_plot) * np.sin(ydim * np.pi * Y_plot)).T

    for r_idx, (method_name, weights_dict) in enumerate(methods_data):
        # We use independent scales because weights magnitudes differ significantly by design
        for c_idx, iter_val in enumerate(checkpoints):
            ax = axes[r_idx, c_idx]
            w = weights_dict[iter_val]

            # Reshape weight to 2D matching coordinates
            # Since exact solution is plotted with x as vertical and y as horizontal:
            # extent=[y_min, y_max, x_min, x_max], origin='lower'
            # we do w.reshape(N_grid, N_grid).T
            w_2d = w.reshape(N_grid, N_grid).T

            # Plot heatmap of weights
            im = ax.imshow(w_2d, cmap='jet', aspect='auto',
                           extent=[y_min, y_max, x_min, x_max], origin='lower')

            # Overlay contour lines of the exact solution
            ax.contour(y0_plot, x0_plot, Exact_plot, levels=5, colors='black', alpha=0.5, linestyles='dashed', linewidths=1.0)

            # Set titles and labels
            if r_idx == 0:
                ax.set_title(f"Iteration {iter_val}", fontsize=14, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f"{method_name}\nx", fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel("x", fontsize=12)
            ax.set_xlabel("y", fontsize=12)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, "helmholtz_weight_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}!")
