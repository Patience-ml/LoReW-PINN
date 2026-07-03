import os
import time

# Set before importing torch so cuBLAS can use deterministic kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import numpy as np
from pyDOE import lhs
import matplotlib as mpl
import matplotlib.pyplot as plt
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


seed = 5899
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


def exact_u_np(x, y):
    return np.cos(C * x) + np.cos(C * y)


def source_torch(x, y):
    return 2.0 * (C ** 2) * (torch.cos(C * x) + torch.cos(C * y))


# ==========================================
# 2. Data generation with analytical solution
#    Paper setting: 2D Poisson, C = 10, Omega = [0, 1]^2.
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'figures')  # Put in Poisson/Reproducible/figures
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

C = 10.0
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
nx, ny = 201, 201
N_f = 10000
N_b_each = 100
learning_rate = 1e-3

x0 = np.linspace(x_min, x_max, nx)[:, None]
y0 = np.linspace(y_min, y_max, ny)[:, None]
X, Y = np.meshgrid(x0.flatten(), y0.flatten(), indexing='ij')
Exact = exact_u_np(X, Y)
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

lb = np.array([x_min, y_min])
ub = np.array([x_max, y_max])
X_f = lb + (ub - lb) * lhs(2, N_f, random_state=seed)

s_left = lhs(1, N_b_each, random_state=seed + 1)
s_right = lhs(1, N_b_each, random_state=seed + 2)
s_bottom = lhs(1, N_b_each, random_state=seed + 3)
s_top = lhs(1, N_b_each, random_state=seed + 4)
X_left = np.hstack((np.full((N_b_each, 1), x_min), y_min + (y_max - y_min) * s_left))
X_right = np.hstack((np.full((N_b_each, 1), x_max), y_min + (y_max - y_min) * s_right))
X_bottom = np.hstack((x_min + (x_max - x_min) * s_bottom, np.full((N_b_each, 1), y_min)))
X_top = np.hstack((x_min + (x_max - x_min) * s_top, np.full((N_b_each, 1), y_max)))
X_b = np.vstack((X_left, X_right, X_bottom, X_top))
u_b = exact_u_np(X_b[:, 0:1], X_b[:, 1:2])

layers = [2] + 8 * [50] + [1]


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
    def __init__(self, X_b, u_b, X_r):
        self.iter = 0
        self.x_b = torch.tensor(X_b[:, 0:1], requires_grad=True).float().to(device)
        self.y_b = torch.tensor(X_b[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.y_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u_b = torch.tensor(u_b).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_b = self.x_b.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        lamr = torch.rand(self.N_r, 1).float().to(device) * 1.0
        lamb = torch.rand(self.N_b, 1).float().to(device) * 100.0
        self.lamr = torch.nn.Parameter(lamr)
        self.lamb = torch.nn.Parameter(lamb)
        self.optimizer2 = torch.optim.Adam([self.lamr, self.lamb], lr=0.005, maximize=True)

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    def net_u(self, x, y):
        return self.dnn(torch.cat([x, y], dim=1))

    def net_r(self, x, y):
        u = self.net_u(x, y)
        u_x = grad(u, x)
        u_y = grad(u, y)
        u_xx = grad(u_x, x)
        u_yy = grad(u_y, y)
        f = -u_xx - u_yy + (C ** 2) * u - source_torch(x, y)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        self.u_b_pred = self.net_u(self.x_b, self.y_b)
        self.r_pred = self.net_r(self.x_r, self.y_r)

        loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
        loss_b = torch.mean((self.lamb * (self.u_b_pred - self.u_b)) ** 2)

        self.loss = loss_r + loss_b
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
                w = tonp(self.lamr).flatten()
                collected_weights[self.iter] = w
        return collected_weights


class RBA_PINN_Runner:
    def __init__(self, X_b, u_b, X_r):
        self.iter = 0
        self.x_b = torch.tensor(X_b[:, 0:1], requires_grad=True).float().to(device)
        self.y_b = torch.tensor(X_b[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.y_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u_b = torch.tensor(u_b).float().to(device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 0.001
        self.gamma = 0.999
        self.init = 1
        self.rba_eps = 1e-12

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    def net_u(self, x, y):
        return self.dnn(torch.cat([x, y], dim=1))

    def net_r(self, x, y):
        u = self.net_u(x, y)
        u_x = grad(u, x)
        u_y = grad(u, y)
        u_xx = grad(u_x, x)
        u_yy = grad(u_y, y)
        f = -u_xx - u_yy + (C ** 2) * u - source_torch(x, y)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_b_pred = self.net_u(self.x_b, self.y_b)
        self.r_pred = self.net_r(self.x_r, self.y_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(self.r_pred).detach()
        r_norm = eta * abs_r_detach / (torch.max(abs_r_detach) + self.rba_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_b = torch.mean((self.u_b_pred - self.u_b) ** 2)

        self.loss = loss_r + loss_b
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
        return collected_weights


class LoReW_PINN_Runner:
    def __init__(self, X_b, u_b, X_r):
        self.iter = 0
        self.x_b = torch.tensor(X_b[:, 0:1], requires_grad=True).float().to(device)
        self.y_b = torch.tensor(X_b[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.y_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u_b = torch.tensor(u_b).float().to(device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 2.5e-4
        self.gamma = 0.997
        self.init = 1
        self.kernel_h_x = 0.05
        self.kernel_h_y = 0.05
        self.kernel_p = 2.0
        self.kernel_trunc = 3.0
        self.num_kernel_centers_x = 31
        self.num_kernel_centers_y = 31
        self.kernel_eps = 1e-12

        centers_x = torch.linspace(float(lb[0]), float(ub[0]), self.num_kernel_centers_x, device=device)
        centers_y = torch.linspace(float(lb[1]), float(ub[1]), self.num_kernel_centers_y, device=device)
        mesh_x, mesh_y = torch.meshgrid(centers_x, centers_y, indexing='ij')
        self.kernel_centers_x = mesh_x.reshape(1, -1)
        self.kernel_centers_y = mesh_y.reshape(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.x_r.detach(), self.y_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    def build_kernel_weights(self, x_points, y_points):
        dx = (x_points - self.kernel_centers_x) / self.kernel_h_x
        dy = (y_points - self.kernel_centers_y) / self.kernel_h_y
        dist2 = dx ** 2 + dy ** 2
        K = torch.exp(-0.5 * dist2)
        if self.kernel_trunc is not None and self.kernel_trunc > 0:
            K = K * (dist2 <= self.kernel_trunc ** 2).float()

        row_sum = K.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            nearest_idx = torch.argmin(dist2[zero_rows], dim=1)
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
        u_x = grad(u, x)
        u_y = grad(u, y)
        u_xx = grad(u_x, x)
        u_yy = grad(u_y, y)
        f = -u_xx - u_yy + (C ** 2) * u - source_torch(x, y)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_b_pred = self.net_u(self.x_b, self.y_b)
        self.r_pred = self.net_r(self.x_r, self.y_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(self.r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_b = torch.mean((self.u_b_pred - self.u_b) ** 2)

        self.loss = loss_r + loss_b
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
        return collected_weights


# ==========================================
# 5. Training execution and plotting
# ==========================================
if __name__ == "__main__":
    checkpoints = [1000, 5000, 15000]

    # Run SA-PINN
    print("Training SA-PINN and collecting weights...")
    sa_runner = SA_PINN_Runner(X_b, u_b, X_f)
    sa_weights = sa_runner.train_and_collect_weights(checkpoints)

    # Run RBA-PINN
    print("Training RBA-PINN and collecting weights...")
    rba_runner = RBA_PINN_Runner(X_b, u_b, X_f)
    rba_weights = rba_runner.train_and_collect_weights(checkpoints)

    # Run LoReW-PINN
    print("Training LoReW-PINN and collecting weights...")
    lorew_runner = LoReW_PINN_Runner(X_b, u_b, X_f)
    lorew_weights = lorew_runner.train_and_collect_weights(checkpoints)

    # Plotting 3x3 Heatmap Grid
    print("Generating 3x3 interpolated heatmap grid...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    methods_data = [
        ("SA-PINN", sa_weights),
        ("RBA-PINN", rba_weights),
        ("LoReW-PINN (Ours)", lorew_weights)
    ]

    x_f_pts = X_f[:, 0]
    y_f_pts = X_f[:, 1]

    from scipy.interpolate import griddata
    x_grid, y_grid = np.meshgrid(x0.flatten(), y0.flatten(), indexing='ij')

    for r_idx, (method_name, weights_dict) in enumerate(methods_data):
        for c_idx, iter_val in enumerate(checkpoints):
            ax = axes[r_idx, c_idx]
            w = weights_dict[iter_val]

            # Interpolate weights onto the regular grid
            w_grid = griddata((x_f_pts, y_f_pts), w, (x_grid, y_grid), method='linear')

            # Fill potential NaNs near boundaries using nearest neighbor interpolation
            nan_mask = np.isnan(w_grid)
            if np.any(nan_mask):
                w_grid_nearest = griddata((x_f_pts, y_f_pts), w, (x_grid, y_grid), method='nearest')
                w_grid[nan_mask] = w_grid_nearest[nan_mask]

            # Plot interpolated heatmap. Note that we plot w_grid.T so:
            # x is on horizontal axis and y is on vertical axis.
            im = ax.imshow(
                w_grid.T,
                cmap='jet',
                aspect='auto',
                extent=[x_min, x_max, y_min, y_max],
                origin='lower'
            )

            # Overlay contour lines of the exact solution
            ax.contour(
                x0.flatten(),
                y0.flatten(),
                Exact.T,
                levels=5,
                colors='black',
                alpha=0.5,
                linestyles='dashed',
                linewidths=1.0
            )

            # Set titles and labels
            if r_idx == 0:
                ax.set_title(f"Iteration {iter_val}", fontsize=14, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f"{method_name}\ny", fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel("y", fontsize=12)
            ax.set_xlabel("x", fontsize=12)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, "poisson_weight_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}!")
