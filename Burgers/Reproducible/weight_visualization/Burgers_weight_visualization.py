import os
import time

# Set before importing torch so cuBLAS can use deterministic kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import scipy.io
import numpy as np
from pyDOE import lhs
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


seed = 3412
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
# 2. Data loading (burgers_shock.mat)
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'figures')  # Put in Burgers/Reproducible/figures
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Look two levels up for burgers_shock.mat (e.g. Burgers/burgers_shock.mat)
data_path = os.path.dirname(os.path.dirname(file_path))
try:
    data = scipy.io.loadmat(os.path.join(data_path, 'burgers_shock.mat'))
except FileNotFoundError:
    # Try one level further up just in case
    data_path_alt = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    data = scipy.io.loadmat(os.path.join(data_path_alt, 'burgers_shock.mat'))

nu = 0.01 / np.pi
Exact = np.real(data['usol'])
t0 = data['t'].flatten()[:, None]
x0 = data['x'].flatten()[:, None]

X, T = np.meshgrid(x0.flatten(), t0.flatten())
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

nx, nt = x0.shape[0], t0.shape[0]
N_i = nx
N_b = nt
N_u = N_i + 2 * N_b
N_f = 10000

X_i = np.hstack((X[0:1, :].T, T[0:1, :].T))
u_i = Exact[:, 0:1]
X_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
X_ub = np.hstack((X[:, -1:], T[:, -1:]))
u_lb = Exact[0:1, :].T
u_ub = Exact[-1:, :].T
X_u = np.vstack([X_i, X_lb, X_ub])
u = np.vstack([u_i, u_lb, u_ub])

lb = X_star.min(0)
ub = X_star.max(0)
X_f = lb + (ub - lb) * lhs(2, N_f, random_state=seed)
X_f = np.vstack((X_f, X_u))

layers = [2] + 8 * [20] + [1]


def source_torch(x, t):
    return torch.zeros_like(x)


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
# 4. Model Class Definitions (SA, RBA, C-LoReW)
# ==========================================

class SA_PINN_Runner:
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.lamr = torch.nn.Parameter(torch.rand(self.N_r, 1, dtype=torch.float32, device=device))
        self.lamu = torch.nn.Parameter(torch.rand(self.N_u, 1, dtype=torch.float32, device=device) * 100.0)
        self.optimizer2 = torch.optim.Adam([self.lamr, self.lamu], lr=0.005, maximize=True)

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.step_size = 2000

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t + u * u_x - nu * u_xx
        return f

    def loss_func(self):
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred = self.net_r(self.x_r, self.t_r)

        loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
        loss_u = torch.mean((self.lamu * (self.u_pred - self.u)) ** 2)

        self.loss = loss_r + loss_u
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
            if (step + 1) % self.step_size == 0:
                self.scheduler.step()
        return collected_weights


class RBA_PINN_Runner:
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 0.0001
        self.gamma = 0.9999
        self.init = 1
        self.rba_eps = 1e-12

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.step_size = 2000

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t + u * u_x - nu * u_xx
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred = self.net_r(self.x_r, self.t_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(self.r_pred).detach()
        r_norm = eta * abs_r_detach / (torch.max(abs_r_detach) + self.rba_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_u = torch.mean((self.u_pred - self.u) ** 2)

        self.loss = loss_r + loss_u
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


class C_LoReW_PINN_Runner:
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 0.0001
        self.gamma = 0.9999
        self.init = 1

        self.alpha = 2.0
        self.epsilon = 2.0
        self.tau_gate = 0
        self.eta_g = 0.00001
        self.rho_g = 0.999
        self.gbar = torch.zeros_like(self.x_r).detach()

        self.kernel_h = 0.1
        self.kernel_p = 1.0
        self.kernel_trunc = 3.0
        self.num_kernel_centers = 41
        self.kernel_eps = 1e-12

        t_min = float(lb[1])
        t_max = float(ub[1])
        self.t_centers = torch.linspace(t_min, t_max, self.num_kernel_centers, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps
        self.gate_abs_res_ref = None

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.step_size = 2000

    def build_kernel_weights(self, t_points):
        diff = t_points - self.t_centers
        K = torch.exp(-0.5 * (diff / self.kernel_h) ** 2)
        if self.kernel_trunc is not None and self.kernel_trunc > 0:
            K = K * (torch.abs(diff) <= self.kernel_trunc * self.kernel_h).float()

        row_sum = K.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            nearest_idx = torch.argmin(torch.abs(diff[zero_rows]), dim=1)
            K[zero_rows] = 0.0
            K[zero_rows, nearest_idx] = 1.0
            row_sum = K.sum(dim=1, keepdim=True)

        return K / (row_sum + self.kernel_eps)

    def kernel_local_scale(self, abs_r_detach):
        center_moment_p = (self.kernel_weights.T @ (abs_r_detach ** self.kernel_p)) / self.kernel_col_mass
        local_moment_p = self.kernel_weights @ center_moment_p
        return (local_moment_p + self.kernel_eps) ** (1.0 / self.kernel_p)

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t + u * u_x - nu * u_xx
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred = self.net_r(self.x_r, self.t_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        g_t = 0.5 * (1.0 - torch.tanh(self.alpha * (self.t_r - self.tau_gate)))
        self.gbar = (self.rho_g * self.gbar + (1.0 - self.rho_g) * g_t.detach()).detach()

        abs_r_detach = torch.abs(self.r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
        causal_r_norm = r_norm * self.gbar
        self.rsum = (self.rsum * self.gamma + causal_r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_u = torch.mean((self.u_pred - self.u) ** 2)

        self.loss = loss_r + loss_u
        self.loss.backward()
        self.iter += 1

        gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.kernel_eps)
        if self.gate_abs_res_ref is None:
            self.gate_abs_res_ref = gate_abs_res.item() + self.kernel_eps
        gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref

        self.tau_gate = self.tau_gate + self.eta_g * np.exp(-self.epsilon * gate_abs_res_norm)
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
    checkpoints = [10000, 25000, 40000]

    # Run SA-PINN
    print("Training SA-PINN and collecting weights...")
    sa_runner = SA_PINN_Runner(X_u, u, X_f)
    sa_weights = sa_runner.train_and_collect_weights(checkpoints)

    # Run RBA-PINN
    print("Training RBA-PINN and collecting weights...")
    rba_runner = RBA_PINN_Runner(X_u, u, X_f)
    rba_weights = rba_runner.train_and_collect_weights(checkpoints)

    # Run C-LoReW-PINN
    print("Training C-LoReW-PINN and collecting weights...")
    clorew_runner = C_LoReW_PINN_Runner(X_u, u, X_f)
    clorew_weights = clorew_runner.train_and_collect_weights(checkpoints)

    # Plotting 3x3 Heatmap Grid
    print("Generating 3x3 interpolated heatmap grid...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    methods_data = [
        ("SA-PINN", sa_weights),
        ("RBA-PINN", rba_weights),
        ("C-LoReW-PINN (Ours)", clorew_weights)
    ]

    t_f = X_f[:, 1]
    x_f = X_f[:, 0]

    from scipy.interpolate import griddata
    t_grid, x_grid = np.meshgrid(t0.flatten(), x0.flatten())

    for r_idx, (method_name, weights_dict) in enumerate(methods_data):
        for c_idx, iter_val in enumerate(checkpoints):
            ax = axes[r_idx, c_idx]
            w = weights_dict[iter_val]

            # Interpolate weights onto the regular grid
            w_grid = griddata((t_f, x_f), w, (t_grid, x_grid), method='linear')

            # Fill potential NaNs near boundaries using nearest neighbor interpolation
            nan_mask = np.isnan(w_grid)
            if np.any(nan_mask):
                w_grid_nearest = griddata((t_f, x_f), w, (t_grid, x_grid), method='nearest')
                w_grid[nan_mask] = w_grid_nearest[nan_mask]

            # Plot interpolated heatmap
            im = ax.imshow(
                w_grid,
                cmap='jet',
                aspect='auto',
                extent=[t0.min(), t0.max(), x0.min(), x0.max()],
                origin='lower'
            )

            # Overlay contour lines of the exact solution
            ax.contour(t0.flatten(), x0.flatten(), Exact, levels=5, colors='black', alpha=0.5, linestyles='dashed', linewidths=1.0)

            # Set titles and labels
            if r_idx == 0:
                ax.set_title(f"Iteration {iter_val}", fontsize=14, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f"{method_name}\nx", fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel("x", fontsize=12)
            ax.set_xlabel("t", fontsize=12)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, "burgers_weight_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}!")
