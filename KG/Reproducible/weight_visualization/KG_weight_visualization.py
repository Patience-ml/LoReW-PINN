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


seed = 8192
seed_torch(seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def tonp(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or np.ndarray')


def grad(u, x):
    gradient = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return gradient


def exact_u_np(x, t):
    return x * np.cos(5.0 * np.pi * t) + (x * t) ** 3


def exact_ut_np(x, t):
    return -5.0 * np.pi * x * np.sin(5.0 * np.pi * t) + 3.0 * (x ** 3) * (t ** 2)


def source_torch(x, t):
    u = x * torch.cos(5.0 * np.pi * t) + (x * t) ** 3
    u_tt = -25.0 * (np.pi ** 2) * x * torch.cos(5.0 * np.pi * t) + 6.0 * (x ** 3) * t
    u_xx = 6.0 * x * (t ** 3)
    return u_tt - u_xx + u ** 3


# ==========================================
# 2. Data generation with analytical solution
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'figures')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0
nx, nt = 256, 201
N_u = nx
N_b = nt
N_f = 1000

x0 = np.linspace(x_min, x_max, nx)[:, None]
t0 = np.linspace(t_min, t_max, nt)[:, None]

X, T = np.meshgrid(x0.flatten(), t0.flatten())
Exact = exact_u_np(X, T)
Exact0 = Exact.T

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# Initial-condition points
x_u = x0
t_u = np.zeros((N_u, 1))
u_u = exact_u_np(x_u, t_u)
ut_u = exact_ut_np(x_u, t_u)
X_u = np.hstack((x_u, t_u))

# Dirichlet boundary points
t_b = t0[:N_b, :]
x_lb = np.full_like(t_b, x_min)
x_ub = np.full_like(t_b, x_max)
X_lb = np.hstack((x_lb, t_b))
X_ub = np.hstack((x_ub, t_b))
u_lb = exact_u_np(x_lb, t_b)
u_ub = exact_u_np(x_ub, t_b)

# Interior collocation points
lb = np.array([x_min, t_min])
ub = np.array([x_max, t_max])
X_f = lb + (ub - lb) * lhs(2, N_f, random_state=seed)

layers = [2] + 5 * [50] + [1]


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
    def __init__(self, X_u, u, ut, X_r, X_lb, X_ub, u_lb, u_ub):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.x_lb = torch.tensor(X_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_lb = torch.tensor(X_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_ub = torch.tensor(X_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_ub = torch.tensor(X_ub[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.ut = torch.tensor(ut).float().to(device)
        self.u_lb = torch.tensor(u_lb).float().to(device)
        self.u_ub = torch.tensor(u_ub).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]
        self.N_b = X_lb.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)
        lamr = torch.rand(self.N_r, 1, requires_grad=True).float().to(device) * 1.0
        lamu = torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 100.0
        lamut = torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 100.0
        lamb = torch.rand(self.N_b, 1, requires_grad=True).float().to(device) * 100.0
        self.lamr = torch.nn.Parameter(lamr)
        self.lamu = torch.nn.Parameter(lamu)
        self.lamut = torch.nn.Parameter(lamut)
        self.lamb = torch.nn.Parameter(lamb)
        self.optimizer2 = torch.optim.Adam([self.lamr, self.lamu, self.lamut, self.lamb], lr=0.005, maximize=True)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_tt = grad(u_t, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_tt - u_xx + u ** 3 - source_torch(x, t)
        return f, u_t

    def loss_func(self):
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, ut_pred = self.net_r(self.x_r, self.t_r)
        _, ut0_pred = self.net_r(self.x_u, self.t_u)
        u_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred = self.net_u(self.x_ub, self.t_ub)

        loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
        loss_u = torch.mean((self.lamu * (self.u_pred - self.u)) ** 2)
        loss_ut = torch.mean((self.lamut * (ut0_pred - self.ut)) ** 2)
        loss_b = 0.5 * (torch.mean((self.lamb * (u_lb_pred - self.u_lb)) ** 2) + torch.mean((self.lamb * (u_ub_pred - self.u_ub)) ** 2))
        
        self.loss = loss_r + loss_u + loss_ut + loss_b
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
    def __init__(self, X_u, u, ut, X_r, X_lb, X_ub, u_lb, u_ub):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.x_lb = torch.tensor(X_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_lb = torch.tensor(X_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_ub = torch.tensor(X_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_ub = torch.tensor(X_ub[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.ut = torch.tensor(ut).float().to(device)
        self.u_lb = torch.tensor(u_lb).float().to(device)
        self.u_ub = torch.tensor(u_ub).float().to(device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 0.001
        self.gamma = 0.999
        self.init = 1
        self.rba_eps = 1e-12

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_tt = grad(u_t, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_tt - u_xx + u ** 3 - source_torch(x, t)
        return f, u_t

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, ut_pred = self.net_r(self.x_r, self.t_r)
        _, ut0_pred = self.net_r(self.x_u, self.t_u)
        u_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred = self.net_u(self.x_ub, self.t_ub)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(self.r_pred).detach()
        r_norm = eta * abs_r_detach / (torch.max(abs_r_detach) + self.rba_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_u = torch.mean((self.u_pred - self.u) ** 2)
        loss_ut = torch.mean((ut0_pred - self.ut) ** 2)
        loss_b = 0.5 * (torch.mean((u_lb_pred - self.u_lb) ** 2) + torch.mean((u_ub_pred - self.u_ub) ** 2))
        
        self.loss = loss_r + loss_u + loss_ut + loss_b
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


class C_LoReW_PINN_Runner:
    def __init__(self, X_u, u, ut, X_r, X_lb, X_ub, u_lb, u_ub, lb, ub):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.x_lb = torch.tensor(X_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_lb = torch.tensor(X_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_ub = torch.tensor(X_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_ub = torch.tensor(X_ub[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.ut = torch.tensor(ut).float().to(device)
        self.u_lb = torch.tensor(u_lb).float().to(device)
        self.u_ub = torch.tensor(u_ub).float().to(device)
        self.N_r = self.x_r.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.eta = 5e-4
        self.gamma = 0.997
        self.init = 1

        self.alpha = 2.0
        self.epsilon = 20.0
        self.tau_gate = 0
        self.eta_g = 0.00001

        self.rho_g = 0.99
        self.gbar = torch.zeros_like(self.x_r).detach()

        self.kernel_h = 0.02
        self.kernel_p = 2.0
        self.kernel_trunc = 3.0
        self.num_kernel_centers = 41
        self.kernel_eps = 1e-12

        t_min_ = float(lb[1])
        t_max_ = float(ub[1])
        self.t_centers = torch.linspace(t_min_, t_max_, self.num_kernel_centers, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

        self.gate_abs_res_ref = None

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))

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
        local_scale = (local_moment_p + self.kernel_eps) ** (1.0 / self.kernel_p)
        return local_scale

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_tt = grad(u_t, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_tt - u_xx + u ** 3 - source_torch(x, t)
        return f, u_t

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, ut_pred = self.net_r(self.x_r, self.t_r)
        _, ut0_pred = self.net_r(self.x_u, self.t_u)
        u_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred = self.net_u(self.x_ub, self.t_ub)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta

        g_t = 0.5 * (1.0 - torch.tanh(self.alpha * (self.t_r - self.tau_gate)))
        self.gbar = (self.rho_g * self.gbar + (1.0 - self.rho_g) * g_t.detach()).detach()

        abs_r_detach = torch.abs(self.r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)

        causal_r_norm = r_norm * self.gbar
        self.rsum = (self.rsum * self.gamma + causal_r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)

        gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.kernel_eps)
        if self.gate_abs_res_ref is None:
            self.gate_abs_res_ref = gate_abs_res.item() + self.kernel_eps
        gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref

        self.tau_gate = self.tau_gate + self.eta_g * np.exp(-self.epsilon * gate_abs_res_norm)

        loss_u = torch.mean((self.u_pred - self.u) ** 2)
        loss_ut = torch.mean((ut0_pred - self.ut) ** 2)
        loss_b = 0.5 * (torch.mean((u_lb_pred - self.u_lb) ** 2) + torch.mean((u_ub_pred - self.u_ub) ** 2))
        
        self.loss = loss_r + loss_u + loss_ut + loss_b
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
    checkpoints = [5000, 20000, 40000]

    # Run SA-PINN
    print("Training SA-PINN and collecting weights...")
    sa_runner = SA_PINN_Runner(X_u, u_u, ut_u, X_f, X_lb, X_ub, u_lb, u_ub)
    sa_weights = sa_runner.train_and_collect_weights(checkpoints)

    # Run RBA-PINN
    print("Training RBA-PINN and collecting weights...")
    rba_runner = RBA_PINN_Runner(X_u, u_u, ut_u, X_f, X_lb, X_ub, u_lb, u_ub)
    rba_weights = rba_runner.train_and_collect_weights(checkpoints)

    # Run C-LoReW-PINN
    print("Training C-LoReW-PINN and collecting weights...")
    clorew_runner = C_LoReW_PINN_Runner(X_u, u_u, ut_u, X_f, X_lb, X_ub, u_lb, u_ub, lb, ub)
    clorew_weights = clorew_runner.train_and_collect_weights(checkpoints)

    # Plotting 3x3 Heatmap Grid
    print("Generating 3x3 interpolated heatmap grid...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    methods_data = [
        ("SA-PINN", sa_weights),
        ("RBA-PINN", rba_weights),
        ("C-LoReW-PINN (Ours)", clorew_weights)
    ]

    # X_f columns: col 0 is x, col 1 is t
    x_f_pts = X_f[:, 0]
    t_f_pts = X_f[:, 1]

    from scipy.interpolate import griddata

    for r_idx, (method_name, weights_dict) in enumerate(methods_data):
        for c_idx, iter_val in enumerate(checkpoints):
            ax = axes[r_idx, c_idx]
            w = weights_dict[iter_val]

            # Interpolate weights onto the regular grid
            w_grid = griddata((t_f_pts, x_f_pts), w, (T, X), method='linear')

            # Fill potential NaNs near boundaries using nearest neighbor interpolation
            nan_mask = np.isnan(w_grid)
            if np.any(nan_mask):
                w_grid_nearest = griddata((t_f_pts, x_f_pts), w, (T, X), method='nearest')
                w_grid[nan_mask] = w_grid_nearest[nan_mask]

            # Plot interpolated heatmap
            im = ax.imshow(
                w_grid.T,
                cmap='jet',
                aspect='auto',
                extent=[t_min, t_max, x_min, x_max],
                origin='lower'
            )

            # Overlay contour lines of the exact solution
            ax.contour(
                t0.flatten(),
                x0.flatten(),
                Exact0,
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
                ax.set_ylabel(f"{method_name}\nx", fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel("x", fontsize=12)
            ax.set_xlabel("t", fontsize=12)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, "kg_weight_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}!")
