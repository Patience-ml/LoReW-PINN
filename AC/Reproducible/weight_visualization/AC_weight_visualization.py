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
from collections import OrderedDict
from scipy.interpolate import griddata

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.max_open_warning'] = 4

# ==========================================
# 1. Environment setup
# ==========================================
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
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

seed = 5656
seed_torch(seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def tonp(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    raise TypeError('Unknown type')

def grad(u, x):
    gradient = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    return gradient

# ==========================================
# 2. Data loading (AC.mat)
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.dirname(os.path.dirname(file_path))
FIGURE_DIR = os.path.join(data_path, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

try:
    data = scipy.io.loadmat(os.path.join(data_path, 'AC.mat'))
except FileNotFoundError:
    raise FileNotFoundError('Cannot find AC.mat.')

Exact = data['uu']
Exact0 = np.real(Exact)
t0 = data['tt'].flatten()[:, None]
x0 = data['x'].flatten()[:, None]
nx, nt = 256 * 2, 201

X, T = np.meshgrid(x0, t0)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

N_u = 256
N_f = 25600
IC_WEIGHT = 100.0

idx_x = np.random.choice(x0.shape[0], N_u, replace=False)
x_u = x0[idx_x, :]
t_u = np.zeros((N_u, 1))
u_u = Exact0[idx_x, 0:1]
X_u = np.hstack((x_u, t_u))
u = u_u

lb = X_star.min(0)
ub = X_star.max(0)
X_f = lb + (ub - lb) * lhs(2, N_f, random_state=seed)
X_r_np = np.vstack((X_f, X_u))

# ==========================================
# 3. Network definition
# ==========================================
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh()

        layer_list = []
        for i in range(self.depth - 1):
            w_layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(w_layer.weight)
            layer_list.append(('layer_%d' % i, w_layer))
            layer_list.append(('activation_%d' % i, self.activation))

        w_layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(w_layer.weight)
        layer_list.append(('layer_%d' % (self.depth - 1), w_layer))

        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)

class Fourier(torch.nn.Module):
    def __init__(self, layers):
        super(Fourier, self).__init__()
        self.L, self.M = 2.0, 10
        fourier_layers = list(layers)
        fourier_layers[0] = 2 * self.M + 2
        self.model = DNN(fourier_layers)
        self.register_buffer('k', torch.arange(1, self.M + 1).float().view(1, -1))

    def encoding(self, x, t):
        w = 2.0 * np.pi / self.L
        return torch.hstack([
            torch.cos(self.k * w * x),
            torch.sin(self.k * w * x),
            t,
            torch.ones_like(t),
        ])

    def forward(self, h):
        x = h[:, 0:1]
        t = h[:, 1:2]
        h = self.encoding(x, t)
        return self.model(h)

layers = [2] + 6 * [128] + [1]
dimx = N_u
dimt = N_u

# ==========================================
# 4. Runners
# ==========================================
class SA_PINN_Runner:
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

        seed_torch(seed)
        self.dnn = Fourier(layers).to(device)
        lamr = torch.rand(self.N_r, 1, requires_grad=True).float().to(device) * 1.0
        lamu = torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 100.0
        self.lamr = torch.nn.Parameter(lamr)
        self.lamu = torch.nn.Parameter(lamu)
        self.optimizer2 = torch.optim.Adam([self.lamr, self.lamu], lr=0.005, maximize=True)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t - 0.0001 * u_xx + 5.0 * u * u * u - 5.0 * u
        return f, u_x

    def loss_func(self):
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, _ = self.net_r(self.x_r, self.t_r)

        loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
        loss_u = torch.mean((self.lamu * (self.u_pred[:self.N_u] - self.u)) ** 2)

        self.loss = loss_r + IC_WEIGHT * loss_u
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
    def __init__(self, X_u, u, X_r):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

        seed_torch(seed)
        self.dnn = Fourier(layers).to(device)

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
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t - 0.0001 * u_xx + 5.0 * u * u * u - 5.0 * u
        return f, u_x

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, _ = self.net_r(self.x_r, self.t_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(self.r_pred).detach()
        r_norm = eta * abs_r_detach / (torch.max(abs_r_detach) + self.rba_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_u = IC_WEIGHT * torch.mean((self.u_pred[:self.N_u] - self.u) ** 2)

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
        return collected_weights


class C_LoReW_PINN_Runner:
    def __init__(self, X_u, u, X_r, lb, ub):
        self.iter = 0
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

        seed_torch(seed)
        self.dnn = Fourier(layers).to(device)

        self.rsum = 0
        self.eta = 0.001
        self.gamma = 0.999
        self.init = 1

        self.alpha = 5.0
        self.epsilon = 5.0
        self.tau_gate = -0.5
        self.eta_g = 0.0001
        self.rho_g = 0.95
        self.gbar = torch.zeros_like(self.x_r).detach()

        self.kernel_h = 0.05
        self.kernel_p = 2.0
        self.kernel_trunc = 3.0
        self.num_kernel_centers = 41
        self.kernel_eps = 1e-12

        t_min_val = float(lb[1])
        t_max_val = float(ub[1])
        self.t_centers = torch.linspace(t_min_val, t_max_val, self.num_kernel_centers, device=device).view(1, -1)
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
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t - 0.0001 * u_xx + 5.0 * u * u * u - 5.0 * u
        return f, u_x

    def loss_func(self):
        self.optimizer.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, _ = self.net_r(self.x_r, self.t_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta

        g_t = 0.5 * (1.0 - torch.tanh(self.alpha * (self.t_r - self.tau_gate)))
        self.gbar = (self.rho_g * self.gbar + (1.0 - self.rho_g) * g_t.detach()).detach()

        abs_r_detach = torch.abs(self.r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)

        causal_r_norm = r_norm * self.gbar
        self.rsum = (self.rsum * self.gamma + causal_r_norm).detach()

        loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        loss_u = IC_WEIGHT * torch.mean((self.u_pred[:self.N_u] - self.u) ** 2)

        gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.kernel_eps)
        if self.gate_abs_res_ref is None:
            self.gate_abs_res_ref = gate_abs_res.item() + self.kernel_eps
        gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref
        self.tau_gate = self.tau_gate + self.eta_g * np.exp(-self.epsilon * gate_abs_res_norm)

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
        return collected_weights

# ==========================================
# 5. Training execution and plotting
# ==========================================
if __name__ == "__main__":
    checkpoints = [10000, 40000, 80000]

    print("Training SA-PINN and collecting weights...")
    sa_runner = SA_PINN_Runner(X_u, u, X_r_np)
    sa_weights = sa_runner.train_and_collect_weights(checkpoints)

    print("Training RBA-PINN and collecting weights...")
    rba_runner = RBA_PINN_Runner(X_u, u, X_r_np)
    rba_weights = rba_runner.train_and_collect_weights(checkpoints)

    print("Training C-LoReW-PINN and collecting weights...")
    clorew_runner = C_LoReW_PINN_Runner(X_u, u, X_r_np, lb, ub)
    clorew_weights = clorew_runner.train_and_collect_weights(checkpoints)

    print("Generating 3x3 interpolated heatmap grid...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    methods_data = [
        ("SA-PINN", sa_weights),
        ("RBA-PINN", rba_weights),
        ("C-LoReW-PINN (Ours)", clorew_weights)
    ]

    x_f_pts = X_r_np[:, 0]
    t_f_pts = X_r_np[:, 1]
    
    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0

    for r_idx, (method_name, weights_dict) in enumerate(methods_data):
        for c_idx, iter_val in enumerate(checkpoints):
            ax = axes[r_idx, c_idx]
            w = weights_dict[iter_val]

            w_grid = griddata((t_f_pts, x_f_pts), w, (T, X), method='linear')

            nan_mask = np.isnan(w_grid)
            if np.any(nan_mask):
                w_grid_nearest = griddata((t_f_pts, x_f_pts), w, (T, X), method='nearest')
                w_grid[nan_mask] = w_grid_nearest[nan_mask]

            im = ax.imshow(
                w_grid.T,
                cmap='jet',
                aspect='auto',
                extent=[t_min, t_max, x_min, x_max],
                origin='lower'
            )

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

            if r_idx == 0:
                ax.set_title(f"Iteration {iter_val}", fontsize=14, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f"{method_name}\nx", fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel("x", fontsize=12)
            ax.set_xlabel("t", fontsize=12)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, "ac_weight_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}!")
