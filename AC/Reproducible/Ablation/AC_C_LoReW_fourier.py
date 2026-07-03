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

seed = 1234
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
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return gradient

# ==========================================
# 2. Data loading (AC.mat)
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)
data_path = os.path.dirname(os.path.dirname(file_path))
file_name = f'losses_clorew_pinn_fourier_{seed}.txt'
train_log_name = f"{os.path.splitext(os.path.basename(__file__))[0]}_train_log_{seed}.txt"
train_log_path = os.path.join(LOG_DIR, train_log_name)

def format_elapsed_time(seconds):
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f'{hours}h{minutes}min{seconds}s'
    if minutes > 0:
        return f'{minutes}min{seconds}s'
    return f'{seconds}s'

def log_message(message):
    print(message)
    with open(train_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(str(message) + '\n')

try:
    data = scipy.io.loadmat(os.path.join(data_path, 'AC.mat'))
except FileNotFoundError:
    raise FileNotFoundError('Cannot find AC.mat. Please make sure it is in the parent AC folder.')

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

# Boundary and initial-condition sampling
idx_x = np.random.choice(x0.shape[0], N_u, replace=False)
x_u = x0[idx_x, :]
t_u = np.zeros((N_u, 1))
u_u = Exact0[idx_x, 0:1]
X_u = np.hstack((x_u, t_u))
u = u_u

# Global PDE collocation points sampling
lb = X_star.min(0)
ub = X_star.max(0)
X_f = lb + (ub - lb) * lhs(2, N_f, random_state=seed)
X_f = np.vstack((X_f, X_u))

# ==========================================
# 3. Network definition (Fourier features + vanilla MLP)
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

        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

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

# ==========================================
# 4. PINN with Fourier features and Kernel-Causal-Residual coupling
#    Keep the original code style as much as possible:
#    gate * residual-normalization -> rsum -> weighted residual loss
# ==========================================
class PINN():
    def __init__(self, X_u, u, X_r, lb, ub, dimx, dimt, savept=None):
        self.rba = 1
        self.iter = 0
        self.exec_time = 0
        self.print_step = 100
        self.savept = savept
        self.dimx, self.dimt = dimx, dimt
        self.dimx_, self.dimt_ = nx, nt
        self.first_opt = 300000
        self.it, self.l2, self.linf = [], [], []
        self.loss, self.losses = None, []

        self.Exact = Exact0
        X, T = np.meshgrid(x0, t0)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.xx = torch.tensor(X_star[:, 0:1]).float().to(device)
        self.tt = torch.tensor(X_star[:, 1:2]).float().to(device)

        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_u = dimt
        self.ic_weight = IC_WEIGHT

        self.dnn = Fourier(layers).to(device)

        if self.rba == 1:
            # --- Keep the original update style and main parameters ---
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
            self.init = 1

            # Causal-gate parameters (kept from the original code)
            self.alpha = 5.0
            self.epsilon = 5.0
            self.tau_gate = -0.5
            self.eta_g = 0.0001

            # New 1: EMA-smoothed causal gate
            self.rho_g = 0.95
            self.gbar = torch.zeros_like(self.x_r).detach()

            # New 2: kernel-local residual normalization
            # Use temporal kernel centers to avoid O(N^2) memory cost.
            self.kernel_h = 0.05
            self.kernel_p = 2.0
            self.kernel_trunc = 3.0
            self.num_kernel_centers = 41
            self.kernel_eps = 1e-12

            t_min = float(lb[1])
            t_max = float(ub[1])
            self.t_centers = torch.linspace(t_min, t_max, self.num_kernel_centers, device=device).view(1, -1)
            self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
            self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

            # New 3: gate-weighted mean absolute residual for tau update
            self.gate_abs_res_ref = None

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.step_size = 5000

    def build_kernel_weights(self, t_points):
        diff = t_points - self.t_centers
        K = torch.exp(-0.5 * (diff / self.kernel_h) ** 2)
        if self.kernel_trunc is not None and self.kernel_trunc > 0:
            K = K * (torch.abs(diff) <= self.kernel_trunc * self.kernel_h).float()

        # Fallback in case a row becomes all-zero after truncation
        row_sum = K.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            nearest_idx = torch.argmin(torch.abs(diff[zero_rows]), dim=1)
            K[zero_rows] = 0.0
            K[zero_rows, nearest_idx] = 1.0
            row_sum = K.sum(dim=1, keepdim=True)

        return K / (row_sum + self.kernel_eps)

    def kernel_local_scale(self, abs_r_detach):
        # Center-wise weighted p-th moment
        center_moment_p = (self.kernel_weights.T @ (abs_r_detach ** self.kernel_p)) / self.kernel_col_mass
        # Interpolate back to each point using the same kernel weights
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

        if self.rba == 1:
            eta = 1 if self.init == 2 and self.iter == 0 else self.eta

            # 1) Instantaneous causal gate
            g_t = 0.5 * (1.0 - torch.tanh(self.alpha * (self.t_r - self.tau_gate)))

            # 2) EMA-smoothed gate
            self.gbar = (self.rho_g * self.gbar + (1.0 - self.rho_g) * g_t.detach()).detach()

            # 3) Kernel-local residual normalization
            abs_r = torch.abs(self.r_pred)
            abs_r_detach = abs_r.detach()
            local_scale = self.kernel_local_scale(abs_r_detach)
            r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)

            # 4) Keep the original multiplicative coupling
            causal_r_norm = r_norm * self.gbar

            # 5) Keep the original EMA accumulation style
            self.rsum = (self.rsum * self.gamma + causal_r_norm).detach()

            # 6) Keep the original weighted-residual loss form
            loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
            loss_u = self.ic_weight * torch.mean((self.u_pred[:self.dimx] - self.u) ** 2)

            # 7) New tau-gate update: gate-weighted mean absolute residual
            gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.kernel_eps)
            if self.gate_abs_res_ref is None:
                self.gate_abs_res_ref = gate_abs_res.item() + self.kernel_eps
            gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref

            self.tau_gate = self.tau_gate + self.eta_g * np.exp(-self.epsilon * gate_abs_res_norm)
        else:
            loss_r = torch.mean(self.r_pred ** 2)
            loss_u = self.ic_weight * torch.mean((self.u_pred[:self.dimx] - self.u) ** 2)


        self.loss = loss_r + loss_u
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:
            with torch.no_grad():
                res = self.net_u(self.xx, self.tt)
                sol = tonp(res)
                sol = np.reshape(sol, (self.dimt_, self.dimx_)).T

                l2_rel = np.linalg.norm(self.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(self.Exact.flatten(), 2)
                l_inf = np.linalg.norm(self.Exact.flatten() - sol.flatten(), np.inf) / np.linalg.norm(self.Exact.flatten(), np.inf)
                gbar_mean = float(torch.mean(self.gbar).item())
                local_scale_mean = float(torch.mean(self.kernel_local_scale(torch.abs(self.r_pred).detach())).item())
                gate_abs_res_log = float((torch.sum(self.gbar * torch.abs(self.r_pred).detach()) / (torch.sum(self.gbar) + self.kernel_eps)).item())

                log_message(
                    'Iter %d, Loss: %.3e, Rel_L2: %.3e, L_inf: %.3e, tau_gate: %.3f, gbar_mean: %.3e, local_scale_mean: %.3e, gate_abs_res: %.3e, Time: %s'
                    % (self.iter, self.loss.item(), l2_rel, l_inf, self.tau_gate, gbar_mean, local_scale_mean, gate_abs_res_log, format_elapsed_time(self.exec_time))
                )

                self.it.append(self.iter)
                self.l2.append(l2_rel)
                self.linf.append(l_inf)

        self.optimizer.step()
        self.losses.append(self.loss.item())

    def train(self):
        self.dnn.train()
        for epoch in range(self.first_opt):
            start_time = time.time()
            self.loss_func()
            end_time = time.time()
            self.exec_time += (end_time - start_time)
            if (epoch + 1) % self.step_size == 0:
                self.scheduler.step()

        d = np.column_stack((np.array(self.it), np.array(self.l2), np.array(self.linf)))
        np.savetxt(os.path.join(LOG_DIR, file_name), d, fmt='%.10f %.10f %.10f')
        if self.savept is not None:
            torch.save(self.dnn.state_dict(), os.path.join(MODEL_DIR, str(self.savept) + ".pt"))

# ==========================================
# 5. Main program and visualization
# ==========================================
if __name__ == "__main__":
    open(train_log_path, "w", encoding="utf-8").close()
    log_message("Initializing Fourier-feature MLP with C-LoReW-PINN...")
    log_message(f"Initial-condition samples N_u = {N_u}")
    log_message(f"Initial-condition loss weight = {IC_WEIGHT:g}")
    layers = [2] + 6 * [128] + [1]
    dimx = N_u
    dimt = N_u

    model = PINN(X_u, u, X_f, lb, ub, dimx, dimt, savept=f"ac_clorew_fourier_model_{seed}")

    log_message("Starting training...")
    model.train()
    log_message('Training finished. Total time: %s' % format_elapsed_time(model.exec_time))

    model.dnn.eval()
    with torch.no_grad():
        u_pred = model.net_u(model.xx, model.tt)
        u_pred_np = tonp(u_pred)
        sol = np.reshape(u_pred_np, (model.dimt_, model.dimx_)).T

    exact_flat = model.Exact.flatten()
    pred_flat = sol.flatten()
    final_l2_rel = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
    final_l_inf = np.linalg.norm(exact_flat - pred_flat, np.inf) / np.linalg.norm(exact_flat, np.inf)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    im0 = ax[0].imshow(model.Exact, cmap='rainbow', aspect='auto', extent=[0, 1, -1, 1], origin='lower')
    ax[0].set_title('Exact u(x,t)', fontsize=14)
    ax[0].set_xlabel('t', fontsize=12)
    ax[0].set_ylabel('x', fontsize=12)
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(sol, cmap='rainbow', aspect='auto', extent=[0, 1, -1, 1], origin='lower')
    ax[1].set_title('Predicted u(x,t) (C-LoReW-Fourier)', fontsize=14)
    ax[1].set_xlabel('t', fontsize=12)
    ax[1].set_ylabel('x', fontsize=12)
    fig.colorbar(im1, ax=ax[1])

    error = np.abs(model.Exact - sol)
    im2 = ax[2].imshow(error, cmap='rainbow', aspect='auto', extent=[0, 1, -1, 1], origin='lower')
    ax[2].set_title('Absolute Error', fontsize=14)
    ax[2].set_xlabel('t', fontsize=12)
    ax[2].set_ylabel('x', fontsize=12)
    fig.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'AC_C_LoReW_PINN_Fourier_Prediction_Results_{seed}.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(model.it, model.l2, label='Relative L2 Error', color='blue', linewidth=1.5)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative $L^2$ Error', fontsize=12)
    plt.title('Convergence History (C-LoReW-Fourier)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'AC_C_LoReW_PINN_Fourier_Loss_History_{seed}.png'), dpi=300)
    plt.show()

    log_message("-" * 50)
    log_message("Final test-set results")
    log_message(f"Final Relative L2 Error: {final_l2_rel:.5e}")
    log_message(f"Final L_inf Error:       {final_l_inf:.5e}")
    log_message("-" * 50)

