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
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)
file_name = f'losses_helmholtz_lorew_pinn_{seed}.txt'
train_log_name = f"{os.path.splitext(os.path.basename(__file__))[0]}_train_log_{seed}.txt"
train_log_path = os.path.join(LOG_DIR, train_log_name)


def log_message(message):
    print(message)
    with open(train_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(str(message) + '\n')


def format_elapsed_time(seconds):
    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f'{hours}h{minutes}min{seconds}s'
    if minutes > 0:
        return f'{minutes}min{seconds}s'
    return f'{seconds}s'


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

# Dirichlet boundary points: 50 equally spaced points on each boundary.
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
# 4. LoReW-PINN
# ==========================================
class PINN():
    def __init__(self, X_u, u, X_r, lb, ub, savept=None):
        self.iter = 0
        self.exec_time = 0
        self.print_step = 100
        self.savept = savept
        self.first_opt = 30000
        self.bc_weight = 1
        self.it, self.l2, self.linf = [], [], []
        self.loss, self.losses = None, []

        self.Exact = Exact
        self.xx = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        self.yy = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)

        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

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
        x_min_ = float(lb[0])
        x_max_ = float(ub[0])
        y_min_ = float(lb[1])
        y_max_ = float(ub[1])
        self.x_centers = torch.linspace(x_min_, x_max_, self.num_kernel_centers_x, device=device).view(1, -1)
        self.y_centers = torch.linspace(y_min_, y_max_, self.num_kernel_centers_y, device=device).view(1, -1)
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

        if self.iter % self.print_step == 0:
            with torch.no_grad():
                res = self.net_u(self.xx, self.yy)
                sol = tonp(res).reshape(ny, nx).T
                l2_rel = np.linalg.norm(self.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(self.Exact.flatten(), 2)
                l_inf = np.linalg.norm(self.Exact.flatten() - sol.flatten(), np.inf) / np.linalg.norm(self.Exact.flatten(), np.inf)
                log_message('Iter %d, Loss: %.3e, Rel_L2: %.3e, L_inf: %.3e, Time: %s' %
                            (self.iter, self.loss.item(), l2_rel, l_inf, format_elapsed_time(self.exec_time)))
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

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(X[:, 1:2], dtype=torch.float32, device=device)
        self.dnn.eval()
        return tonp(self.net_u(x, y))


# ==========================================
# 5. Main program and visualization
# ==========================================
if __name__ == "__main__":
    open(train_log_path, "w", encoding="utf-8").close()
    log_message('Initializing LoReW-PINN for the Helmholtz equation...')

    model = PINN(X_u, u, X_r, lb, ub, savept=f"helmholtz_lorew_pinn_model_{seed}")

    log_message('Starting training...')
    model.train()
    log_message('Training finished. Total time: %s' % format_elapsed_time(model.exec_time))

    model.dnn.eval()
    with torch.no_grad():
        u_pred = model.net_u(model.xx, model.yy)
        sol = tonp(u_pred).reshape(ny, nx).T

    final_l2_rel = np.linalg.norm(model.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(model.Exact.flatten(), 2)
    final_l_inf = np.linalg.norm(model.Exact.flatten() - sol.flatten(), np.inf) / np.linalg.norm(model.Exact.flatten(), np.inf)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    im0 = ax[0].imshow(model.Exact, cmap='rainbow', aspect='auto', extent=[y_min, y_max, x_min, x_max], origin='lower')
    ax[0].set_title('Exact u(x,y)', fontsize=14)
    ax[0].set_xlabel('y', fontsize=12)
    ax[0].set_ylabel('x', fontsize=12)
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(sol, cmap='rainbow', aspect='auto', extent=[y_min, y_max, x_min, x_max], origin='lower')
    ax[1].set_title('Predicted u(x,y) (LoReW-PINN)', fontsize=14)
    ax[1].set_xlabel('y', fontsize=12)
    ax[1].set_ylabel('x', fontsize=12)
    fig.colorbar(im1, ax=ax[1])

    error = np.abs(model.Exact - sol)
    im2 = ax[2].imshow(error, cmap='rainbow', aspect='auto', extent=[y_min, y_max, x_min, x_max], origin='lower')
    ax[2].set_title('Absolute Error', fontsize=14)
    ax[2].set_xlabel('y', fontsize=12)
    ax[2].set_ylabel('x', fontsize=12)
    fig.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"Helmholtz_LoReW_PINN_Prediction_Results_{seed}.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(model.it, model.l2, label='Relative L2 Error', color='blue', linewidth=1.5)
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative $L^2$ Error', fontsize=12)
    plt.title('Convergence History (LoReW-PINN)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"Helmholtz_LoReW_PINN_Loss_History_{seed}.png"), dpi=300)
    plt.show()

    log_message('-' * 50)
    log_message('Final test-set results')
    log_message(f'Final Relative L2 Error: {final_l2_rel:.5e}')
    log_message(f'Final L_inf Error:       {final_l_inf:.5e}')
    log_message('-' * 50)
