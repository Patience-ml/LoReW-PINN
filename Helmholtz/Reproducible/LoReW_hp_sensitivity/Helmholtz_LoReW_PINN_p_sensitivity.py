import csv
import os
import time
from collections import OrderedDict

# Set before importing torch so cuBLAS can use deterministic kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.optim import lr_scheduler


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
#    Same setting as Helmholtz_LoReW_PINN.py.
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)


def log_message(message, train_log_path):
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


def format_tag(value):
    return str(value).replace('-', 'm').replace('.', 'p')


x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
nx, ny = 1001, 1001
xdim, ydim = 4.0, 1.0
ksq = 1.0
N_grid = 101
N_u = 50
N_r = N_grid * N_grid
first_opt = 30000
print_step = 100
layers = [2] + 4 * [50] + [1]

# Only kernel_p is changed in this sensitivity experiment.
# The original Helmholtz LoReW setting is h = 0.12, p = 1.5.
FIXED_KERNEL_H = 0.12
P_VALUES = [1.0 + 0.5 * i for i in range(7)]
HP_CONFIGS = [
    {
        'tag': f'p_sweep_p{format_tag(p)}',
        'kernel_h': FIXED_KERNEL_H,
        'kernel_p': p,
    }
    for p in P_VALUES
]

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
# 4. LoReW-PINN
# ==========================================
class PINN():
    def __init__(self, X_u, u, X_r, lb, ub, hp_config, train_log_path, savept=None):
        self.iter = 0
        self.exec_time = 0
        self.print_step = print_step
        self.savept = savept
        self.hp_config = hp_config
        self.train_log_path = train_log_path
        self.first_opt = first_opt
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
        self.kernel_h_x = hp_config['kernel_h']
        self.kernel_h_y = hp_config['kernel_h']
        self.kernel_p = hp_config['kernel_p']
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
                l_inf = np.linalg.norm(self.Exact.flatten() - sol.flatten(), np.inf)
                log_message(
                    'Iter %d, Loss: %.3e, Rel_L2: %.3e, L_inf: %.3e, h: %.3e, p: %.3e, Time: %s'
                    % (
                        self.iter,
                        self.loss.item(),
                        l2_rel,
                        l_inf,
                        self.kernel_h_x,
                        self.kernel_p,
                        format_elapsed_time(self.exec_time),
                    ),
                    self.train_log_path,
                )
                self.it.append(self.iter)
                self.l2.append(l2_rel)
                self.linf.append(l_inf)

        self.optimizer.step()
        self.losses.append(self.loss.item())

    def train(self, loss_file_name):
        self.dnn.train()
        for epoch in range(self.first_opt):
            start_time = time.time()
            self.loss_func()
            end_time = time.time()
            self.exec_time += (end_time - start_time)
            if (epoch + 1) % self.step_size == 0:
                self.scheduler.step()

        d = np.column_stack((np.array(self.it), np.array(self.l2), np.array(self.linf)))
        np.savetxt(os.path.join(LOG_DIR, loss_file_name), d, fmt='%.10f %.10f %.10f')
        if self.savept is not None:
            torch.save(self.dnn.state_dict(), os.path.join(MODEL_DIR, str(self.savept) + ".pt"))

    def predict_on_test_grid(self):
        self.dnn.eval()
        with torch.no_grad():
            u_pred = self.net_u(self.xx, self.yy)
        return tonp(u_pred).reshape(ny, nx).T


def run_one_config(hp_config):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tag = hp_config['tag']
    train_log_path = os.path.join(LOG_DIR, f'Helmholtz_LoReW_PINN_p_sensitivity_{tag}_train_log_{seed}.txt')
    loss_file_name = f'losses_helmholtz_lorew_p_sensitivity_{tag}_{seed}.txt'
    model_name = f'helmholtz_lorew_p_sensitivity_{tag}_model_{seed}'

    open(train_log_path, 'w', encoding='utf-8').close()
    log_message('=' * 70, train_log_path)
    log_message('Initializing LoReW-PINN p-sensitivity run...', train_log_path)
    log_message(
        'Config: h = %.5f, p = %.5f, seed = %d'
        % (hp_config['kernel_h'], hp_config['kernel_p'], seed),
        train_log_path,
    )
    log_message('Only kernel_p is changed from the base LoReW script; kernel_h_x/kernel_h_y are fixed at 0.12.', train_log_path)
    log_message('Starting training...', train_log_path)

    seed_torch(seed)
    model = PINN(X_u, u, X_r, lb, ub, hp_config, train_log_path, savept=model_name)
    model.train(loss_file_name)
    log_message('Training finished. Total time: %s' % format_elapsed_time(model.exec_time), train_log_path)

    sol = model.predict_on_test_grid()
    exact_flat = model.Exact.flatten()
    pred_flat = sol.flatten()
    final_l2_rel = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
    final_l_inf = np.linalg.norm(exact_flat - pred_flat, np.inf)

    log_message('-' * 50, train_log_path)
    log_message('Final test-set results', train_log_path)
    log_message(f'Final Relative L2 Error: {final_l2_rel:.5e}', train_log_path)
    log_message(f'Final L_inf Error:       {final_l_inf:.5e}', train_log_path)
    log_message('-' * 50, train_log_path)

    return {
        'tag': tag,
        'seed': seed,
        'kernel_h': hp_config['kernel_h'],
        'kernel_p': hp_config['kernel_p'],
        'final_l2_rel': final_l2_rel,
        'final_l_inf': final_l_inf,
        'total_time': format_elapsed_time(model.exec_time),
        'loss_file': loss_file_name,
        'model_file': model_name + '.pt',
        'train_log': os.path.basename(train_log_path),
    }


def save_summary(results):
    summary_path = os.path.join(LOG_DIR, f'helmholtz_lorew_p_sensitivity_summary_{seed}.csv')
    fieldnames = [
        'tag',
        'seed',
        'kernel_h',
        'kernel_p',
        'final_l2_rel',
        'final_l_inf',
        'total_time',
        'loss_file',
        'model_file',
        'train_log',
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(results, key=lambda item: item['final_l2_rel']):
            writer.writerow(row)

    print('=' * 70)
    print(f'Saved summary to: {summary_path}')
    print('Final Relative L2 Error for each p value:')
    for row in sorted(results, key=lambda item: item['kernel_p']):
        print(
            'p=%.5f | h=%.5f, final_L2RE=%.5e, final_Linf=%.5e, time=%s'
            % (
                row['kernel_p'],
                row['kernel_h'],
                row['final_l2_rel'],
                row['final_l_inf'],
                row['total_time'],
            )
        )
    min_row = min(results, key=lambda item: item['final_l2_rel'])
    print(
        'Lowest final error: p=%.5f, h=%.5f, final_L2RE=%.5e'
        % (
            min_row['kernel_p'],
            min_row['kernel_h'],
            min_row['final_l2_rel'],
        )
    )
    print('=' * 70)
    return summary_path


def plot_l2_comparison(results):
    plt.figure(figsize=(9, 5.5))
    for row in results:
        loss_path = os.path.join(LOG_DIR, row['loss_file'])
        if not os.path.exists(loss_path):
            continue
        data = np.loadtxt(loss_path)
        if data.ndim != 2 or data.shape[0] == 0:
            continue
        plt.plot(data[:, 0], data[:, 1], linewidth=1.4, label=row['tag'])

    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.45)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative $L^2$ Error', fontsize=12)
    plt.title(r'Helmholtz LoReW-PINN Sensitivity to $p$ ($h=0.12$)', fontsize=14)
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, f'Helmholtz_LoReW_p_sensitivity_L2_{seed}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved L2 comparison figure to: {output_path}')


# ==========================================
# 5. Main program
# ==========================================
if __name__ == "__main__":
    print('Running LoReW-PINN p-sensitivity on the Helmholtz equation.')
    print('All settings are inherited from Helmholtz_LoReW_PINN.py except kernel_p.')
    print(f'Fixed kernel_h_x = kernel_h_y = {FIXED_KERNEL_H}')
    print(f'Seed: {seed}')
    print(f'Number of configurations: {len(HP_CONFIGS)}')

    all_results = []
    for config_id, hp_config in enumerate(HP_CONFIGS, start=1):
        print('=' * 70)
        print(
            'Configuration %d/%d: %s (h=%.5f, p=%.5f)'
            % (
                config_id,
                len(HP_CONFIGS),
                hp_config['tag'],
                hp_config['kernel_h'],
                hp_config['kernel_p'],
            )
        )
        all_results.append(run_one_config(hp_config))

    save_summary(all_results)
    plot_l2_comparison(all_results)
