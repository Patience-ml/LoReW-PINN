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
from pyDOE import lhs


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


seed = 2138
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
        create_graph=True
    )[0]


def exact_u_np(x, y):
    return np.cos(C * x) + np.cos(C * y)


def source_torch(x, y):
    return 2.0 * (C ** 2) * (torch.cos(C * x) + torch.cos(C * y))


# ==========================================
# 2. Data generation with analytical solution
#    Same setting as Poisson_LoReW_PINN.py.
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)


def format_elapsed_time(seconds):
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f'{hours}h{minutes}min{seconds}s'
    if minutes > 0:
        return f'{minutes}min{seconds}s'
    return f'{seconds}s'


def log_message(message, train_log_path):
    print(message)
    with open(train_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(str(message) + '\n')


def format_tag(value):
    return str(value).replace('-', 'm').replace('.', 'p')


C = 10.0
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
nx, ny = 201, 201
N_f = 10000
N_b_each = 100
first_opt = 50000
print_step = 100
learning_rate = 1e-3
layers = [2] + 8 * [50] + [1]

# Only kernel_h_x/kernel_h_y are changed in this sensitivity experiment.
# The original Poisson LoReW setting is h = 0.05, p = 2.0.
FIXED_KERNEL_P = 2.0
H_VALUES = [0.02 + 0.02 * i for i in range(7)]
HP_CONFIGS = [
    {
        'tag': f'h_sweep_h{format_tag(h)}',
        'kernel_h': h,
        'kernel_p': FIXED_KERNEL_P,
    }
    for h in H_VALUES
]

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
# 4. PINN with 2D Kernel-Residual coupling
# ==========================================
class PINN():
    def __init__(self, X_b, u_b, X_r, lb, ub, hp_config, train_log_path, savept=None):
        self.rba = 1
        self.iter = 0
        self.exec_time = 0
        self.print_step = print_step
        self.savept = savept
        self.hp_config = hp_config
        self.train_log_path = train_log_path
        self.dimx_, self.dimy_ = nx, ny
        self.first_opt = first_opt
        self.it, self.l2, self.linf = [], [], []
        self.loss, self.losses = None, []
        self.best_l2_rel = np.inf
        self.best_l_inf = np.inf
        self.best_iter = 0
        self.best_state_dict = None

        self.Exact = Exact
        self.xx = torch.tensor(X_star[:, 0:1]).float().to(device)
        self.yy = torch.tensor(X_star[:, 1:2]).float().to(device)
        self.x_b = torch.tensor(X_b[:, 0:1], requires_grad=True).float().to(device)
        self.y_b = torch.tensor(X_b[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.y_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u_b = torch.tensor(u_b).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_b = self.x_b.shape[0]

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)
        if self.rba == 1:
            self.rsum = 0
            self.eta = 2.5e-4
            self.gamma = 0.997
            self.init = 1

            self.kernel_h_x = hp_config['kernel_h']
            self.kernel_h_y = hp_config['kernel_h']
            self.kernel_p = hp_config['kernel_p']
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
        local_scale = (local_moment_p + self.kernel_eps) ** (1.0 / self.kernel_p)
        return local_scale

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

        if self.rba == 1:
            eta = 1 if self.init == 2 and self.iter == 0 else self.eta
            abs_r_detach = torch.abs(self.r_pred).detach()
            local_scale = self.kernel_local_scale(abs_r_detach)
            r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
        else:
            loss_r = torch.mean(self.r_pred ** 2)

        loss_b = torch.mean((self.u_b_pred - self.u_b) ** 2)
        self.loss = loss_r + loss_b
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:
            with torch.no_grad():
                res = self.net_u(self.xx, self.yy)
                sol = np.reshape(tonp(res), (self.dimx_, self.dimy_))
                l2_rel = np.linalg.norm(self.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(self.Exact.flatten(), 2)
                l_inf = np.linalg.norm(self.Exact.flatten() - sol.flatten(), np.inf)
                local_scale_mean = float(torch.mean(self.kernel_local_scale(torch.abs(self.r_pred).detach())).item())
                rsum_mean = float(torch.mean(self.rsum).item())

                log_message(
                    'Iter %d, Loss: %.3e, Rel_L2: %.3e, L_inf: %.3e, h: %.3e, p: %.3e, local_scale_mean: %.3e, rsum_mean: %.3e, Time: %s'
                    % (
                        self.iter,
                        self.loss.item(),
                        l2_rel,
                        l_inf,
                        self.kernel_h_x,
                        self.kernel_p,
                        local_scale_mean,
                        rsum_mean,
                        format_elapsed_time(self.exec_time),
                    ),
                    self.train_log_path,
                )
                self.it.append(self.iter)
                self.l2.append(l2_rel)
                self.linf.append(l_inf)
                if l2_rel < self.best_l2_rel:
                    self.best_l2_rel = l2_rel
                    self.best_l_inf = l_inf
                    self.best_iter = self.iter
                    self.best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self.dnn.state_dict().items()
                    }

        self.optimizer.step()
        self.losses.append(self.loss.item())

    def train(self, loss_file_name):
        self.dnn.train()
        for _ in range(self.first_opt):
            start_time = time.time()
            self.loss_func()
            end_time = time.time()
            self.exec_time += (end_time - start_time)

        d = np.column_stack((np.array(self.it), np.array(self.l2), np.array(self.linf)))
        np.savetxt(os.path.join(LOG_DIR, loss_file_name), d, fmt='%.10f %.10f %.10f')
        if self.savept is not None:
            torch.save(self.dnn.state_dict(), os.path.join(MODEL_DIR, str(self.savept) + ".pt"))
            if self.best_state_dict is not None:
                torch.save(self.best_state_dict, os.path.join(MODEL_DIR, str(self.savept) + "_best.pt"))

    def predict_on_test_grid(self):
        self.dnn.eval()
        with torch.no_grad():
            u_pred = self.net_u(self.xx, self.yy)
        return np.reshape(tonp(u_pred), (self.dimx_, self.dimy_))


def run_one_config(hp_config):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tag = hp_config['tag']
    train_log_path = os.path.join(LOG_DIR, f'Poisson_LoReW_PINN_h_sensitivity_{tag}_train_log_{seed}.txt')
    loss_file_name = f'losses_poisson_lorew_h_sensitivity_{tag}_{seed}.txt'
    model_name = f'poisson_lorew_h_sensitivity_{tag}_model_{seed}'

    open(train_log_path, 'w', encoding='utf-8').close()
    log_message('=' * 70, train_log_path)
    log_message('Initializing LoReW-PINN h-sensitivity run...', train_log_path)
    log_message(
        'Config: h = %.5f, p = %.5f, seed = %d'
        % (hp_config['kernel_h'], hp_config['kernel_p'], seed),
        train_log_path,
    )
    log_message('Only kernel_h_x/kernel_h_y are changed from the base LoReW script; kernel_p is fixed at 2.0.', train_log_path)
    log_message('Starting training...', train_log_path)

    seed_torch(seed)
    model = PINN(X_b, u_b, X_f, lb, ub, hp_config, train_log_path, savept=model_name)
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
    log_message('Best checkpoint results', train_log_path)
    log_message(f'Best Iteration:          {model.best_iter}', train_log_path)
    log_message(f'Best Relative L2 Error:  {model.best_l2_rel:.5e}', train_log_path)
    log_message(f'Best L_inf Error:        {model.best_l_inf:.5e}', train_log_path)
    log_message('-' * 50, train_log_path)

    return {
        'tag': tag,
        'seed': seed,
        'kernel_h': hp_config['kernel_h'],
        'kernel_p': hp_config['kernel_p'],
        'final_l2_rel': final_l2_rel,
        'final_l_inf': final_l_inf,
        'best_l2_rel': model.best_l2_rel,
        'best_l_inf': model.best_l_inf,
        'best_iter': model.best_iter,
        'total_time': format_elapsed_time(model.exec_time),
        'loss_file': loss_file_name,
        'model_file': model_name + '.pt',
        'best_model_file': model_name + '_best.pt',
        'train_log': os.path.basename(train_log_path),
    }


def save_summary(results):
    summary_path = os.path.join(LOG_DIR, f'poisson_lorew_h_sensitivity_summary_{seed}.csv')
    fieldnames = [
        'tag',
        'seed',
        'kernel_h',
        'kernel_p',
        'final_l2_rel',
        'final_l_inf',
        'best_l2_rel',
        'best_l_inf',
        'best_iter',
        'total_time',
        'loss_file',
        'model_file',
        'best_model_file',
        'train_log',
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(results, key=lambda item: item['best_l2_rel']):
            writer.writerow(row)

    print('=' * 70)
    print(f'Saved summary to: {summary_path}')
    print('Best Relative L2 Error for each h value:')
    for row in sorted(results, key=lambda item: item['kernel_h']):
        print(
            'h=%.5f | p=%.5f, best_L2RE=%.5e, best_iter=%d, final_L2RE=%.5e, time=%s'
            % (
                row['kernel_h'],
                row['kernel_p'],
                row['best_l2_rel'],
                row['best_iter'],
                row['final_l2_rel'],
                row['total_time'],
            )
        )
    best_row = min(results, key=lambda item: item['best_l2_rel'])
    print(
        'Best overall: h=%.5f, p=%.5f, best_L2RE=%.5e at iter %d'
        % (
            best_row['kernel_h'],
            best_row['kernel_p'],
            best_row['best_l2_rel'],
            best_row['best_iter'],
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
    plt.title(r'Poisson LoReW-PINN Sensitivity to $h$ ($p=2.0$)', fontsize=14)
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, f'Poisson_LoReW_h_sensitivity_L2_{seed}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved L2 comparison figure to: {output_path}')


# ==========================================
# 5. Main program
# ==========================================
if __name__ == "__main__":
    print('Running LoReW-PINN h-sensitivity on the Poisson equation.')
    print('All settings are inherited from Poisson_LoReW_PINN.py except kernel_h_x/kernel_h_y.')
    print(f'Fixed kernel_p = {FIXED_KERNEL_P}')
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
