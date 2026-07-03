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


def exact_u_np(x, t):
    return np.exp(-t) * (
        np.sin(x)
        + 0.5 * np.sin(2.0 * x)
        + np.sin(3.0 * x) / 3.0
        + 0.25 * np.sin(4.0 * x)
        + 0.125 * np.sin(8.0 * x)
    )


def reaction_torch(x, t):
    return torch.exp(-t) * (
        1.5 * torch.sin(2.0 * x)
        + (8.0 / 3.0) * torch.sin(3.0 * x)
        + (15.0 / 4.0) * torch.sin(4.0 * x)
        + (63.0 / 8.0) * torch.sin(8.0 * x)
    )


# ==========================================
# 2. Data generation with analytical solution
# ==========================================
file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)
file_name = f'losses_rd_sa_{seed}.txt'
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


D = 1.0
x_min, x_max = -np.pi, np.pi
t_min, t_max = 0.0, 1.0
nx, nt = 256 * 2, 201
N_i = nx
N_b = nt
N_u = N_i + 2 * N_b
N_f = 140

x0 = np.linspace(x_min, x_max, nx)[:, None]
t0 = np.linspace(t_min, t_max, nt)[:, None]

X, T = np.meshgrid(x0.flatten(), t0.flatten())
Exact = exact_u_np(X, T)
Exact0 = Exact.T

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# Dirichlet initial and boundary points: use every equidistant grid point.
X_i = np.hstack((X[0:1, :].T, T[0:1, :].T))
u_i = Exact[0:1, :].T
X_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
u_lb = Exact[:, 0:1]
X_ub = np.hstack((X[:, -1:], T[:, -1:]))
u_ub = Exact[:, -1:]
X_u = np.vstack([X_i, X_lb, X_ub])
u = np.vstack([u_i, u_lb, u_ub])

# PDE residual points
lb = X_star.min(0)
ub = X_star.max(0)
X_f = lb + (ub - lb) * lhs(2, N_f, random_state=seed)
X_f = np.vstack((X_f, X_u))


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
# 4. PINN with SA-PINN weighting
# ==========================================
class PINN():
    def __init__(self, X_u, u, X_r, lb, ub, savept=None):
        self.sa = 1
        self.iter = 0
        self.exec_time = 0
        self.print_step = 100
        self.savept = savept
        self.first_opt = 30000
        self.it, self.l2, self.linf = [], [], []
        self.loss, self.losses = None, []

        self.Exact = Exact0
        self.xx = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        self.tt = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)

        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

        self.dnn = DNN(layers).to(device)
        if self.sa == 1:
            lamr = torch.rand(self.N_r, 1, requires_grad=True).float().to(device) * 1.0
            lamu = torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 100.0
            self.lamr = torch.nn.Parameter(lamr)
            self.lamu = torch.nn.Parameter(lamu)
            self.optimizer2 = torch.optim.Adam([self.lamr, self.lamu], lr=0.005, maximize=True)

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.step_size = 5000

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_t - D * u_xx - reaction_torch(x, t)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()
        if self.sa == 1:
            self.optimizer2.zero_grad()

        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred = self.net_r(self.x_r, self.t_r)

        if self.sa == 1:
            loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
            loss_u = torch.mean((self.lamu * (self.u_pred - self.u)) ** 2)
        else:
            loss_r = torch.mean(self.r_pred ** 2)
            loss_u = torch.mean((self.u_pred - self.u) ** 2)

        self.loss = loss_r + loss_u
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:
            with torch.no_grad():
                res = self.net_u(self.xx, self.tt)
                sol = tonp(res)
                sol = np.reshape(sol, (nt, nx)).T

                l2_rel = np.linalg.norm(self.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(self.Exact.flatten(), 2)
                l_inf = np.linalg.norm(self.Exact.flatten() - sol.flatten(), np.inf) / np.linalg.norm(self.Exact.flatten(), np.inf)

                log_message('Iter %d, Loss: %.3e, Rel_L2: %.3e, L_inf: %.3e, Time: %s' %
                            (self.iter, self.loss.item(), l2_rel, l_inf, format_elapsed_time(self.exec_time)))
                self.it.append(self.iter)
                self.l2.append(l2_rel)
                self.linf.append(l_inf)

        self.optimizer.step()
        if self.sa == 1:
            self.optimizer2.step()
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
    log_message('Initializing Vanilla MLP with SA-PINN for the diffusion-reaction equation...')
    layers = [2] + 4 * [20] + [1]

    model = PINN(X_u, u, X_f, lb, ub, savept=f"rd_sa_model_{seed}")

    log_message('Starting training...')
    model.train()
    log_message('Training finished. Total time: %s' % format_elapsed_time(model.exec_time))

    model.dnn.eval()
    with torch.no_grad():
        u_pred = model.net_u(model.xx, model.tt)
        u_pred_np = tonp(u_pred)
        sol = np.reshape(u_pred_np, (nt, nx)).T

    exact_flat = model.Exact.flatten()
    pred_flat = sol.flatten()
    final_l2_rel = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
    final_l_inf = np.linalg.norm(exact_flat - pred_flat, np.inf) / np.linalg.norm(exact_flat, np.inf)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    im0 = ax[0].imshow(model.Exact, cmap='rainbow', aspect='auto', extent=[t_min, t_max, x_min, x_max], origin='lower')
    ax[0].set_title('Exact u(x,t)', fontsize=14)
    ax[0].set_xlabel('t', fontsize=12)
    ax[0].set_ylabel('x', fontsize=12)
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(sol, cmap='rainbow', aspect='auto', extent=[t_min, t_max, x_min, x_max], origin='lower')
    ax[1].set_title('Predicted u(x,t) (SA-PINN)', fontsize=14)
    ax[1].set_xlabel('t', fontsize=12)
    ax[1].set_ylabel('x', fontsize=12)
    fig.colorbar(im1, ax=ax[1])

    error = np.abs(model.Exact - sol)
    im2 = ax[2].imshow(error, cmap='rainbow', aspect='auto', extent=[t_min, t_max, x_min, x_max], origin='lower')
    ax[2].set_title('Absolute Error', fontsize=14)
    ax[2].set_xlabel('t', fontsize=12)
    ax[2].set_ylabel('x', fontsize=12)
    fig.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'RD_SA_Prediction_Results_{seed}.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(model.it, model.l2, label='Relative L2 Error', color='blue', linewidth=1.5)
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative $L^2$ Error', fontsize=12)
    plt.title('Convergence History (SA-PINN)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'RD_SA_Loss_History_{seed}.png'), dpi=300)
    plt.show()

    log_message("-" * 50)
    log_message('Final test-set results')
    log_message(f'Final Relative L2 Error: {final_l2_rel:.5e}')
    log_message(f'Final L_inf Error:       {final_l_inf:.5e}')
    log_message("-" * 50)
