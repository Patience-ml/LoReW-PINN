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
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)
file_name = f'losses_poisson_sa_{seed}.txt'
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


C = 10.0
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
nx, ny = 201, 201
N_f = 10000
N_b_each = 100
first_opt = 50000
print_step = 100
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
    def __init__(self, X_b, u_b, X_r, lb, ub, savept=None):
        self.sa = 1
        self.iter = 0
        self.exec_time = 0
        self.print_step = print_step
        self.savept = savept
        self.dimx_, self.dimy_ = nx, ny
        self.first_opt = first_opt
        self.it, self.l2, self.linf = [], [], []
        self.loss, self.losses = None, []

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

        self.dnn = DNN(layers).to(device)
        if self.sa == 1:
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
        if self.sa == 1:
            self.optimizer2.zero_grad()

        self.u_b_pred = self.net_u(self.x_b, self.y_b)
        self.r_pred = self.net_r(self.x_r, self.y_r)

        if self.sa == 1:
            loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
            loss_b = torch.mean((self.lamb * (self.u_b_pred - self.u_b)) ** 2)
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
        for _ in range(self.first_opt):
            start_time = time.time()
            self.loss_func()
            end_time = time.time()
            self.exec_time += (end_time - start_time)

        d = np.column_stack((np.array(self.it), np.array(self.l2), np.array(self.linf)))
        np.savetxt(os.path.join(LOG_DIR, file_name), d, fmt='%.10f %.10f %.10f')
        if self.savept is not None:
            torch.save(self.dnn.state_dict(), os.path.join(MODEL_DIR, str(self.savept) + ".pt"))


# ==========================================
# 5. Main program and visualization
# ==========================================
if __name__ == "__main__":
    open(train_log_path, "w", encoding="utf-8").close()
    log_message("Initializing SA-PINN for the Poisson equation...")
    layers = [2] + 8 * [50] + [1]

    model = PINN(X_b, u_b, X_f, lb, ub, savept=f"poisson_sa_model_{seed}")

    log_message("Starting training...")
    model.train()
    log_message('Training finished. Total time: %s' % format_elapsed_time(model.exec_time))

    model.dnn.eval()
    with torch.no_grad():
        u_pred = model.net_u(model.xx, model.yy)
        sol = np.reshape(tonp(u_pred), (model.dimx_, model.dimy_))

    exact_flat = model.Exact.flatten()
    pred_flat = sol.flatten()
    final_l2_rel = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
    final_l_inf = np.linalg.norm(exact_flat - pred_flat, np.inf) / np.linalg.norm(exact_flat, np.inf)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    im0 = ax[0].imshow(model.Exact.T, cmap='rainbow', aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    ax[0].set_title('Exact u(x,y)', fontsize=14)
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].set_ylabel('y', fontsize=12)
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(sol.T, cmap='rainbow', aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    ax[1].set_title('Predicted u(x,y) (SA)', fontsize=14)
    ax[1].set_xlabel('x', fontsize=12)
    ax[1].set_ylabel('y', fontsize=12)
    fig.colorbar(im1, ax=ax[1])

    error = np.abs(model.Exact - sol)
    im2 = ax[2].imshow(error.T, cmap='rainbow', aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    ax[2].set_title('Absolute Error', fontsize=14)
    ax[2].set_xlabel('x', fontsize=12)
    ax[2].set_ylabel('y', fontsize=12)
    fig.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"Poisson_SA_Prediction_Results_{seed}.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(model.it, model.l2, label='Relative L2 Error', color='blue', linewidth=1.5)
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative $L^2$ Error', fontsize=12)
    plt.title('Convergence History (SA)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"Poisson_SA_Loss_History_{seed}.png"), dpi=300)
    plt.show()

    log_message('-' * 50)
    log_message('Final test-set results')
    log_message(f'Final Relative L2 Error: {final_l2_rel:.5e}')
    log_message(f'Final L_inf Error:       {final_l_inf:.5e}')
    log_message('-' * 50)
