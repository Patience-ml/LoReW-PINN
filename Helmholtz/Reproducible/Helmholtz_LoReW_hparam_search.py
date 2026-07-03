import csv
import os
import random
import time
from collections import OrderedDict

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.optim import lr_scheduler


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


def tonp(tensor):
    return tensor.detach().cpu().numpy()


def grad(u, x):
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]


def format_elapsed_time(seconds):
    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    if minutes > 0:
        return f"{minutes}min{seconds}s"
    return f"{seconds}s"


file_path = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(file_path, 'logs')
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)
result_dir = os.path.join(file_path, "LoReW_hparam_search")
os.makedirs(result_dir, exist_ok=True)
summary_path = os.path.join(result_dir, "Helmholtz_LoReW_hparam_search_summary.csv")

seed = 2138
seed_torch(seed)

x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
nx, ny = 1001, 1001
xdim, ydim = 4.0, 1.0
ksq = 1.0
N_grid = 101
N_u = 50
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

X_left = np.hstack((np.full((N_u, 1), x_min), boundary_y))
X_right = np.hstack((np.full((N_u, 1), x_max), boundary_y))
X_top = np.hstack((boundary_x, np.full((N_u, 1), y_max)))
X_bottom = np.hstack((boundary_x, np.full((N_u, 1), y_min)))
X_u = np.vstack([X_left, X_top, X_bottom, X_right])
u_bc = np.zeros((X_u.shape[0], 1))

lb = np.array([x_min, y_min], dtype=np.float32)
ub = np.array([x_max, y_max], dtype=np.float32)
X_r_grid, Y_r_grid = np.meshgrid(train_x.flatten(), train_y.flatten())
X_r = np.hstack((X_r_grid.flatten()[:, None], Y_r_grid.flatten()[:, None]))


def source_torch(x, y):
    exact = torch.sin(xdim * np.pi * x) * torch.sin(ydim * np.pi * y)
    return -((xdim * np.pi) ** 2) * exact - ((ydim * np.pi) ** 2) * exact + ksq * exact


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh()

        layer_list = []
        for i in range(self.depth - 1):
            w_layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(w_layer.weight)
            layer_list.append((f"layer_{i}", w_layer))
            layer_list.append((f"activation_{i}", self.activation))

        w_layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(w_layer.weight)
        layer_list.append((f"layer_{self.depth - 1}", w_layer))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


class LoReW_PINN:
    def __init__(self, config):
        self.iter = 0
        self.first_opt = 40000
        self.bc_weight = 1.0
        self.eta = config["eta"]
        self.gamma = config["gamma"]
        self.init = 1
        self.kernel_h_x = config["kernel_h_x"]
        self.kernel_h_y = config["kernel_h_y"]
        self.kernel_p = config["kernel_p"]
        self.kernel_trunc = config.get("kernel_trunc", 3.0)
        self.num_kernel_centers_x = config.get("num_kernel_centers_x", 31)
        self.num_kernel_centers_y = config.get("num_kernel_centers_y", 31)
        self.kernel_eps = 1e-12

        self.xx = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        self.yy = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(X_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_r = torch.tensor(X_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(u_bc, dtype=torch.float32, device=device)

        seed_torch(seed)
        self.dnn = DNN(layers).to(device)
        self.rsum = 0

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
        return u_xx + u_yy + ksq * u - source_torch(x, y)

    def train_step(self):
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.y_u)
        r_pred = self.net_r(self.x_r, self.y_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        abs_r_detach = torch.abs(r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
        self.rsum = (self.rsum * self.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * r_pred) ** 2)
        loss_u = torch.mean((u_pred - self.u) ** 2)
        loss = loss_r + self.bc_weight * loss_u
        loss.backward()
        self.optimizer.step()
        self.iter += 1
        return float(loss.detach().cpu())

    def train(self):
        self.dnn.train()
        last_loss = np.nan
        for epoch in range(self.first_opt):
            last_loss = self.train_step()
            if (epoch + 1) % self.step_size == 0:
                self.scheduler.step()
        return last_loss

    def evaluate(self):
        self.dnn.eval()
        chunks = []
        chunk_size = 200000
        with torch.no_grad():
            for start in range(0, self.xx.shape[0], chunk_size):
                end = start + chunk_size
                pred = self.net_u(self.xx[start:end], self.yy[start:end])
                chunks.append(tonp(pred))
        sol = np.vstack(chunks).reshape(ny, nx).T
        l2_rel = np.linalg.norm(Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(Exact.flatten(), 2)
        l_inf = np.linalg.norm(Exact.flatten() - sol.flatten(), np.inf) / np.linalg.norm(Exact.flatten(), np.inf)
        return float(l2_rel), float(l_inf)


CANDIDATES = [
    {"name": "c00_current", "eta": 5e-4, "gamma": 0.95, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c01_g970", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c02_g980", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c03_g990", "eta": 5e-4, "gamma": 0.99, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c04_eta2e4", "eta": 2e-4, "gamma": 0.98, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c05_eta1e3", "eta": 1e-3, "gamma": 0.98, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c06_p125", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.25},
    {"name": "c07_p175", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.75},
    {"name": "c08_h008_010", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.08, "kernel_h_y": 0.10, "kernel_p": 1.5},
    {"name": "c09_h012_012", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c10_h012_014", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.14, "kernel_p": 1.5},
    {"name": "c11_h015_015", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.15, "kernel_h_y": 0.15, "kernel_p": 1.5},
    {"name": "c12_h012_012_g960", "eta": 5e-4, "gamma": 0.96, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c13_h012_012_g970", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c14_h012_012_g975", "eta": 5e-4, "gamma": 0.975, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c15_h011_012_g970", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.11, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c16_h012_013_g970", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.12, "kernel_h_y": 0.13, "kernel_p": 1.5},
    {"name": "c17_h013_013_g970", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.13, "kernel_h_y": 0.13, "kernel_p": 1.5},
    {"name": "c18_h012_012_g970_p140", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.4},
    {"name": "c19_h012_012_g970_p160", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.6},
    {"name": "c20_h012_012_g970_eta3e4", "eta": 3e-4, "gamma": 0.97, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c21_h012_012_g970_eta7e4", "eta": 7e-4, "gamma": 0.97, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c22_h014_014_g970", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.14, "kernel_h_y": 0.14, "kernel_p": 1.5},
    {"name": "c23_h014_012_g970", "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.14, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c24_h012_012_g982", "eta": 5e-4, "gamma": 0.982, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c25_h012_012_g985", "eta": 5e-4, "gamma": 0.985, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c26_h012_012_g980_eta3e4", "eta": 3e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c27_h012_012_g980_eta4e4", "eta": 4e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c28_h012_012_g980_eta6e4", "eta": 6e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c29_h012_012_g980_p160", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.6},
    {"name": "c30_h012_012_g980_p170", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.7},
    {"name": "c31_h012_012_g980_p180", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.8},
    {"name": "c32_h011_011_g980", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.11, "kernel_h_y": 0.11, "kernel_p": 1.5},
    {"name": "c33_h011_012_g980", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.11, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c34_h013_012_g980", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.13, "kernel_h_y": 0.12, "kernel_p": 1.5},
    {"name": "c35_h013_013_g980", "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.13, "kernel_h_y": 0.13, "kernel_p": 1.5},
    {
        "name": "c36_h012_012_g980_centers21",
        "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "num_kernel_centers_x": 21, "num_kernel_centers_y": 21,
    },
    {
        "name": "c37_h012_012_g980_centers41",
        "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "num_kernel_centers_x": 41, "num_kernel_centers_y": 41,
    },
    {
        "name": "c38_h012_012_g980_centers51",
        "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "num_kernel_centers_x": 51, "num_kernel_centers_y": 51,
    },
    {
        "name": "c39_h012_012_g980_trunc2",
        "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "kernel_trunc": 2.0,
    },
    {
        "name": "c40_h012_012_g980_trunc4",
        "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "kernel_trunc": 4.0,
    },
    {
        "name": "c41_h012_012_g980_notrunc",
        "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "kernel_trunc": 0.0,
    },
    {
        "name": "c42_h012_012_g980_centers41_trunc2",
        "eta": 5e-4, "gamma": 0.98, "kernel_h_x": 0.12, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "num_kernel_centers_x": 41, "num_kernel_centers_y": 41, "kernel_trunc": 2.0,
    },
    {
        "name": "c43_h010_012_g970_centers41",
        "eta": 5e-4, "gamma": 0.97, "kernel_h_x": 0.10, "kernel_h_y": 0.12, "kernel_p": 1.5,
        "num_kernel_centers_x": 41, "num_kernel_centers_y": 41,
    },
]


def run_candidate(config):
    seed_torch(seed)
    start = time.time()
    model = LoReW_PINN(config)
    final_loss = model.train()
    l2_rel, l_inf = model.evaluate()
    elapsed = time.time() - start
    return {
        **config,
        "seed": seed,
        "final_loss": final_loss,
        "l2_rel": l2_rel,
        "l_inf": l_inf,
        "elapsed": format_elapsed_time(elapsed),
    }


def main():
    fieldnames = [
        "name", "seed", "eta", "gamma", "kernel_h_x", "kernel_h_y", "kernel_p",
        "kernel_trunc", "num_kernel_centers_x", "num_kernel_centers_y",
        "final_loss", "l2_rel", "l_inf", "elapsed",
    ]
    completed_names = set()
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                completed_names.add(row["name"])
    else:
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for config in CANDIDATES:
        if config["name"] in completed_names:
            continue
        print(f"Running {config['name']} ...", flush=True)
        result = run_candidate(config)
        with open(summary_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result)
        print(
            f"{result['name']}: L2={result['l2_rel']:.6e}, "
            f"Linf={result['l_inf']:.6e}, time={result['elapsed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
