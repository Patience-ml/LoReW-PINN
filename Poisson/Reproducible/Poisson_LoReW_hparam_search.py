import argparse
import csv
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass

# Set before importing torch so cuBLAS can use deterministic kernels.
def format_elapsed_time(seconds):
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f'{hours}h{minutes}min{seconds}s'
    if minutes > 0:
        return f'{minutes}min{seconds}s'
    return f'{seconds}s'

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch


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


def lhs(dim, samples, random_state=None):
    rng = np.random.RandomState(random_state)
    result = np.empty((samples, dim), dtype=np.float32)
    cut = np.linspace(0.0, 1.0, samples + 1, dtype=np.float32)
    for j in range(dim):
        points = cut[:samples] + rng.rand(samples).astype(np.float32) * (cut[1:] - cut[:samples])
        rng.shuffle(points)
        result[:, j] = points
    return result


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


C = 10.0
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
nx, ny = 201, 201
N_f = 10000
N_b_each = 100
layers = [2] + 8 * [50] + [1]


def exact_u_np(x, y):
    return np.cos(C * x) + np.cos(C * y)


def source_torch(x, y):
    return 2.0 * (C ** 2) * (torch.cos(C * x) + torch.cos(C * y))


@dataclass(frozen=True)
class LoReWConfig:
    config_id: str
    eta: float
    gamma: float
    kernel_h_x: float
    kernel_h_y: float
    kernel_p: float
    kernel_trunc: float
    num_kernel_centers_x: int
    num_kernel_centers_y: int


CONFIGS = [
    LoReWConfig("c00_current", 1e-3, 0.999, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c01_h003", 1e-3, 0.999, 0.03, 0.03, 2.0, 3.0, 31, 31),
    LoReWConfig("c02_h008", 1e-3, 0.999, 0.08, 0.08, 2.0, 3.0, 31, 31),
    LoReWConfig("c03_eta5e4", 5e-4, 0.999, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c04_g997", 1e-3, 0.997, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c05_eta5e4_g997", 5e-4, 0.997, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c06_p1", 1e-3, 0.999, 0.05, 0.05, 1.0, 3.0, 31, 31),
    LoReWConfig("c07_p4", 1e-3, 0.999, 0.05, 0.05, 4.0, 3.0, 31, 31),
    LoReWConfig("c08_h008_p1", 1e-3, 0.999, 0.08, 0.08, 1.0, 3.0, 31, 31),
    LoReWConfig("c09_eta25e5_g997", 2.5e-4, 0.997, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c10_eta75e5_g997", 7.5e-4, 0.997, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c11_eta5e4_g995", 5e-4, 0.995, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c12_eta5e4_g998", 5e-4, 0.998, 0.05, 0.05, 2.0, 3.0, 31, 31),
    LoReWConfig("c13_h004_eta5e4_g997", 5e-4, 0.997, 0.04, 0.04, 2.0, 3.0, 31, 31),
    LoReWConfig("c14_h006_eta5e4_g997", 5e-4, 0.997, 0.06, 0.06, 2.0, 3.0, 31, 31),
    LoReWConfig("c15_h007_eta5e4_g997", 5e-4, 0.997, 0.07, 0.07, 2.0, 3.0, 31, 31),
    LoReWConfig("c16_p15_eta5e4_g997", 5e-4, 0.997, 0.05, 0.05, 1.5, 3.0, 31, 31),
    LoReWConfig("c17_p3_eta5e4_g997", 5e-4, 0.997, 0.05, 0.05, 3.0, 3.0, 31, 31),
    LoReWConfig("c18_c41_eta5e4_g997", 5e-4, 0.997, 0.05, 0.05, 2.0, 3.0, 41, 41),
]


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh()

        layer_list = []
        for i in range(self.depth - 1):
            layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(layer.weight)
            layer_list.append((f"layer_{i}", layer))
            layer_list.append((f"activation_{i}", self.activation))

        layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(layer.weight)
        layer_list.append((f"layer_{self.depth - 1}", layer))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


class LoReWPINN:
    def __init__(self, data, cfg, adam_iters, eval_interval, eval_grid):
        self.cfg = cfg
        self.iter = 0
        self.exec_time = 0.0
        self.first_opt = adam_iters
        self.eval_interval = eval_interval
        self.eval_grid = eval_grid
        self.it, self.l2, self.linf = [], [], []
        self.loss = None

        self.x_b = torch.tensor(data["X_b"][:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_b = torch.tensor(data["X_b"][:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(data["X_r"][:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.y_r = torch.tensor(data["X_r"][:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u_b = torch.tensor(data["u_b"], dtype=torch.float32, device=device)

        self.dnn = DNN(layers).to(device)
        self.rsum = 0
        self.init = 1
        self.kernel_eps = 1e-12

        centers_x = torch.linspace(x_min, x_max, cfg.num_kernel_centers_x, device=device)
        centers_y = torch.linspace(y_min, y_max, cfg.num_kernel_centers_y, device=device)
        mesh_x, mesh_y = torch.meshgrid(centers_x, centers_y, indexing="ij")
        self.kernel_centers_x = mesh_x.reshape(1, -1)
        self.kernel_centers_y = mesh_y.reshape(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.x_r.detach(), self.y_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def build_kernel_weights(self, x_points, y_points):
        dx = (x_points - self.kernel_centers_x) / self.cfg.kernel_h_x
        dy = (y_points - self.kernel_centers_y) / self.cfg.kernel_h_y
        dist2 = dx ** 2 + dy ** 2
        K = torch.exp(-0.5 * dist2)
        if self.cfg.kernel_trunc is not None and self.cfg.kernel_trunc > 0:
            K = K * (dist2 <= self.cfg.kernel_trunc ** 2).float()

        row_sum = K.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            nearest_idx = torch.argmin(dist2[zero_rows], dim=1)
            K[zero_rows] = 0.0
            K[zero_rows, nearest_idx] = 1.0
            row_sum = K.sum(dim=1, keepdim=True)

        return K / (row_sum + self.kernel_eps)

    def kernel_local_scale(self, abs_r_detach):
        center_moment_p = (self.kernel_weights.T @ (abs_r_detach ** self.cfg.kernel_p)) / self.kernel_col_mass
        local_moment_p = self.kernel_weights @ center_moment_p
        return (local_moment_p + self.kernel_eps) ** (1.0 / self.cfg.kernel_p)

    def net_u(self, x, y):
        return self.dnn(torch.cat([x, y], dim=1))

    def net_r(self, x, y):
        u = self.net_u(x, y)
        u_x = grad(u, x)
        u_y = grad(u, y)
        u_xx = grad(u_x, x)
        u_yy = grad(u_y, y)
        return -u_xx - u_yy + (C ** 2) * u - source_torch(x, y)

    def loss_func(self):
        self.optimizer.zero_grad()

        u_b_pred = self.net_u(self.x_b, self.y_b)
        r_pred = self.net_r(self.x_r, self.y_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.cfg.eta
        abs_r_detach = torch.abs(r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
        self.rsum = (self.rsum * self.cfg.gamma + r_norm).detach()

        loss_r = torch.mean((self.rsum * r_pred) ** 2)
        loss_b = torch.mean((u_b_pred - self.u_b) ** 2)
        self.loss = loss_r + loss_b
        self.loss.backward()
        self.iter += 1
        self.optimizer.step()
        return float(self.loss.item())

    def predict_grid(self, grid_size, batch_size=65536):
        x = np.linspace(x_min, x_max, grid_size, dtype=np.float32)[:, None]
        y = np.linspace(y_min, y_max, grid_size, dtype=np.float32)[:, None]
        X, Y = np.meshgrid(x.flatten(), y.flatten(), indexing="ij")
        coords = np.hstack((X.flatten()[:, None], Y.flatten()[:, None])).astype(np.float32)
        exact = exact_u_np(coords[:, 0:1], coords[:, 1:2]).reshape(-1)

        preds = []
        self.dnn.eval()
        with torch.no_grad():
            for start in range(0, coords.shape[0], batch_size):
                batch = coords[start:start + batch_size]
                xb = torch.tensor(batch[:, 0:1], dtype=torch.float32, device=device)
                yb = torch.tensor(batch[:, 1:2], dtype=torch.float32, device=device)
                preds.append(tonp(self.net_u(xb, yb)).reshape(-1))
        pred = np.concatenate(preds)
        l2 = np.linalg.norm(exact - pred, 2) / np.linalg.norm(exact, 2)
        linf = np.linalg.norm(exact - pred, np.inf) / np.linalg.norm(exact, np.inf)
        self.dnn.train()
        return float(l2), float(linf)

    def train(self, log):
        self.dnn.train()
        last_loss = None
        for _ in range(self.first_opt):
            start = time.time()
            last_loss = self.loss_func()
            self.exec_time += time.time() - start
            if self.iter % self.eval_interval == 0:
                l2, linf = self.predict_grid(self.eval_grid)
                self.it.append(self.iter)
                self.l2.append(l2)
                self.linf.append(linf)
                log(f"Iter {self.iter}, Loss: {last_loss:.3e}, coarse_L2: {l2:.3e}, coarse_Linf: {linf:.3e}, Time: {format_elapsed_time(self.exec_time)}")

        l2, linf = self.predict_grid(self.eval_grid)
        self.it.append(self.iter)
        self.l2.append(l2)
        self.linf.append(linf)
        log(f"Final coarse eval at iter {self.iter}: L2={l2:.6e}, Linf={linf:.6e}")
        return last_loss


def build_data(seed):
    seed_torch(seed)
    lb = np.array([x_min, y_min], dtype=np.float32)
    ub = np.array([x_max, y_max], dtype=np.float32)
    X_r = lb + (ub - lb) * lhs(2, N_f, random_state=seed)

    s_left = lhs(1, N_b_each, random_state=seed + 1)
    s_right = lhs(1, N_b_each, random_state=seed + 2)
    s_bottom = lhs(1, N_b_each, random_state=seed + 3)
    s_top = lhs(1, N_b_each, random_state=seed + 4)
    X_left = np.hstack((np.full((N_b_each, 1), x_min), y_min + (y_max - y_min) * s_left))
    X_right = np.hstack((np.full((N_b_each, 1), x_max), y_min + (y_max - y_min) * s_right))
    X_bottom = np.hstack((x_min + (x_max - x_min) * s_bottom, np.full((N_b_each, 1), y_min)))
    X_top = np.hstack((x_min + (x_max - x_min) * s_top, np.full((N_b_each, 1), y_max)))
    X_b = np.vstack((X_left, X_right, X_bottom, X_top)).astype(np.float32)
    u_b = exact_u_np(X_b[:, 0:1], X_b[:, 1:2]).astype(np.float32)
    return {"X_b": X_b, "u_b": u_b, "X_r": X_r.astype(np.float32)}


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_config(output_dir, seed, cfg, adam_iters, eval_interval, eval_grid, final_grid, force):
    result_path = os.path.join(output_dir, f"losses_poisson_lorew_search_{cfg.config_id}_seed_{seed}.txt")
    log_path = os.path.join(output_dir, f"Poisson_LoReW_search_{cfg.config_id}_seed_{seed}.log")
    summary_path = os.path.join(output_dir, "Poisson_LoReW_hparam_search_results.csv")

    if os.path.exists(result_path) and not force:
        data = np.loadtxt(result_path)
        return {
            "config_id": cfg.config_id,
            "seed": seed,
            "skipped": True,
            "final_l2": float(data[-1, 1]),
            "final_linf": float(data[-1, 2]),
        }

    seed_torch(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = build_data(seed)
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"Device: {device}")
        log(f"Seed: {seed}, Adam iters: {adam_iters}, eval_grid: {eval_grid}, final_grid: {final_grid}")
        log(f"Config: {asdict(cfg)}")
        model = LoReWPINN(data, cfg, adam_iters=adam_iters, eval_interval=eval_interval, eval_grid=eval_grid)
        start = time.time()
        final_loss = model.train(log)
        wall_time = time.time() - start
        final_l2, final_linf = model.predict_grid(final_grid)
        log(
            f"Finished {cfg.config_id}: final_loss={final_loss:.6e}, "
            f"final_L2={final_l2:.6e}, final_Linf={final_linf:.6e}, wall_time={format_elapsed_time(wall_time)}"
        )

    history = np.column_stack((np.array(model.it), np.array(model.l2), np.array(model.linf)))
    np.savetxt(result_path, history, fmt="%.10f %.10f %.10f")

    best_idx = int(np.argmin(model.l2))
    row = {
        "config_id": cfg.config_id,
        "seed": seed,
        "N_f": N_f,
        "N_b_each": N_b_each,
        "adam_iters": adam_iters,
        "eval_grid": eval_grid,
        "final_grid": final_grid,
        "final_l2": final_l2,
        "final_linf": final_linf,
        "best_coarse_l2": float(np.min(model.l2)),
        "best_coarse_l2_iter": int(model.it[best_idx]),
        "wall_time_sec": float(wall_time),
        **asdict(cfg),
    }
    append_csv(summary_path, row)
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--adam-iters", type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-grid", type=int, default=101)
    parser.add_argument("--final-grid", type=int, default=201)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "LoReW_hparam_search")
    os.makedirs(output_dir, exist_ok=True)

    configs = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [cfg for cfg in configs if cfg.config_id in wanted]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    rows = []
    for cfg in configs:
        print(f"\n=== Running {cfg.config_id} ===", flush=True)
        row = run_config(
            output_dir=output_dir,
            seed=args.seed,
            cfg=cfg,
            adam_iters=args.adam_iters,
            eval_interval=args.eval_interval,
            eval_grid=args.eval_grid,
            final_grid=args.final_grid,
            force=args.force,
        )
        rows.append(row)

    ranked = sorted(rows, key=lambda item: item["final_l2"])
    print("\nSearch ranking by final L2RE:")
    for row in ranked:
        print(f"  {row['config_id']}: final_l2={row['final_l2']:.6e}, final_linf={row['final_linf']:.6e}")


if __name__ == "__main__":
    main()
