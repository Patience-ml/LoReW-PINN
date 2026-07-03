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
import scipy.io
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)
DATA = scipy.io.loadmat(os.path.join(DATA_DIR, "burgers_shock.mat"))

nu = 0.01 / np.pi
Exact = np.real(DATA["usol"])
t0 = DATA["t"].flatten()[:, None]
x0 = DATA["x"].flatten()[:, None]
X, T = np.meshgrid(x0.flatten(), t0.flatten())
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(np.float32)
nx, nt = x0.shape[0], t0.shape[0]
N_i = nx
N_b = nt
N_u = N_i + 2 * N_b
N_f_default = 10000


@dataclass(frozen=True)
class CLoReWConfig:
    config_id: str
    eta: float
    gamma: float
    alpha: float
    epsilon: float
    eta_g: float
    rho_g: float
    kernel_h: float
    kernel_p: float
    kernel_trunc: float
    num_kernel_centers: int
    lr: float
    lr_decay: float
    step_size: int
    width: int
    depth: int
    n_f: int


CONFIGS = [
    CLoReWConfig("c00_current", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c01_eta5e5", 5e-5, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c02_g9995", 1e-4, 0.9995, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c03_eta5e5_g9995", 5e-5, 0.9995, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c04_g9997", 1e-4, 0.9997, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c05_eta75e6_g9997", 7.5e-5, 0.9997, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c06_eta_g5e6", 1e-4, 0.9999, 2.0, 2.0, 5e-6, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c07_eps4", 1e-4, 0.9999, 2.0, 4.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c08_eta_g5e6_eps4", 1e-4, 0.9999, 2.0, 4.0, 5e-6, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c09_rho9995", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.9995, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c10_alpha1", 1e-4, 0.9999, 1.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c11_alpha4", 1e-4, 0.9999, 4.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c12_h006", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.06, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c13_h015", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.15, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c14_p2", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.10, 2.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c15_lr5e5", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 5e-5, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c16_lr1e4_decay95", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.95, 2000, 20, 8, 10000),
    CLoReWConfig("c17_w40", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 40, 6, 10000),
    CLoReWConfig("c18_nf20000", 1e-4, 0.9999, 2.0, 2.0, 1e-5, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 20000),
    CLoReWConfig("c19_eta_g75e7_eps3", 1e-4, 0.9999, 2.0, 3.0, 7.5e-6, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c20_eta_g75e7_eps4", 1e-4, 0.9999, 2.0, 4.0, 7.5e-6, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c21_eta_g6e6_eps3", 1e-4, 0.9999, 2.0, 3.0, 6e-6, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
    CLoReWConfig("c22_eta_g85e7_eps4", 1e-4, 0.9999, 2.0, 4.0, 8.5e-6, 0.999, 0.10, 1.0, 3.0, 41, 1e-4, 0.9, 2000, 20, 8, 10000),
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


class CLoReWPINN:
    def __init__(self, data, cfg, adam_iters, eval_interval):
        self.cfg = cfg
        self.iter = 0
        self.exec_time = 0.0
        self.first_opt = adam_iters
        self.eval_interval = eval_interval
        self.it, self.l2, self.linf = [], [], []
        self.loss = None

        self.xx = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        self.tt = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        self.x_u = torch.tensor(data["X_u"][:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_u = torch.tensor(data["X_u"][:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(data["X_f"][:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_r = torch.tensor(data["X_f"][:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(data["u"], dtype=torch.float32, device=device)

        layers = [2] + cfg.depth * [cfg.width] + [1]
        self.dnn = DNN(layers).to(device)

        self.rsum = 0
        self.tau_gate = 0.0
        self.gbar = torch.zeros_like(self.x_r).detach()
        self.kernel_eps = 1e-12
        self.gate_abs_res_ref = None

        t_min = float(data["lb"][1])
        t_max = float(data["ub"][1])
        self.t_centers = torch.linspace(t_min, t_max, cfg.num_kernel_centers, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.lr_decay)

    def build_kernel_weights(self, t_points):
        diff = t_points - self.t_centers
        K = torch.exp(-0.5 * (diff / self.cfg.kernel_h) ** 2)
        if self.cfg.kernel_trunc is not None and self.cfg.kernel_trunc > 0:
            K = K * (torch.abs(diff) <= self.cfg.kernel_trunc * self.cfg.kernel_h).float()

        row_sum = K.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            nearest_idx = torch.argmin(torch.abs(diff[zero_rows]), dim=1)
            K[zero_rows] = 0.0
            K[zero_rows, nearest_idx] = 1.0
            row_sum = K.sum(dim=1, keepdim=True)
        return K / (row_sum + self.kernel_eps)

    def kernel_local_scale(self, abs_r_detach):
        moment = (self.kernel_weights.T @ (abs_r_detach ** self.cfg.kernel_p)) / self.kernel_col_mass
        local_moment = self.kernel_weights @ moment
        return (local_moment + self.kernel_eps) ** (1.0 / self.cfg.kernel_p)

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        return u_t + u * u_x - nu * u_xx

    def evaluate(self):
        self.dnn.eval()
        with torch.no_grad():
            pred = tonp(self.net_u(self.xx, self.tt)).reshape(nt, nx).T
        l2 = np.linalg.norm(Exact - pred, 2) / np.linalg.norm(Exact, 2)
        linf = np.linalg.norm(Exact - pred, np.inf) / np.linalg.norm(Exact, np.inf)
        self.dnn.train()
        return float(l2), float(linf)

    def loss_func(self):
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)

        g_t = 0.5 * (1.0 - torch.tanh(self.cfg.alpha * (self.t_r - self.tau_gate)))
        self.gbar = (self.cfg.rho_g * self.gbar + (1.0 - self.cfg.rho_g) * g_t.detach()).detach()

        abs_r_detach = torch.abs(r_pred).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = self.cfg.eta * abs_r_detach / (local_scale + self.kernel_eps)
        causal_r_norm = r_norm * self.gbar
        self.rsum = (self.rsum * self.cfg.gamma + causal_r_norm).detach()

        loss_r = torch.mean((self.rsum * r_pred) ** 2)
        loss_u = torch.mean((u_pred - self.u) ** 2)
        self.loss = loss_r + loss_u
        self.loss.backward()
        self.iter += 1
        self.optimizer.step()

        gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.kernel_eps)
        if self.gate_abs_res_ref is None:
            self.gate_abs_res_ref = gate_abs_res.item() + self.kernel_eps
        gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref
        self.tau_gate = self.tau_gate + self.cfg.eta_g * np.exp(-self.cfg.epsilon * gate_abs_res_norm)
        return float(self.loss.item())

    def train(self, log):
        last_loss = None
        self.dnn.train()
        for epoch in range(self.first_opt):
            start = time.time()
            last_loss = self.loss_func()
            self.exec_time += time.time() - start
            if (epoch + 1) % self.cfg.step_size == 0:
                self.scheduler.step()
            if self.iter % self.eval_interval == 0:
                l2, linf = self.evaluate()
                self.it.append(self.iter)
                self.l2.append(l2)
                self.linf.append(linf)
                log(
                    "Iter %d, Loss: %.3e, L2: %.3e, Linf: %.3e, tau: %.3f, gbar_mean: %.3e, Time: %s"
                    % (self.iter, last_loss, l2, linf, self.tau_gate, float(torch.mean(self.gbar).item()), format_elapsed_time(self.exec_time))
                )

        if not self.it or self.it[-1] != self.iter:
            l2, linf = self.evaluate()
            self.it.append(self.iter)
            self.l2.append(l2)
            self.linf.append(linf)
        log("Final eval at iter %d: L2=%.6e, Linf=%.6e" % (self.iter, self.l2[-1], self.linf[-1]))
        return last_loss


def build_data(seed, n_f):
    seed_torch(seed)
    x_mesh, t_mesh = X, T
    X_i = np.hstack((x_mesh[0:1, :].T, t_mesh[0:1, :].T))
    u_i = Exact[:, 0:1]
    X_lb = np.hstack((x_mesh[:, 0:1], t_mesh[:, 0:1]))
    X_ub = np.hstack((x_mesh[:, -1:], t_mesh[:, -1:]))
    u_lb = Exact[0:1, :].T
    u_ub = Exact[-1:, :].T
    X_u = np.vstack([X_i, X_lb, X_ub]).astype(np.float32)
    u = np.vstack([u_i, u_lb, u_ub]).astype(np.float32)

    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)
    X_f = lb + (ub - lb) * lhs(2, n_f, random_state=seed)
    X_f = np.vstack((X_f, X_u)).astype(np.float32)
    return {"X_u": X_u, "u": u, "X_f": X_f, "lb": lb, "ub": ub}


def read_baselines(seed):
    baselines = {}
    for method, filename in {
        "PINN": f"losses_burgers_pinn_{seed}.txt",
        "SA": f"losses_burgers_sa_{seed}.txt",
        "RBA": f"losses_burgers_rba_{seed}.txt",
        "C-LoReW_existing": f"losses_burgers_clorew_pinn_{seed}.txt",
    }.items():
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            data = np.loadtxt(path)
            if data.ndim == 2 and data.shape[0] > 0:
                baselines[method] = (float(data[-1, 1]), float(data[-1, 2]), float(np.min(data[:, 1])))
    return baselines


def summarize_curve(model):
    l2 = np.array(model.l2, dtype=np.float64)
    linf = np.array(model.linf, dtype=np.float64)
    it = np.array(model.it, dtype=np.int64)
    best_idx = int(np.argmin(l2))
    after = l2[best_idx:]
    tail = l2[-10:] if l2.size >= 10 else l2
    final_l2 = float(l2[-1])
    best_l2 = float(l2[best_idx])
    max_after_best = float(np.max(after))
    final_over_best = final_l2 / max(best_l2, 1e-12)
    max_after_best_ratio = max_after_best / max(best_l2, 1e-12)
    tail_std = float(np.std(tail))
    tail_range = float(np.max(tail) - np.min(tail))
    stability_score = final_l2 * (1.0 + 0.15 * max(0.0, final_over_best - 1.0) + 0.10 * max(0.0, max_after_best_ratio - 1.0))
    return {
        "final_l2": final_l2,
        "final_linf": float(linf[-1]),
        "best_l2": best_l2,
        "best_l2_iter": int(it[best_idx]),
        "final_over_best": float(final_over_best),
        "max_after_best_ratio": float(max_after_best_ratio),
        "tail_l2_std": tail_std,
        "tail_l2_range": tail_range,
        "stability_score": float(stability_score),
    }


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_config(output_dir, seed, cfg, adam_iters, eval_interval, force):
    result_path = os.path.join(output_dir, f"losses_burgers_clorew_search_{cfg.config_id}_seed_{seed}.txt")
    log_path = os.path.join(output_dir, f"Burgers_C-LoReW_search_{cfg.config_id}_seed_{seed}.log")
    summary_path = os.path.join(output_dir, "Burgers_C_LoReW_hparam_search_results.csv")

    if os.path.exists(result_path) and not force:
        data = np.loadtxt(result_path)
        l2 = data[:, 1]
        best_idx = int(np.argmin(l2))
        return {
            "config_id": cfg.config_id,
            "seed": seed,
            "skipped": True,
            "final_l2": float(l2[-1]),
            "best_l2": float(l2[best_idx]),
            "stability_score": float(l2[-1]),
        }

    seed_torch(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = build_data(seed, cfg.n_f)
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"Device: {device}")
        log(f"Seed: {seed}, Adam iters: {adam_iters}, eval_interval: {eval_interval}")
        log(f"Config: {asdict(cfg)}")
        baselines = read_baselines(seed)
        if baselines:
            log(f"Baselines: {baselines}")
        model = CLoReWPINN(data, cfg, adam_iters=adam_iters, eval_interval=eval_interval)
        start = time.time()
        final_loss = model.train(log)
        wall_time = time.time() - start
        summary = summarize_curve(model)
        log(f"Finished {cfg.config_id}: final_loss={final_loss:.6e}, metrics={summary}, wall_time={format_elapsed_time(wall_time)}")

    history = np.column_stack((np.array(model.it), np.array(model.l2), np.array(model.linf)))
    np.savetxt(result_path, history, fmt="%.10f %.10f %.10f")

    row = {
        "config_id": cfg.config_id,
        "seed": seed,
        "adam_iters": adam_iters,
        "eval_interval": eval_interval,
        "wall_time_sec": float(wall_time),
        **summary,
        **asdict(cfg),
    }
    append_csv(summary_path, row)
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4123)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--adam-iters", type=int, default=40000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(BASE_DIR, "C_LoReW_hparam_search")
    os.makedirs(output_dir, exist_ok=True)

    configs = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [cfg for cfg in configs if cfg.config_id in wanted]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    seeds = args.seeds if args.seeds is not None and len(args.seeds) > 0 else [args.seed]
    rows = []
    for seed in seeds:
        for cfg in configs:
            print(f"\n=== Running seed {seed}, {cfg.config_id} ===", flush=True)
            row = run_config(
                output_dir=output_dir,
                seed=seed,
                cfg=cfg,
                adam_iters=args.adam_iters,
                eval_interval=args.eval_interval,
                force=args.force,
            )
            rows.append(row)

    ranked = sorted(rows, key=lambda item: item.get("stability_score", item["final_l2"]))
    print("\nSearch ranking by stability score:")
    for row in ranked:
        print(
            "  seed %s %s: score=%.6e, final_l2=%.6e, best_l2=%.6e, final/best=%.3f, max_after/best=%.3f"
            % (
                row["seed"],
                row["config_id"],
                row.get("stability_score", row["final_l2"]),
                row["final_l2"],
                row.get("best_l2", row["final_l2"]),
                row.get("final_over_best", 1.0),
                row.get("max_after_best_ratio", 1.0),
            )
        )


if __name__ == "__main__":
    main()
