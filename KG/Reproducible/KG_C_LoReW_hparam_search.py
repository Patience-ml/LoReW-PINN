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


def lhs_sample(dim, n, seed):
    rng = np.random.default_rng(seed)
    result = np.empty((n, dim), dtype=np.float64)
    cut = np.linspace(0.0, 1.0, n + 1)
    for j in range(dim):
        u = rng.random(n)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])
        result[:, j] = points[rng.permutation(n)]
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


def exact_u_np(x, t):
    return x * np.cos(5.0 * np.pi * t) + (x * t) ** 3


def exact_ut_np(x, t):
    return -5.0 * np.pi * x * np.sin(5.0 * np.pi * t) + 3.0 * (x ** 3) * (t ** 2)


def source_torch(x, t):
    u = x * torch.cos(5.0 * np.pi * t) + (x * t) ** 3
    u_tt = -25.0 * (np.pi ** 2) * x * torch.cos(5.0 * np.pi * t) + 6.0 * (x ** 3) * t
    u_xx = 6.0 * x * (t ** 3)
    return u_tt - u_xx + u ** 3


@dataclass(frozen=True)
class CLoReWConfig:
    config_id: str
    eta: float
    gamma: float
    alpha: float
    epsilon: float
    tau_gate: float
    eta_g: float
    rho_g: float
    kernel_h: float
    kernel_p: float
    kernel_trunc: float
    num_kernel_centers: int


CONFIGS = [
    CLoReWConfig("c00_current", 1e-3, 0.999, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.01, 2.0, 3.0, 41),
    CLoReWConfig("c01_h002", 1e-3, 0.999, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.02, 2.0, 3.0, 41),
    CLoReWConfig("c02_h005", 1e-3, 0.999, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.05, 2.0, 3.0, 41),
    CLoReWConfig("c03_g997_h002", 1e-3, 0.997, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.02, 2.0, 3.0, 41),
    CLoReWConfig("c04_eta5e4_h002", 5e-4, 0.999, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.02, 2.0, 3.0, 41),
    CLoReWConfig("c05_alpha1_gate3e5", 1e-3, 0.999, 1.0, 20.0, 0.0, 3e-5, 0.99, 0.02, 2.0, 3.0, 41),
    CLoReWConfig("c06_rho98_h002", 1e-3, 0.999, 2.0, 20.0, 0.0, 1e-5, 0.98, 0.02, 2.0, 3.0, 41),
    CLoReWConfig("c07_p1_h002", 1e-3, 0.999, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.02, 1.0, 3.0, 41),
    CLoReWConfig("c08_eta5e4_g997_h002", 5e-4, 0.997, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.02, 2.0, 3.0, 41),
    CLoReWConfig("c09_g995_h002", 1e-3, 0.995, 2.0, 20.0, 0.0, 1e-5, 0.99, 0.02, 2.0, 3.0, 41),
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
    def __init__(self, data, cfg, max_iters, eval_interval):
        self.cfg = cfg
        self.iter = 0
        self.exec_time = 0.0
        self.first_opt = max_iters
        self.eval_interval = eval_interval
        self.step_size = 5000
        self.it, self.l2, self.linf = [], [], []
        self.dimx_, self.dimt_ = data["nx"], data["nt"]
        self.exact = data["Exact0"]

        self.xx = torch.tensor(data["X_star"][:, 0:1]).float().to(device)
        self.tt = torch.tensor(data["X_star"][:, 1:2]).float().to(device)
        self.x_u = torch.tensor(data["X_u"][:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(data["X_u"][:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(data["X_f"][:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(data["X_f"][:, 1:2], requires_grad=True).float().to(device)
        self.x_lb = torch.tensor(data["X_lb"][:, 0:1], requires_grad=True).float().to(device)
        self.t_lb = torch.tensor(data["X_lb"][:, 1:2], requires_grad=True).float().to(device)
        self.x_ub = torch.tensor(data["X_ub"][:, 0:1], requires_grad=True).float().to(device)
        self.t_ub = torch.tensor(data["X_ub"][:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(data["u_u"]).float().to(device)
        self.ut = torch.tensor(data["ut_u"]).float().to(device)
        self.u_lb = torch.tensor(data["u_lb"]).float().to(device)
        self.u_ub = torch.tensor(data["u_ub"]).float().to(device)

        self.dnn = DNN([2, 50, 50, 50, 50, 50, 1]).to(device)
        self.rsum = 0
        self.init = 1
        self.gbar = torch.zeros_like(self.x_r).detach()
        self.kernel_eps = 1e-12
        self.gate_abs_res_ref = None

        t_min = float(data["lb"][1])
        t_max = float(data["ub"][1])
        self.t_centers = torch.linspace(t_min, t_max, cfg.num_kernel_centers, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

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
        u_tt = grad(u_t, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        f = u_tt - u_xx + u ** 3 - source_torch(x, t)
        return f, u_t

    def evaluate(self):
        with torch.no_grad():
            pred = self.net_u(self.xx, self.tt)
            sol = tonp(pred).reshape((self.dimt_, self.dimx_)).T
        exact_flat = self.exact.flatten()
        pred_flat = sol.flatten()
        l2 = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
        linf = np.linalg.norm(exact_flat - pred_flat, np.inf) / np.linalg.norm(exact_flat, np.inf)
        return float(l2), float(linf)

    def step(self):
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred, _ = self.net_r(self.x_r, self.t_r)
        _, ut0_pred = self.net_r(self.x_u, self.t_u)
        u_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred = self.net_u(self.x_ub, self.t_ub)

        eta = 1 if self.init == 2 and self.iter == 0 else self.cfg.eta
        g_t = 0.5 * (1.0 - torch.tanh(self.cfg.alpha * (self.t_r - self.cfg.tau_gate)))
        self.gbar = (self.cfg.rho_g * self.gbar + (1.0 - self.cfg.rho_g) * g_t.detach()).detach()

        abs_r = torch.abs(r_pred)
        abs_r_detach = abs_r.detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
        causal_r_norm = r_norm * self.gbar
        self.rsum = (self.rsum * self.cfg.gamma + causal_r_norm).detach()
        loss_r = torch.mean((self.rsum * r_pred) ** 2)

        gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.kernel_eps)
        if self.gate_abs_res_ref is None:
            self.gate_abs_res_ref = gate_abs_res.item() + self.kernel_eps
        gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref
        new_tau = self.cfg.tau_gate + self.cfg.eta_g * np.exp(-self.cfg.epsilon * gate_abs_res_norm)
        object.__setattr__(self.cfg, "tau_gate", float(new_tau))

        loss_u = torch.mean((u_pred - self.u) ** 2)
        loss_ut = torch.mean((ut0_pred - self.ut) ** 2)
        loss_b = 0.5 * (torch.mean((u_lb_pred - self.u_lb) ** 2) + torch.mean((u_ub_pred - self.u_ub) ** 2))
        loss = loss_r + loss_u + loss_ut + loss_b
        loss.backward()
        self.optimizer.step()
        self.iter += 1
        return float(loss.item())

    def train(self, log):
        self.dnn.train()
        last_loss = None
        for epoch in range(self.first_opt):
            start = time.time()
            last_loss = self.step()
            self.exec_time += time.time() - start

            if (epoch + 1) % self.step_size == 0:
                self.scheduler.step()

            if self.iter % self.eval_interval == 0 or self.iter == self.first_opt:
                l2, linf = self.evaluate()
                self.it.append(self.iter)
                self.l2.append(l2)
                self.linf.append(linf)
                log(
                    f"Iter {self.iter}, Loss: {last_loss:.3e}, Rel_L2: {l2:.3e}, "
                    f"L_inf: {linf:.3e}, tau_gate: {self.cfg.tau_gate:.6f}, "
                    f"gbar_mean: {float(torch.mean(self.gbar).item()):.3e}"
                )
        return last_loss


def build_data(seed, n_f):
    x_min, x_max = 0.0, 1.0
    t_min, t_max = 0.0, 1.0
    nx, nt = 256, 201
    x0 = np.linspace(x_min, x_max, nx)[:, None]
    t0 = np.linspace(t_min, t_max, nt)[:, None]
    X, T = np.meshgrid(x0.flatten(), t0.flatten())
    Exact = exact_u_np(X, T)
    Exact0 = Exact.T
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    x_u = x0
    t_u = np.zeros((nx, 1))
    X_u = np.hstack((x_u, t_u))
    u_u = exact_u_np(x_u, t_u)
    ut_u = exact_ut_np(x_u, t_u)

    t_b = t0
    x_lb = np.full_like(t_b, x_min)
    x_ub = np.full_like(t_b, x_max)
    X_lb = np.hstack((x_lb, t_b))
    X_ub = np.hstack((x_ub, t_b))
    u_lb = exact_u_np(x_lb, t_b)
    u_ub = exact_u_np(x_ub, t_b)

    lb = np.array([x_min, t_min])
    ub = np.array([x_max, t_max])
    X_f = lb + (ub - lb) * lhs_sample(2, n_f, seed)

    return {
        "nx": nx,
        "nt": nt,
        "Exact0": Exact0,
        "X_star": X_star,
        "X_u": X_u,
        "u_u": u_u,
        "ut_u": ut_u,
        "X_lb": X_lb,
        "X_ub": X_ub,
        "u_lb": u_lb,
        "u_ub": u_ub,
        "X_f": X_f,
        "lb": lb,
        "ub": ub,
    }


def read_baseline(base_dir, seed):
    baselines = {}
    for method, filename in {
        "PINN": f"losses_kg_pinn_{seed}.txt",
        "SA": f"losses_kg_sa_{seed}.txt",
        "RBA": f"losses_kg_rba_{seed}.txt",
        "C-LoReW_existing": f"losses_kg_clorew_pinn_{seed}.txt",
    }.items():
        path = os.path.join(base_dir, 'logs', filename)
        if os.path.exists(path):
            data = np.loadtxt(path)
            if data.ndim == 2 and data.shape[0] > 0:
                baselines[method] = (float(data[-1, 1]), float(data[-1, 2]))
    return baselines


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_config(base_dir, output_dir, seed, n_f, max_iters, eval_interval, cfg, force):
    result_path = os.path.join(output_dir, f"losses_kg_clorew_search_{cfg.config_id}_seed_{seed}.txt")
    log_path = os.path.join(output_dir, f"KG_C-LoReW_search_{cfg.config_id}_seed_{seed}.log")
    summary_path = os.path.join(output_dir, "KG_C_LoReW_hparam_search_results.csv")

    if os.path.exists(result_path) and not force:
        data = np.loadtxt(result_path)
        final_l2 = float(data[-1, 1])
        final_linf = float(data[-1, 2])
        return {"config_id": cfg.config_id, "seed": seed, "skipped": True, "final_l2": final_l2, "final_linf": final_linf}

    seed_torch(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = build_data(seed, n_f)
    cfg_runtime = CLoReWConfig(**asdict(cfg))

    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"Device: {device}")
        log(f"Seed: {seed}, N_f: {n_f}, max_iters: {max_iters}, eval_interval: {eval_interval}")
        log(f"Config: {asdict(cfg_runtime)}")
        model = CLoReWPINN(data, cfg_runtime, max_iters=max_iters, eval_interval=eval_interval)
        start = time.time()
        final_loss = model.train(log)
        wall_time = time.time() - start
        final_l2, final_linf = model.evaluate()
        log(f"Finished {cfg.config_id}: final_loss={final_loss:.6e}, final_l2={final_l2:.6e}, final_linf={final_linf:.6e}, wall_time={format_elapsed_time(wall_time)}")

    history = np.column_stack((np.array(model.it), np.array(model.l2), np.array(model.linf)))
    np.savetxt(result_path, history, fmt="%.10f %.10f %.10f")

    best_idx = int(np.argmin(model.l2))
    row = {
        "config_id": cfg.config_id,
        "seed": seed,
        "N_f": n_f,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "final_l2": final_l2,
        "final_linf": final_linf,
        "best_l2": float(np.min(model.l2)),
        "best_l2_iter": int(model.it[best_idx]),
        "wall_time_sec": float(wall_time),
        **asdict(cfg),
    }
    append_csv(summary_path, row)
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--n-f", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=40000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--configs", nargs="*", default=None, help="Config ids to run. Default: all.")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "C_LoReW_hparam_search")
    os.makedirs(output_dir, exist_ok=True)

    baselines = read_baseline(base_dir, args.seed)
    if baselines:
        print("Existing baseline logs found:")
        for method, (l2, linf) in baselines.items():
            print(f"  {method}: final_l2={l2:.6e}, final_linf={linf:.6e}")

    configs = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [cfg for cfg in configs if cfg.config_id in wanted]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    all_rows = []
    for cfg in configs:
        print(f"\n=== Running {cfg.config_id} ===", flush=True)
        row = run_config(
            base_dir=base_dir,
            output_dir=output_dir,
            seed=args.seed,
            n_f=args.n_f,
            max_iters=args.iters,
            eval_interval=args.eval_interval,
            cfg=cfg,
            force=args.force,
        )
        all_rows.append(row)

    ranked = sorted(all_rows, key=lambda item: item["final_l2"])
    print("\nSearch ranking by final L2RE:")
    for row in ranked:
        print(f"  {row['config_id']}: final_l2={row['final_l2']:.6e}, final_linf={row['final_linf']:.6e}")


if __name__ == "__main__":
    main()
