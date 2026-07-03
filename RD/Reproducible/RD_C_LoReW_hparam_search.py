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
from pyDOE import lhs as pydoe_lhs
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


def lhs(dim, samples, random_state=None):
    return pydoe_lhs(dim, samples, random_state=random_state)


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


D = 1.0
x_min, x_max = -np.pi, np.pi
t_min, t_max = 0.0, 1.0
nx, nt = 256 * 2, 201
N_i = nx
N_b = nt
N_u = N_i + 2 * N_b
N_f = 140
layers = [2] + 4 * [20] + [1]


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
    CLoReWConfig("c00_current", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.05, 8.0, 3.0, 41),
    CLoReWConfig("c01_eta5e4", 5e-4, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.05, 8.0, 3.0, 41),
    CLoReWConfig("c02_g997", 1e-3, 0.997, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.05, 8.0, 3.0, 41),
    CLoReWConfig("c03_eta5e4_g997", 5e-4, 0.997, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.05, 8.0, 3.0, 41),
    CLoReWConfig("c04_gate1e4", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-4, 0.95, 0.05, 8.0, 3.0, 41),
    CLoReWConfig("c05_p4_gate1e4", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-4, 0.95, 0.05, 4.0, 3.0, 41),
    CLoReWConfig("c06_p2_gate1e4", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-4, 0.95, 0.05, 2.0, 3.0, 41),
    CLoReWConfig("c07_eta5e4_g997_p4_gate1e4", 5e-4, 0.997, 5.0, 2.0, -0.5, 1e-4, 0.95, 0.05, 4.0, 3.0, 41),
    CLoReWConfig("c08_h003", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.03, 8.0, 3.0, 41),
    CLoReWConfig("c09_h008", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.08, 8.0, 3.0, 41),
    CLoReWConfig("c10_p6", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.05, 6.0, 3.0, 41),
    CLoReWConfig("c11_p10", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.05, 10.0, 3.0, 41),
    CLoReWConfig("c12_gate5e4", 1e-3, 0.999, 5.0, 2.0, -0.5, 5e-4, 0.95, 0.05, 8.0, 3.0, 41),
    CLoReWConfig("c13_rho99", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.99, 0.05, 8.0, 3.0, 41),
    CLoReWConfig("c14_h004", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.04, 8.0, 3.0, 41),
    CLoReWConfig("c15_h0035", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.035, 8.0, 3.0, 41),
    CLoReWConfig("c16_h0025", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.025, 8.0, 3.0, 41),
    CLoReWConfig("c17_h006", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.06, 8.0, 3.0, 41),
    CLoReWConfig("c18_h007", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.07, 8.0, 3.0, 41),
    CLoReWConfig("c19_h008_p10", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.08, 10.0, 3.0, 41),
    CLoReWConfig("c20_h008_p12", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.08, 12.0, 3.0, 41),
    CLoReWConfig("c21_h008_p6", 1e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.08, 6.0, 3.0, 41),
    CLoReWConfig("c22_h008_eta5e4", 5e-4, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.08, 8.0, 3.0, 41),
    CLoReWConfig("c23_h008_eta2e3", 2e-3, 0.999, 5.0, 2.0, -0.5, 1e-3, 0.95, 0.08, 8.0, 3.0, 41),
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
        self.loss = None
        self.Exact = data["Exact0"]
        self.tau_gate = cfg.tau_gate

        self.xx = torch.tensor(data["X_star"][:, 0:1], dtype=torch.float32, device=device)
        self.tt = torch.tensor(data["X_star"][:, 1:2], dtype=torch.float32, device=device)
        self.x_u = torch.tensor(data["X_u"][:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_u = torch.tensor(data["X_u"][:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.x_r = torch.tensor(data["X_f"][:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_r = torch.tensor(data["X_f"][:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(data["u"], dtype=torch.float32, device=device)

        self.dnn = DNN(layers).to(device)
        self.rsum = 0
        self.init = 1
        self.gbar = torch.zeros_like(self.x_r).detach()
        self.kernel_eps = 1e-12
        self.gate_abs_res_ref = None

        self.t_centers = torch.linspace(t_min, t_max, cfg.num_kernel_centers, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-4, betas=(0.9, 0.999))
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
        center_moment_p = (self.kernel_weights.T @ (abs_r_detach ** self.cfg.kernel_p)) / self.kernel_col_mass
        local_moment_p = self.kernel_weights @ center_moment_p
        return (local_moment_p + self.kernel_eps) ** (1.0 / self.cfg.kernel_p)

    def net_u(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        return u_t - D * u_xx - reaction_torch(x, t)

    def evaluate(self):
        with torch.no_grad():
            pred = self.net_u(self.xx, self.tt)
            sol = tonp(pred).reshape((nt, nx)).T
        exact_flat = self.Exact.flatten()
        pred_flat = sol.flatten()
        l2 = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
        linf = np.linalg.norm(exact_flat - pred_flat, np.inf) / np.linalg.norm(exact_flat, np.inf)
        return float(l2), float(linf)

    def step(self):
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)

        eta = 1 if self.init == 2 and self.iter == 0 else self.cfg.eta
        g_t = 0.5 * (1.0 - torch.tanh(self.cfg.alpha * (self.t_r - self.tau_gate)))
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
        self.tau_gate = self.tau_gate + self.cfg.eta_g * np.exp(-self.cfg.epsilon * gate_abs_res_norm)

        loss_u = torch.mean((u_pred - self.u) ** 2)
        self.loss = loss_r + loss_u
        self.loss.backward()
        self.optimizer.step()
        self.iter += 1
        return float(self.loss.item())

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
                    f"L_inf: {linf:.3e}, tau_gate: {self.tau_gate:.3f}, "
                    f"gbar_mean: {float(torch.mean(self.gbar).item()):.3e}"
                )
        return last_loss


def build_data(seed):
    seed_torch(seed)
    x0 = np.linspace(x_min, x_max, nx)[:, None]
    t0 = np.linspace(t_min, t_max, nt)[:, None]
    X, T = np.meshgrid(x0.flatten(), t0.flatten())
    Exact = exact_u_np(X, T)
    Exact0 = Exact.T
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    X_i = np.hstack((X[0:1, :].T, T[0:1, :].T))
    u_i = Exact[0:1, :].T
    X_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    u_lb = Exact[:, 0:1]
    X_ub = np.hstack((X[:, -1:], T[:, -1:]))
    u_ub = Exact[:, -1:]
    X_u = np.vstack([X_i, X_lb, X_ub])
    u = np.vstack([u_i, u_lb, u_ub])

    lb = X_star.min(0)
    ub = X_star.max(0)
    X_f = lb + (ub - lb) * lhs(2, N_f, random_state=seed)
    X_f = np.vstack((X_f, X_u))

    return {"Exact0": Exact0, "X_star": X_star, "X_u": X_u, "u": u, "X_f": X_f}


def read_baselines(base_dir, seed):
    baselines = {}
    for method, filename in {
        "PINN": f"losses_rd_pinn_{seed}.txt",
        "SA": f"losses_rd_sa_{seed}.txt",
        "RBA": f"losses_rd_rba_{seed}.txt",
        "C-LoReW_existing": f"losses_rd_clorew_pinn_{seed}.txt",
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


def run_config(base_dir, output_dir, seed, cfg, max_iters, eval_interval, force):
    result_path = os.path.join(
        output_dir,
        f"losses_rd_clorew_search_{cfg.config_id}_iters_{max_iters}_seed_{seed}.txt",
    )
    log_path = os.path.join(
        output_dir,
        f"RD_C-LoReW_search_{cfg.config_id}_iters_{max_iters}_seed_{seed}.log",
    )
    summary_path = os.path.join(output_dir, "RD_C_LoReW_hparam_search_results.csv")

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
        log(f"Seed: {seed}, max_iters: {max_iters}, eval_interval: {eval_interval}")
        log(f"Config: {asdict(cfg)}")
        model = CLoReWPINN(data, cfg, max_iters=max_iters, eval_interval=eval_interval)
        start = time.time()
        final_loss = model.train(log)
        wall_time = time.time() - start
        final_l2, final_linf = model.evaluate()
        log(
            f"Finished {cfg.config_id}: final_loss={final_loss:.6e}, "
            f"final_l2={final_l2:.6e}, final_linf={final_linf:.6e}, wall_time={format_elapsed_time(wall_time)}"
        )

    history = np.column_stack((np.array(model.it), np.array(model.l2), np.array(model.linf)))
    np.savetxt(result_path, history, fmt="%.10f %.10f %.10f")

    best_idx = int(np.argmin(model.l2))
    row = {
        "config_id": cfg.config_id,
        "seed": seed,
        "N_f": N_f,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "final_l2": final_l2,
        "final_linf": final_linf,
        "best_l2": float(np.min(model.l2)),
        "best_l2_iter": int(model.it[best_idx]),
        "wall_time_sec": float(wall_time),
        "tau_gate_final": float(model.tau_gate),
        "gbar_mean_final": float(torch.mean(model.gbar).item()),
        **asdict(cfg),
    }
    append_csv(summary_path, row)
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "C_LoReW_hparam_search")
    os.makedirs(output_dir, exist_ok=True)

    configs = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [cfg for cfg in configs if cfg.config_id in wanted]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    seeds = args.seeds if args.seeds else [args.seed]
    rows = []
    for seed in seeds:
        print(f"\n##### Seed {seed} #####", flush=True)
        baselines = read_baselines(base_dir, seed)
        if baselines:
            print("Existing baseline logs found:")
            for method, (l2, linf) in baselines.items():
                print(f"  {method}: final_l2={l2:.6e}, final_linf={linf:.6e}")

        for cfg in configs:
            print(f"\n=== Running {cfg.config_id} ===", flush=True)
            row = run_config(
                base_dir=base_dir,
                output_dir=output_dir,
                seed=seed,
                cfg=cfg,
                max_iters=args.iters,
                eval_interval=args.eval_interval,
                force=args.force,
            )
            rows.append(row)

    ranked = sorted(rows, key=lambda item: item["final_l2"])
    print("\nSearch ranking by final L2RE:")
    for row in ranked:
        print(
            f"  seed={row['seed']}, {row['config_id']}: "
            f"final_l2={row['final_l2']:.6e}, final_linf={row['final_linf']:.6e}"
        )


if __name__ == "__main__":
    main()
