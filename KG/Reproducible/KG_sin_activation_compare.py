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
    eta: float = 5e-4
    gamma: float = 0.997
    alpha: float = 2.0
    epsilon: float = 20.0
    tau_gate: float = 0.0
    eta_g: float = 1e-5
    rho_g: float = 0.99
    kernel_h: float = 0.02
    kernel_p: float = 2.0
    kernel_trunc: float = 3.0
    num_kernel_centers: int = 41


class SinActivation(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.depth = len(layers) - 1

        layer_list = []
        for i in range(self.depth - 1):
            layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(layer.weight)
            layer_list.append((f"layer_{i}", layer))
            layer_list.append((f"activation_{i}", SinActivation()))

        layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(layer.weight)
        layer_list.append((f"layer_{self.depth - 1}", layer))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


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


class KGModel:
    def __init__(self, data, method, clorew_cfg, max_iters, eval_interval):
        self.method = method
        self.clorew_cfg = clorew_cfg
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
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]
        self.N_b = self.x_lb.shape[0]

        self.dnn = DNN([2, 50, 50, 50, 50, 50, 1]).to(device)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        if self.method == "SA":
            lamr = torch.rand(self.N_r, 1, requires_grad=True).float().to(device) * 1.0
            lamu = torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 100.0
            lamut = torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 100.0
            lamb = torch.rand(self.N_b, 1, requires_grad=True).float().to(device) * 100.0
            self.lamr = torch.nn.Parameter(lamr)
            self.lamu = torch.nn.Parameter(lamu)
            self.lamut = torch.nn.Parameter(lamut)
            self.lamb = torch.nn.Parameter(lamb)
            self.optimizer2 = torch.optim.Adam([self.lamr, self.lamu, self.lamut, self.lamb], lr=0.005, maximize=True)

        if self.method == "RBA":
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
            self.init = 1

        if self.method == "C-LoReW":
            self.rsum = 0
            self.init = 1
            self.gbar = torch.zeros_like(self.x_r).detach()
            self.kernel_eps = 1e-12
            self.gate_abs_res_ref = None
            self.t_centers = torch.linspace(
                float(data["lb"][1]),
                float(data["ub"][1]),
                self.clorew_cfg.num_kernel_centers,
                device=device,
            ).view(1, -1)
            self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
            self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + self.kernel_eps

    def build_kernel_weights(self, t_points):
        diff = t_points - self.t_centers
        K = torch.exp(-0.5 * (diff / self.clorew_cfg.kernel_h) ** 2)
        if self.clorew_cfg.kernel_trunc is not None and self.clorew_cfg.kernel_trunc > 0:
            K = K * (torch.abs(diff) <= self.clorew_cfg.kernel_trunc * self.clorew_cfg.kernel_h).float()

        row_sum = K.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            nearest_idx = torch.argmin(torch.abs(diff[zero_rows]), dim=1)
            K[zero_rows] = 0.0
            K[zero_rows, nearest_idx] = 1.0
            row_sum = K.sum(dim=1, keepdim=True)

        return K / (row_sum + self.kernel_eps)

    def kernel_local_scale(self, abs_r_detach):
        moment = (self.kernel_weights.T @ (abs_r_detach ** self.clorew_cfg.kernel_p)) / self.kernel_col_mass
        local_moment = self.kernel_weights @ moment
        return (local_moment + self.kernel_eps) ** (1.0 / self.clorew_cfg.kernel_p)

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

    def weighted_residual_loss(self, r_pred):
        if self.method == "RBA":
            eta = 1 if self.init == 2 and self.iter == 0 else self.eta
            r_norm = eta * torch.abs(r_pred) / torch.max(torch.abs(r_pred))
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            return torch.mean((self.rsum * r_pred) ** 2)

        if self.method == "C-LoReW":
            eta = 1 if self.init == 2 and self.iter == 0 else self.clorew_cfg.eta
            g_t = 0.5 * (1.0 - torch.tanh(self.clorew_cfg.alpha * (self.t_r - self.clorew_cfg.tau_gate)))
            self.gbar = (self.clorew_cfg.rho_g * self.gbar + (1.0 - self.clorew_cfg.rho_g) * g_t.detach()).detach()

            abs_r_detach = torch.abs(r_pred).detach()
            local_scale = self.kernel_local_scale(abs_r_detach)
            r_norm = eta * abs_r_detach / (local_scale + self.kernel_eps)
            causal_r_norm = r_norm * self.gbar
            self.rsum = (self.rsum * self.clorew_cfg.gamma + causal_r_norm).detach()

            gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.kernel_eps)
            if self.gate_abs_res_ref is None:
                self.gate_abs_res_ref = gate_abs_res.item() + self.kernel_eps
            gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref
            new_tau = self.clorew_cfg.tau_gate + self.clorew_cfg.eta_g * np.exp(-self.clorew_cfg.epsilon * gate_abs_res_norm)
            object.__setattr__(self.clorew_cfg, "tau_gate", float(new_tau))
            return torch.mean((self.rsum * r_pred) ** 2)

        return torch.mean(r_pred ** 2)

    def step(self):
        self.optimizer.zero_grad()
        if self.method == "SA":
            self.optimizer2.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred, _ = self.net_r(self.x_r, self.t_r)
        _, ut0_pred = self.net_r(self.x_u, self.t_u)
        u_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred = self.net_u(self.x_ub, self.t_ub)

        if self.method == "SA":
            loss_r = torch.mean((self.lamr * r_pred) ** 2)
            loss_u = torch.mean((self.lamu * (u_pred - self.u)) ** 2)
            loss_ut = torch.mean((self.lamut * (ut0_pred - self.ut)) ** 2)
            loss_b = 0.5 * (
                torch.mean((self.lamb * (u_lb_pred - self.u_lb)) ** 2)
                + torch.mean((self.lamb * (u_ub_pred - self.u_ub)) ** 2)
            )
        else:
            loss_r = self.weighted_residual_loss(r_pred)
            loss_u = torch.mean((u_pred - self.u) ** 2)
            loss_ut = torch.mean((ut0_pred - self.ut) ** 2)
            loss_b = 0.5 * (torch.mean((u_lb_pred - self.u_lb) ** 2) + torch.mean((u_ub_pred - self.u_ub) ** 2))

        loss = loss_r + loss_u + loss_ut + loss_b
        loss.backward()
        self.optimizer.step()
        if self.method == "SA":
            self.optimizer2.step()
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
                extra = ""
                if self.method == "C-LoReW":
                    extra = f", tau_gate: {self.clorew_cfg.tau_gate:.6f}, gbar_mean: {float(torch.mean(self.gbar).item()):.3e}"
                log(f"Iter {self.iter}, Loss: {last_loss:.3e}, Rel_L2: {l2:.3e}, L_inf: {linf:.3e}{extra}, Time: {format_elapsed_time(self.exec_time)}")
        return last_loss


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_method(output_dir, seed, n_f, max_iters, eval_interval, method, force):
    result_path = os.path.join(output_dir, f"losses_kg_{method.lower()}_sin_seed_{seed}.txt")
    log_path = os.path.join(output_dir, f"KG_{method}_sin_train_log_{seed}.txt")
    summary_path = os.path.join(output_dir, f"KG_sin_activation_summary_seed_{seed}.csv")

    if os.path.exists(result_path) and not force:
        history = np.loadtxt(result_path)
        if history.ndim == 1:
            history = history[None, :]
        final_l2 = float(history[-1, 1])
        final_linf = float(history[-1, 2])
        return {
            "method": method,
            "seed": seed,
            "activation": "sin",
            "N_f": n_f,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "final_l2": final_l2,
            "final_linf": final_linf,
            "best_l2": float(np.min(history[:, 1])),
            "best_l2_iter": int(history[int(np.argmin(history[:, 1])), 0]),
            "wall_time_sec": 0.0,
            "skipped": True,
        }

    seed_torch(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = build_data(seed, n_f)
    cfg = CLoReWConfig()

    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(f"Device: {device}")
        log(f"Method: {method}, activation: sin")
        log(f"Seed: {seed}, N_f: {n_f}, max_iters: {max_iters}, eval_interval: {eval_interval}")
        if method == "C-LoReW":
            log(f"C-LoReW config: {asdict(cfg)}")

        model = KGModel(data, method, cfg, max_iters=max_iters, eval_interval=eval_interval)
        start = time.time()
        final_loss = model.train(log)
        wall_time = time.time() - start
        final_l2, final_linf = model.evaluate()
        log(
            f"Finished {method}: final_loss={final_loss:.6e}, final_l2={final_l2:.6e}, "
            f"final_linf={final_linf:.6e}, wall_time={format_elapsed_time(wall_time)}"
        )

    history = np.column_stack((np.array(model.it), np.array(model.l2), np.array(model.linf)))
    np.savetxt(result_path, history, fmt="%.10f %.10f %.10f")
    best_idx = int(np.argmin(model.l2))

    row = {
        "method": method,
        "seed": seed,
        "activation": "sin",
        "N_f": n_f,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "final_l2": final_l2,
        "final_linf": final_linf,
        "best_l2": float(np.min(model.l2)),
        "best_l2_iter": int(model.it[best_idx]),
        "wall_time_sec": float(wall_time),
        "skipped": False,
    }
    append_csv(summary_path, row)
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3899)
    parser.add_argument("--n-f", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=40000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--methods", nargs="*", default=["PINN", "SA", "RBA", "C-LoReW"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "sin_activation_compare")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for method in args.methods:
        method = method.upper()
        if method not in {"PINN", "SA", "RBA", "C-LoReW"}:
            raise ValueError(f"Unknown method: {method}")
        print(f"\n=== Running {method} with sin activation ===", flush=True)
        rows.append(
            run_method(
                output_dir=output_dir,
                seed=args.seed,
                n_f=args.n_f,
                max_iters=args.iters,
                eval_interval=args.eval_interval,
                method=method,
                force=args.force,
            )
        )

    print("\nFinal ranking by L2RE:")
    for row in sorted(rows, key=lambda item: item["final_l2"]):
        print(
            f"  {row['method']}: final_l2={row['final_l2']:.6e}, "
            f"final_linf={row['final_linf']:.6e}, best_l2={row['best_l2']:.6e} @ {row['best_l2_iter']}"
        )


if __name__ == "__main__":
    main()
