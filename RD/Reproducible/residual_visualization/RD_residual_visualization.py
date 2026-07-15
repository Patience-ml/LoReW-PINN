import argparse
import os
import random
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pyDOE import lhs as _lhs

    def latin_hypercube(dim: int, samples: int, seed: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return _lhs(dim, samples, random_state=seed)

except ImportError:
    from scipy.stats import qmc

    def latin_hypercube(dim: int, samples: int, seed: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        return sampler.random(samples)


plt.rcParams.update(
    {
        "figure.max_open_warning": 4,
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


@dataclass(frozen=True)
class Config:
    seed: int
    diffusion: float
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    grid_size_x: int
    grid_size_t: int
    n_initial: int
    n_boundary: int
    n_residual: int
    learning_rate: float
    scheduler_gamma: float
    scheduler_step_size: int
    layers: Tuple[int, ...]
    rba_eta: float
    rba_gamma: float
    sa_weight_lr: float
    clorew_eta: float
    clorew_gamma: float
    alpha: float
    epsilon: float
    tau0: float
    eta_g: float
    rho_g: float
    kernel_h: float
    kernel_p: float
    kernel_trunc: float
    num_kernel_centers: int
    kernel_eps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reaction-diffusion residual-evolution heatmaps for Vanilla PINN, "
            "SA-PINN, RBA-PINN, and C-LoReW-PINN."
        )
    )
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 5000, 15000])
    parser.add_argument("--grid-size-x", type=int, default=201)
    parser.add_argument("--grid-size-t", type=int, default=201)
    parser.add_argument("--n-initial", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=201)
    parser.add_argument("--n-residual", type=int, default=140)
    parser.add_argument("--seed", type=int, default=3053)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    parser.add_argument("--scheduler-step-size", type=int, default=5000)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--residual-eps", type=float, default=1e-12)
    parser.add_argument("--output-name", type=str, default="rd_residual_evolution")
    parser.add_argument("--no-vanilla", action="store_true")
    parser.add_argument("--show-contours", action="store_true")
    parser.add_argument("--value-mode", choices=["raw", "log"], default="raw")
    parser.add_argument("--scale-mode", choices=["global", "column", "panel"], default="column")
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        default=[0.0, 100.0],
        metavar=("LOW", "HIGH"),
        help="Color-scale clipping percentiles. The default 0 100 uses the full range.",
    )

    parser.add_argument("--rba-eta", type=float, default=1e-3)
    parser.add_argument("--rba-gamma", type=float, default=0.999)
    parser.add_argument("--sa-weight-lr", type=float, default=0.005)
    parser.add_argument("--clorew-eta", type=float, default=1e-3)
    parser.add_argument("--clorew-gamma", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--tau0", type=float, default=-0.5)
    parser.add_argument("--eta-g", type=float, default=0.001)
    parser.add_argument("--rho-g", type=float, default=0.95)
    parser.add_argument("--kernel-h", type=float, default=0.05)
    parser.add_argument("--kernel-p", type=float, default=8.0)
    parser.add_argument("--kernel-trunc", type=float, default=3.0)
    parser.add_argument("--num-kernel-centers", type=int, default=41)
    parser.add_argument("--kernel-eps", type=float, default=1e-12)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def tonp(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    raise TypeError("Unknown type of input, expected torch.Tensor or np.ndarray.")


def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]


def exact_u_np(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.exp(-t) * (
        np.sin(x)
        + 0.5 * np.sin(2.0 * x)
        + np.sin(3.0 * x) / 3.0
        + 0.25 * np.sin(4.0 * x)
        + 0.125 * np.sin(8.0 * x)
    )


def reaction_torch(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-t) * (
        1.5 * torch.sin(2.0 * x)
        + (8.0 / 3.0) * torch.sin(3.0 * x)
        + (15.0 / 4.0) * torch.sin(4.0 * x)
        + (63.0 / 8.0) * torch.sin(8.0 * x)
    )


def make_problem(cfg: Config) -> Dict[str, np.ndarray]:
    x_eval = np.linspace(cfg.x_min, cfg.x_max, cfg.grid_size_x)[:, None]
    t_eval = np.linspace(cfg.t_min, cfg.t_max, cfg.grid_size_t)[:, None]
    X_eval, T_eval = np.meshgrid(x_eval.flatten(), t_eval.flatten(), indexing="ij")
    Exact = exact_u_np(X_eval, T_eval)
    X_star = np.hstack((X_eval.flatten()[:, None], T_eval.flatten()[:, None]))

    x_ic = np.linspace(cfg.x_min, cfg.x_max, cfg.n_initial)[:, None]
    t_bc = np.linspace(cfg.t_min, cfg.t_max, cfg.n_boundary)[:, None]
    X_i = np.hstack((x_ic, np.zeros_like(x_ic)))
    u_i = exact_u_np(X_i[:, 0:1], X_i[:, 1:2])
    X_lb = np.hstack((np.full((cfg.n_boundary, 1), cfg.x_min), t_bc))
    u_lb = exact_u_np(X_lb[:, 0:1], X_lb[:, 1:2])
    X_ub = np.hstack((np.full((cfg.n_boundary, 1), cfg.x_max), t_bc))
    u_ub = exact_u_np(X_ub[:, 0:1], X_ub[:, 1:2])
    X_u = np.vstack([X_i, X_lb, X_ub])
    u = np.vstack([u_i, u_lb, u_ub])

    lb = np.array([cfg.x_min, cfg.t_min])
    ub = np.array([cfg.x_max, cfg.t_max])
    X_f = lb + (ub - lb) * latin_hypercube(2, cfg.n_residual, cfg.seed)
    X_f = np.vstack((X_f, X_u))

    return {
        "x_eval": x_eval,
        "t_eval": t_eval,
        "X_eval": X_eval,
        "T_eval": T_eval,
        "Exact": Exact,
        "X_star": X_star,
        "X_u": X_u,
        "u": u,
        "X_f": X_f,
    }


class DNN(torch.nn.Module):
    def __init__(self, layers: Tuple[int, ...]):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BaseRunner:
    name = "Base"

    def __init__(self, cfg: Config, device: torch.device, X_u, u, X_r):
        self.cfg = cfg
        self.device = device
        self.iter = 0

        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.N_r = self.x_r.shape[0]
        self.N_u = self.u.shape[0]

        seed_torch(cfg.seed)
        self.dnn = DNN(cfg.layers).to(device)
        self.optimizer = torch.optim.Adam(
            self.dnn.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999)
        )
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.scheduler_gamma)

    def net_u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        return u_t - self.cfg.diffusion * u_xx - reaction_torch(x, t)

    def residual_grid(
        self, X_star: np.ndarray, grid_shape: Tuple[int, int], chunk_size: int
    ) -> np.ndarray:
        self.dnn.eval()
        residual_chunks: List[np.ndarray] = []
        for start in range(0, X_star.shape[0], chunk_size):
            pts = X_star[start : start + chunk_size]
            x = torch.tensor(pts[:, 0:1], requires_grad=True).float().to(self.device)
            t = torch.tensor(pts[:, 1:2], requires_grad=True).float().to(self.device)
            r = self.net_r(x, t)
            residual_chunks.append(tonp(torch.abs(r)).reshape(-1))

        self.dnn.train()
        return np.concatenate(residual_chunks, axis=0).reshape(grid_shape)

    def train_step(self) -> None:
        raise NotImplementedError

    def maybe_step_scheduler(self) -> None:
        if self.iter % self.cfg.scheduler_step_size == 0:
            self.scheduler.step()

    def train_and_collect_residuals(
        self,
        checkpoints: List[int],
        X_star: np.ndarray,
        grid_shape: Tuple[int, int],
        chunk_size: int,
    ) -> Dict[int, np.ndarray]:
        self.dnn.train()
        checkpoint_set = set(checkpoints)
        collected = {}
        max_steps = max(checkpoints)
        start_time = time.time()
        for _ in range(max_steps):
            self.train_step()
            self.maybe_step_scheduler()
            if self.iter in checkpoint_set:
                elapsed = time.time() - start_time
                print(f"  {self.name}: collected residuals at iteration {self.iter} ({elapsed:.1f}s)")
                collected[self.iter] = self.residual_grid(X_star, grid_shape, chunk_size)
        return collected


class VanillaPINNRunner(BaseRunner):
    name = "Vanilla PINN"

    def train_step(self) -> None:
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)
        loss_r = torch.mean(r_pred**2)
        loss_u = torch.mean((u_pred - self.u) ** 2)
        loss = loss_r + loss_u
        loss.backward()
        self.iter += 1
        self.optimizer.step()


class SAPINNRunner(BaseRunner):
    name = "SA-PINN"

    def __init__(self, cfg: Config, device: torch.device, X_u, u, X_r):
        super().__init__(cfg, device, X_u, u, X_r)
        lamr = torch.rand(self.N_r, 1).float().to(device) * 1.0
        lamu = torch.rand(self.N_u, 1).float().to(device) * 100.0
        self.lamr = torch.nn.Parameter(lamr)
        self.lamu = torch.nn.Parameter(lamu)
        self.optimizer2 = torch.optim.Adam(
            [self.lamr, self.lamu], lr=cfg.sa_weight_lr, maximize=True
        )

    def train_step(self) -> None:
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)
        loss_r = torch.mean((self.lamr * r_pred) ** 2)
        loss_u = torch.mean((self.lamu * (u_pred - self.u)) ** 2)
        loss = loss_r + loss_u
        loss.backward()
        self.iter += 1
        self.optimizer.step()
        self.optimizer2.step()


class RBAPINNRunner(BaseRunner):
    name = "RBA-PINN"

    def __init__(self, cfg: Config, device: torch.device, X_u, u, X_r):
        super().__init__(cfg, device, X_u, u, X_r)
        self.rsum = torch.zeros((self.N_r, 1), device=device)

    def train_step(self) -> None:
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)
        abs_r = torch.abs(r_pred)
        r_norm = self.cfg.rba_eta * abs_r / (torch.max(abs_r) + self.cfg.kernel_eps)
        self.rsum = (self.rsum * self.cfg.rba_gamma + r_norm).detach()
        loss_r = torch.mean((self.rsum * r_pred) ** 2)
        loss_u = torch.mean((u_pred - self.u) ** 2)
        loss = loss_r + loss_u
        loss.backward()
        self.iter += 1
        self.optimizer.step()


class CLoReWPINNRunner(BaseRunner):
    name = "C-LoReW-PINN"

    def __init__(self, cfg: Config, device: torch.device, X_u, u, X_r):
        super().__init__(cfg, device, X_u, u, X_r)
        self.rsum = torch.zeros((self.N_r, 1), device=device)
        self.tau_gate = cfg.tau0
        self.gbar = torch.zeros_like(self.x_r).detach()
        self.gate_abs_res_ref = None

        self.t_centers = torch.linspace(cfg.t_min, cfg.t_max, cfg.num_kernel_centers, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + cfg.kernel_eps

    def build_kernel_weights(self, t_points: torch.Tensor) -> torch.Tensor:
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
        return K / (row_sum + self.cfg.kernel_eps)

    def kernel_local_scale(self, abs_r_detach: torch.Tensor) -> torch.Tensor:
        center_moment_p = (self.kernel_weights.T @ (abs_r_detach**self.cfg.kernel_p)) / (
            self.kernel_col_mass
        )
        local_moment_p = self.kernel_weights @ center_moment_p
        return (local_moment_p + self.cfg.kernel_eps) ** (1.0 / self.cfg.kernel_p)

    def train_step(self) -> None:
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)

        g_t = 0.5 * (1.0 - torch.tanh(self.cfg.alpha * (self.t_r - self.tau_gate)))
        self.gbar = (self.cfg.rho_g * self.gbar + (1.0 - self.cfg.rho_g) * g_t.detach()).detach()

        abs_r = torch.abs(r_pred)
        abs_r_detach = abs_r.detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = self.cfg.clorew_eta * abs_r_detach / (local_scale + self.cfg.kernel_eps)
        causal_r_norm = r_norm * self.gbar
        self.rsum = (self.rsum * self.cfg.clorew_gamma + causal_r_norm).detach()
        loss_r = torch.mean((self.rsum * r_pred) ** 2)

        gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.cfg.kernel_eps)
        if self.gate_abs_res_ref is None:
            self.gate_abs_res_ref = gate_abs_res.item() + self.cfg.kernel_eps
        gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref
        self.tau_gate = self.tau_gate + self.cfg.eta_g * np.exp(-self.cfg.epsilon * gate_abs_res_norm)

        loss_u = torch.mean((u_pred - self.u) ** 2)
        loss = loss_r + loss_u
        loss.backward()
        self.iter += 1
        self.optimizer.step()


def build_config(args: argparse.Namespace) -> Config:
    layers = tuple([2] + args.hidden_layers * [args.hidden_width] + [1])
    return Config(
        seed=args.seed,
        diffusion=1.0,
        x_min=-np.pi,
        x_max=np.pi,
        t_min=0.0,
        t_max=1.0,
        grid_size_x=args.grid_size_x,
        grid_size_t=args.grid_size_t,
        n_initial=args.n_initial,
        n_boundary=args.n_boundary,
        n_residual=args.n_residual,
        learning_rate=args.learning_rate,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_step_size=args.scheduler_step_size,
        layers=layers,
        rba_eta=args.rba_eta,
        rba_gamma=args.rba_gamma,
        sa_weight_lr=args.sa_weight_lr,
        clorew_eta=args.clorew_eta,
        clorew_gamma=args.clorew_gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        tau0=args.tau0,
        eta_g=args.eta_g,
        rho_g=args.rho_g,
        kernel_h=args.kernel_h,
        kernel_p=args.kernel_p,
        kernel_trunc=args.kernel_trunc,
        num_kernel_centers=args.num_kernel_centers,
        kernel_eps=args.kernel_eps,
    )


def plot_residual_evolution(
    all_residuals: Dict[str, Dict[int, np.ndarray]],
    checkpoints: List[int],
    problem: Dict[str, np.ndarray],
    cfg: Config,
    args: argparse.Namespace,
    figure_dir: str,
    data_dir: str,
) -> None:
    low, high = args.clip_percentiles
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("--clip-percentiles must satisfy 0 <= LOW < HIGH <= 100.")

    def display_map(residual_map: np.ndarray) -> np.ndarray:
        if args.value_mode == "log":
            return np.log10(residual_map + args.residual_eps)
        return residual_map

    method_names = list(all_residuals.keys())
    value_maps: Dict[str, Dict[int, np.ndarray]] = {
        name: {step: display_map(all_residuals[name][step]) for step in checkpoints}
        for name in method_names
    }

    def limits_from_arrays(arrays: List[np.ndarray]) -> Tuple[float, float]:
        stacked = np.concatenate([arr.reshape(-1) for arr in arrays], axis=0)
        vmin, vmax = np.percentile(stacked, [low, high])
        return float(vmin), float(vmax)

    panel_limits: Dict[Tuple[str, int], Tuple[float, float]] = {}
    if args.scale_mode == "global":
        vmin, vmax = limits_from_arrays(
            [value_maps[name][step] for name in method_names for step in checkpoints]
        )
        for name in method_names:
            for step in checkpoints:
                panel_limits[(name, step)] = (vmin, vmax)
    elif args.scale_mode == "column":
        for step in checkpoints:
            vmin, vmax = limits_from_arrays([value_maps[name][step] for name in method_names])
            for name in method_names:
                panel_limits[(name, step)] = (vmin, vmax)
    else:
        for name in method_names:
            for step in checkpoints:
                panel_limits[(name, step)] = limits_from_arrays([value_maps[name][step]])

    n_rows = len(all_residuals)
    n_cols = len(checkpoints)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5.0 * n_cols, 9.0), max(3.35 * n_rows, 6.0)),
        squeeze=False,
    )

    x = problem["x_eval"].flatten()
    t = problem["t_eval"].flatten()
    exact = problem["Exact"]

    for r_idx, (method_name, _) in enumerate(all_residuals.items()):
        for c_idx, step in enumerate(checkpoints):
            ax = axes[r_idx, c_idx]
            residual_display = value_maps[method_name][step]
            vmin, vmax = panel_limits[(method_name, step)]
            im = ax.imshow(
                residual_display,
                cmap="rainbow",
                aspect="auto",
                extent=[cfg.t_min, cfg.t_max, cfg.x_min, cfg.x_max],
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            if args.show_contours:
                ax.contour(
                    t,
                    x,
                    exact,
                    levels=5,
                    colors="white",
                    alpha=0.45,
                    linestyles="dashed",
                    linewidths=0.8,
                )

            if r_idx == 0:
                ax.set_title(f"Iteration {step}", fontsize=12, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"{method_name}\nx", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("x", fontsize=10)
            ax.set_xlabel("t", fontsize=10)
            ax.tick_params(labelsize=9)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.06, top=0.94, wspace=0.28, hspace=0.32)

    png_path = os.path.join(figure_dir, f"{args.output_name}.png")
    pdf_path = os.path.join(figure_dir, f"{args.output_name}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    npz_path = os.path.join(data_dir, f"{args.output_name}.npz")
    np.savez_compressed(
        npz_path,
        checkpoints=np.array(checkpoints),
        method_names=np.array(method_names),
        x=problem["x_eval"].flatten(),
        t=problem["t_eval"].flatten(),
        exact=problem["Exact"],
        residual_maps=np.array(
            [[all_residuals[name][step] for step in checkpoints] for name in method_names],
            dtype=np.float64,
        ),
        value_mode=args.value_mode,
        scale_mode=args.scale_mode,
        clip_percentiles=np.array(args.clip_percentiles, dtype=np.float64),
        panel_limits=np.array(
            [[panel_limits[(name, step)] for step in checkpoints] for name in method_names],
            dtype=np.float64,
        ),
    )

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved residual data: {npz_path}")


def main() -> None:
    args = parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("Checkpoints must be positive integers.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    seed_torch(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cfg = build_config(args)
    problem = make_problem(cfg)
    grid_shape = (cfg.grid_size_x, cfg.grid_size_t)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figure_dir = os.path.join(script_dir, "figures")
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    runner_specs = []
    if not args.no_vanilla:
        runner_specs.append(("Vanilla PINN", VanillaPINNRunner))
    runner_specs.extend(
        [
            ("SA-PINN", SAPINNRunner),
            ("RBA-PINN", RBAPINNRunner),
            ("C-LoReW-PINN", CLoReWPINNRunner),
        ]
    )

    all_residuals = {}
    for method_name, runner_cls in runner_specs:
        print(f"Training {method_name} and collecting residual fields...")
        runner = runner_cls(cfg, device, problem["X_u"], problem["u"], problem["X_f"])
        all_residuals[method_name] = runner.train_and_collect_residuals(
            checkpoints=checkpoints,
            X_star=problem["X_star"],
            grid_shape=grid_shape,
            chunk_size=args.chunk_size,
        )

    plot_residual_evolution(
        all_residuals=all_residuals,
        checkpoints=checkpoints,
        problem=problem,
        cfg=cfg,
        args=args,
        figure_dir=figure_dir,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()
