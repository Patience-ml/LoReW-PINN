import argparse
import os
import random
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

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
    nu: float
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
            "Generate Burgers residual-evolution heatmaps for Vanilla PINN, "
            "SA-PINN, RBA-PINN, and C-LoReW-PINN."
        )
    )
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[2000, 20000, 40000])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    parser.add_argument("--scheduler-step-size", type=int, default=2000)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--hidden-width", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-name", type=str, default="burgers_residual_evolution")
    parser.add_argument("--value-mode", choices=["raw", "log"], default="raw")
    parser.add_argument("--scale-mode", choices=["global", "column", "panel"], default="panel")
    parser.add_argument("--residual-eps", type=float, default=1e-12)
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        default=[0.0, 100.0],
        metavar=("LOW", "HIGH"),
        help="Color-scale clipping percentiles. The default 0 100 uses the full range.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["vanilla", "sa", "rba", "clorew"],
        default=None,
    )
    parser.add_argument(
        "--no-final-models",
        action="store_true",
        help="Train through the final checkpoint instead of loading existing final checkpoints.",
    )
    parser.add_argument("--rba-eta", type=float, default=1e-4)
    parser.add_argument("--rba-gamma", type=float, default=0.9999)
    parser.add_argument("--sa-weight-lr", type=float, default=0.005)
    parser.add_argument("--clorew-eta", type=float, default=1e-4)
    parser.add_argument("--clorew-gamma", type=float, default=0.9999)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--tau0", type=float, default=0.0)
    parser.add_argument("--eta-g", type=float, default=1e-5)
    parser.add_argument("--rho-g", type=float, default=0.999)
    parser.add_argument("--kernel-h", type=float, default=0.1)
    parser.add_argument("--kernel-p", type=float, default=1.0)
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


def make_problem(cfg: Config, script_dir: Path) -> Dict[str, np.ndarray]:
    data_path = script_dir.parent.parent / "burgers_shock.mat"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find Burgers data file: {data_path}")

    data = scipy.io.loadmat(data_path)
    exact = np.real(data["usol"])
    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    x_grid, t_grid = np.meshgrid(x.flatten(), t.flatten())
    x_star = np.hstack((x_grid.flatten()[:, None], t_grid.flatten()[:, None]))

    nx, nt = x.shape[0], t.shape[0]
    x_i = np.hstack((x_grid[0:1, :].T, t_grid[0:1, :].T))
    u_i = exact[:, 0:1]
    x_lb = np.hstack((x_grid[:, 0:1], t_grid[:, 0:1]))
    x_ub = np.hstack((x_grid[:, -1:], t_grid[:, -1:]))
    u_lb = exact[0:1, :].T
    u_ub = exact[-1:, :].T
    x_u = np.vstack([x_i, x_lb, x_ub])
    u = np.vstack([u_i, u_lb, u_ub])

    lb = x_star.min(0)
    ub = x_star.max(0)
    x_f = lb + (ub - lb) * latin_hypercube(2, cfg.n_residual, cfg.seed)
    x_f = np.vstack((x_f, x_u))

    return {
        "x": x.flatten(),
        "t": t.flatten(),
        "exact": exact,
        "x_star": x_star,
        "x_u": x_u,
        "u": u,
        "x_f": x_f,
        "lb": lb,
        "ub": ub,
        "grid_shape": (nx, nt),
    }


class DNN(torch.nn.Module):
    def __init__(self, layers: Tuple[int, ...]):
        super().__init__()
        depth = len(layers) - 1
        activation = torch.nn.Tanh()
        layer_list = []
        for i in range(depth - 1):
            layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(layer.weight)
            layer_list.append((f"layer_{i}", layer))
            layer_list.append((f"activation_{i}", activation))

        layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(layer.weight)
        layer_list.append((f"layer_{depth - 1}", layer))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BaseRunner:
    def __init__(self, cfg: Config, device: torch.device, problem: Dict[str, np.ndarray]):
        self.cfg = cfg
        self.device = device
        self.problem = problem
        self.iter = 0
        self.dnn = DNN(cfg.layers).to(device)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.scheduler_gamma)

        x_u = problem["x_u"]
        x_f = problem["x_f"]
        self.x_u = torch.tensor(x_u[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_u = torch.tensor(x_u[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u = torch.tensor(problem["u"], dtype=torch.float32, device=device)
        self.x_r = torch.tensor(x_f[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_r = torch.tensor(x_f[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)

    def net_u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.dnn(torch.cat([x, t], dim=1))

    def net_r(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        return u_t + u * u_x - self.cfg.nu * u_xx

    def loss_step(self) -> None:
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)
        loss = torch.mean(r_pred**2) + torch.mean((u_pred - self.u) ** 2)
        loss.backward()
        self.optimizer.step()

    def residual_grid(self, x_star: np.ndarray, grid_shape: Tuple[int, int], chunk_size: int) -> np.ndarray:
        self.dnn.eval()
        chunks: List[np.ndarray] = []
        for start in range(0, x_star.shape[0], chunk_size):
            pts = x_star[start : start + chunk_size]
            x = torch.tensor(pts[:, 0:1], dtype=torch.float32, device=self.device, requires_grad=True)
            t = torch.tensor(pts[:, 1:2], dtype=torch.float32, device=self.device, requires_grad=True)
            residual = self.net_r(x, t)
            chunks.append(tonp(torch.abs(residual)).reshape(-1))
        self.dnn.train()
        nt = grid_shape[1]
        nx = grid_shape[0]
        return np.concatenate(chunks, axis=0).reshape(nt, nx).T

    def train_and_collect_residuals(
        self,
        checkpoints: List[int],
        x_star: np.ndarray,
        grid_shape: Tuple[int, int],
        chunk_size: int,
    ) -> Dict[int, np.ndarray]:
        checkpoint_set = set(checkpoints)
        residuals: Dict[int, np.ndarray] = {}
        max_iter = max(checkpoints) if checkpoints else 0
        self.dnn.train()
        for _ in range(max_iter):
            self.loss_step()
            self.iter += 1
            if self.iter % self.cfg.scheduler_step_size == 0:
                self.scheduler.step()
            if self.iter in checkpoint_set:
                residuals[self.iter] = self.residual_grid(x_star, grid_shape, chunk_size)
                print(f"  collected residuals at iteration {self.iter}", flush=True)
        return residuals


class SAPINNRunner(BaseRunner):
    def __init__(self, cfg: Config, device: torch.device, problem: Dict[str, np.ndarray]):
        super().__init__(cfg, device, problem)
        self.lamr = torch.nn.Parameter(torch.rand(self.x_r.shape[0], 1, dtype=torch.float32, device=device))
        self.lamu = torch.nn.Parameter(torch.rand(self.u.shape[0], 1, dtype=torch.float32, device=device) * 100.0)
        self.weight_optimizer = torch.optim.Adam([self.lamr, self.lamu], lr=cfg.sa_weight_lr, maximize=True)

    def loss_step(self) -> None:
        self.optimizer.zero_grad()
        self.weight_optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)
        loss = torch.mean((self.lamr * r_pred) ** 2) + torch.mean((self.lamu * (u_pred - self.u)) ** 2)
        loss.backward()
        self.optimizer.step()
        self.weight_optimizer.step()


class RBAPINNRunner(BaseRunner):
    def __init__(self, cfg: Config, device: torch.device, problem: Dict[str, np.ndarray]):
        super().__init__(cfg, device, problem)
        self.rsum = 0.0

    def loss_step(self) -> None:
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)
        abs_r = torch.abs(r_pred)
        r_norm = self.cfg.rba_eta * abs_r / (torch.max(abs_r) + self.cfg.kernel_eps)
        self.rsum = (self.rsum * self.cfg.rba_gamma + r_norm).detach()
        loss = torch.mean((self.rsum * r_pred) ** 2) + torch.mean((u_pred - self.u) ** 2)
        loss.backward()
        self.optimizer.step()


class CLoReWPINNRunner(BaseRunner):
    def __init__(self, cfg: Config, device: torch.device, problem: Dict[str, np.ndarray]):
        super().__init__(cfg, device, problem)
        self.rsum = 0.0
        self.tau_gate = cfg.tau0
        self.gbar = torch.zeros_like(self.x_r).detach()
        self.gate_abs_res_ref = None
        t_min = float(problem["lb"][1])
        t_max = float(problem["ub"][1])
        self.t_centers = torch.linspace(t_min, t_max, cfg.num_kernel_centers, device=device).view(1, -1)
        self.kernel_weights = self.build_kernel_weights(self.t_r.detach())
        self.kernel_col_mass = torch.sum(self.kernel_weights, dim=0, keepdim=True).T + cfg.kernel_eps

    def build_kernel_weights(self, t_points: torch.Tensor) -> torch.Tensor:
        diff = t_points - self.t_centers
        weights = torch.exp(-0.5 * (diff / self.cfg.kernel_h) ** 2)
        if self.cfg.kernel_trunc is not None and self.cfg.kernel_trunc > 0:
            weights = weights * (torch.abs(diff) <= self.cfg.kernel_trunc * self.cfg.kernel_h).float()
        row_sum = weights.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if torch.any(zero_rows):
            nearest_idx = torch.argmin(torch.abs(diff[zero_rows]), dim=1)
            weights[zero_rows] = 0.0
            weights[zero_rows, nearest_idx] = 1.0
            row_sum = weights.sum(dim=1, keepdim=True)
        return weights / (row_sum + self.cfg.kernel_eps)

    def kernel_local_scale(self, abs_r_detach: torch.Tensor) -> torch.Tensor:
        center_moment_p = (self.kernel_weights.T @ (abs_r_detach ** self.cfg.kernel_p)) / self.kernel_col_mass
        local_moment_p = self.kernel_weights @ center_moment_p
        return (local_moment_p + self.cfg.kernel_eps) ** (1.0 / self.cfg.kernel_p)

    def loss_step(self) -> None:
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        r_pred = self.net_r(self.x_r, self.t_r)
        abs_r_detach = torch.abs(r_pred).detach()

        gate = 0.5 * (1.0 - torch.tanh(self.cfg.alpha * (self.t_r - self.tau_gate)))
        self.gbar = (self.cfg.rho_g * self.gbar + (1.0 - self.cfg.rho_g) * gate.detach()).detach()
        local_scale = self.kernel_local_scale(abs_r_detach)
        r_norm = self.cfg.clorew_eta * abs_r_detach / (local_scale + self.cfg.kernel_eps)
        causal_r_norm = r_norm * self.gbar
        self.rsum = (self.rsum * self.cfg.clorew_gamma + causal_r_norm).detach()

        loss = torch.mean((self.rsum * r_pred) ** 2) + torch.mean((u_pred - self.u) ** 2)
        loss.backward()
        self.optimizer.step()

        gate_abs_res = torch.sum(self.gbar * abs_r_detach) / (torch.sum(self.gbar) + self.cfg.kernel_eps)
        if self.gate_abs_res_ref is None:
            self.gate_abs_res_ref = gate_abs_res.item() + self.cfg.kernel_eps
        gate_abs_res_norm = gate_abs_res.item() / self.gate_abs_res_ref
        self.tau_gate = self.tau_gate + self.cfg.eta_g * np.exp(-self.cfg.epsilon * gate_abs_res_norm)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        seed=args.seed,
        nu=0.01 / np.pi,
        n_residual=args.n_residual,
        learning_rate=args.learning_rate,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_step_size=args.scheduler_step_size,
        layers=tuple([2] + args.hidden_layers * [args.hidden_width] + [1]),
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


def load_final_residual(
    cfg: Config,
    device: torch.device,
    problem: Dict[str, np.ndarray],
    model_path: Path,
    chunk_size: int,
) -> np.ndarray:
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find final model checkpoint: {model_path}")
    runner = BaseRunner(cfg, device, problem)
    runner.dnn.load_state_dict(torch.load(model_path, map_location=device))
    return runner.residual_grid(problem["x_star"], problem["grid_shape"], chunk_size)


def compute_limits(
    value_maps: Dict[str, Dict[int, np.ndarray]],
    method_names: List[str],
    checkpoints: List[int],
    scale_mode: str,
    clip_percentiles: List[float],
) -> Dict[Tuple[str, int], Tuple[float, float]]:
    low, high = clip_percentiles
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("--clip-percentiles must satisfy 0 <= LOW < HIGH <= 100.")

    def limits(arrays: List[np.ndarray]) -> Tuple[float, float]:
        stacked = np.concatenate([array.reshape(-1) for array in arrays], axis=0)
        return tuple(float(x) for x in np.percentile(stacked, [low, high]))

    panel_limits: Dict[Tuple[str, int], Tuple[float, float]] = {}
    if scale_mode == "global":
        vmin, vmax = limits([value_maps[name][step] for name in method_names for step in checkpoints])
        for name in method_names:
            for step in checkpoints:
                panel_limits[(name, step)] = (vmin, vmax)
    elif scale_mode == "column":
        for step in checkpoints:
            vmin, vmax = limits([value_maps[name][step] for name in method_names])
            for name in method_names:
                panel_limits[(name, step)] = (vmin, vmax)
    else:
        for name in method_names:
            for step in checkpoints:
                panel_limits[(name, step)] = limits([value_maps[name][step]])
    return panel_limits


def plot_residual_evolution(
    all_residuals: Dict[str, Dict[int, np.ndarray]],
    checkpoints: List[int],
    problem: Dict[str, np.ndarray],
    args: argparse.Namespace,
    figure_dir: Path,
    data_dir: Path,
) -> None:
    def display_map(residual_map: np.ndarray) -> np.ndarray:
        if args.value_mode == "log":
            return np.log10(residual_map + args.residual_eps)
        return residual_map

    method_names = list(all_residuals.keys())
    value_maps = {
        name: {step: display_map(all_residuals[name][step]) for step in checkpoints}
        for name in method_names
    }
    panel_limits = compute_limits(
        value_maps=value_maps,
        method_names=method_names,
        checkpoints=checkpoints,
        scale_mode=args.scale_mode,
        clip_percentiles=args.clip_percentiles,
    )

    n_rows = len(method_names)
    n_cols = len(checkpoints)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5.0 * n_cols, 9.0), max(3.35 * n_rows, 6.0)),
        squeeze=False,
    )
    extent = [
        float(problem["t"].min()),
        float(problem["t"].max()),
        float(problem["x"].min()),
        float(problem["x"].max()),
    ]

    for row, method_name in enumerate(method_names):
        for col, step in enumerate(checkpoints):
            ax = axes[row, col]
            vmin, vmax = panel_limits[(method_name, step)]
            im = ax.imshow(
                value_maps[method_name][step],
                cmap="rainbow",
                aspect="auto",
                extent=extent,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(f"Iteration {step}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{method_name}\nx", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("x", fontsize=10)
            ax.set_xlabel("t", fontsize=10)
            ax.tick_params(labelsize=9)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.06, top=0.94, wspace=0.28, hspace=0.32)

    png_path = figure_dir / f"{args.output_name}.png"
    pdf_path = figure_dir / f"{args.output_name}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    npz_path = data_dir / f"{args.output_name}.npz"
    np.savez_compressed(
        npz_path,
        checkpoints=np.array(checkpoints),
        method_names=np.array(method_names),
        x=problem["x"],
        t=problem["t"],
        exact=problem["exact"],
        residual_maps=np.array(
            [[all_residuals[name][step] for step in checkpoints] for name in method_names],
            dtype=np.float64,
        ),
        value_mode=np.array(args.value_mode),
        scale_mode=np.array(args.scale_mode),
        clip_percentiles=np.array(args.clip_percentiles, dtype=np.float64),
        panel_limits=np.array(
            [[panel_limits[(name, step)] for step in checkpoints] for name in method_names],
            dtype=np.float64,
        ),
    )
    print(f"Saved figure: {png_path.resolve()}")
    print(f"Saved figure: {pdf_path.resolve()}")
    print(f"Saved residual data: {npz_path.resolve()}")


def main() -> None:
    args = parse_args()
    checkpoints = sorted(set(int(step) for step in args.checkpoints))
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("Checkpoints must be positive integers.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    seed_torch(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    script_dir = Path(__file__).resolve().parent
    repro_dir = script_dir.parent
    figure_dir = script_dir / "figures"
    data_dir = script_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    problem = make_problem(cfg, script_dir)

    all_specs = [
        ("vanilla", "Vanilla PINN", BaseRunner, "burgers_pinn_model_{seed}.pt"),
        ("sa", "SA-PINN", SAPINNRunner, "burgers_sa_model_{seed}.pt"),
        ("rba", "RBA-PINN", RBAPINNRunner, "burgers_rba_model_{seed}.pt"),
        ("clorew", "C-LoReW-PINN", CLoReWPINNRunner, "burgers_clorew_pinn_model_{seed}.pt"),
    ]
    requested = args.methods if args.methods is not None else ["vanilla", "sa", "rba", "clorew"]
    specs = [spec for spec in all_specs if spec[0] in requested]

    final_step = 40000
    use_final_models = (not args.no_final_models) and final_step in checkpoints
    train_checkpoints = [step for step in checkpoints if not (use_final_models and step == final_step)]
    checkpoint_key = "-".join(str(step) for step in checkpoints)
    all_residuals: Dict[str, Dict[int, np.ndarray]] = {}

    for method_key, method_name, runner_cls, model_pattern in specs:
        cache_path = data_dir / f"{args.output_name}_{method_key}_seed{args.seed}_steps{checkpoint_key}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            cached_steps = [int(step) for step in cached["checkpoints"]]
            if cached_steps == checkpoints:
                print(f"Loading cached residual fields for {method_name}: {cache_path}", flush=True)
                all_residuals[method_name] = {
                    int(step): cached["residual_maps"][idx] for idx, step in enumerate(cached_steps)
                }
                continue

        residuals: Dict[int, np.ndarray] = {}
        if train_checkpoints:
            print(f"Training {method_name} to iteration {max(train_checkpoints)}...", flush=True)
            runner = runner_cls(cfg, device, problem)
            residuals.update(
                runner.train_and_collect_residuals(
                    checkpoints=train_checkpoints,
                    x_star=problem["x_star"],
                    grid_shape=problem["grid_shape"],
                    chunk_size=args.chunk_size,
                )
            )

        if use_final_models:
            model_path = repro_dir / "models" / model_pattern.format(seed=args.seed)
            print(f"Loading final model for {method_name}: {model_path}", flush=True)
            residuals[final_step] = load_final_residual(cfg, device, problem, model_path, args.chunk_size)

        missing = [step for step in checkpoints if step not in residuals]
        if missing:
            raise RuntimeError(f"Missing residual fields for {method_name}: {missing}")

        all_residuals[method_name] = {step: residuals[step] for step in checkpoints}
        np.savez_compressed(
            cache_path,
            method_name=np.array(method_name),
            checkpoints=np.array(checkpoints),
            residual_maps=np.array([all_residuals[method_name][step] for step in checkpoints], dtype=np.float64),
        )
        print(f"Saved method cache: {cache_path.resolve()}", flush=True)

    plot_residual_evolution(
        all_residuals=all_residuals,
        checkpoints=checkpoints,
        problem=problem,
        args=args,
        figure_dir=figure_dir,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()
