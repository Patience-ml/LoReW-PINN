"""Generate publication-ready per-PDE convergence figures for the LoReW-PINN paper.

The script reads saved loss/error histories from the reproducible experiment
folders and writes one 1x2 convergence figure per PDE:

    output/figures/poisson_convergence.{pdf,png}
    output/figures/helmholtz_convergence.{pdf,png}
    output/figures/reaction_diffusion_convergence.{pdf,png}
    output/figures/klein_gordon_convergence.{pdf,png}
    output/figures/burgers_convergence.{pdf,png}
    output/figures/allen_cahn_convergence.{pdf,png}

In each figure, the left panel shows the relative L2 error and the right panel
shows the L-infinity error. Each curve is the arithmetic mean over the
configured seeds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
SUMMARY_CSV = OUTPUT_DIR / "lorew_paper_figure_summary.csv"

Y_FLOOR = 1e-10
MAX_PLOT_POINTS = 220

METRIC_COLUMNS = {
    "l2": 1,
    "linf": 2,
}

METRIC_LABELS = {
    "l2": r"Relative $L^2$ error",
    "linf": r"$L^{\infty}$ error",
}

PANEL_LABELS = {
    "l2": r"(a) Relative $L^2$ error",
    "linf": r"(b) $L^{\infty}$ error",
}

METHOD_STYLES = {
    "Vanilla PINN": {"color": "#7A7A7A", "marker": "o"},
    "SA-PINN": {"color": "#74C69D", "marker": "^"},
    "RBA-PINN": {"color": "#64B5F6", "marker": "D"},
    "LoReW-PINN": {"color": "#F28482", "marker": "s"},
    "C-LoReW-PINN": {"color": "#F28482", "marker": "s"},
}

OUTPUT_STEMS = {
    "Poisson": "poisson_convergence",
    "Helmholtz": "helmholtz_convergence",
    "Reaction-Diffusion": "reaction_diffusion_convergence",
    "Klein-Gordon": "klein_gordon_convergence",
    "Burgers": "burgers_convergence",
    "Allen-Cahn": "allen_cahn_convergence",
}


@dataclass(frozen=True)
class MethodConfig:
    label: str
    pattern: str


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    base_dir: Path
    seeds: tuple[int, ...]
    methods: tuple[MethodConfig, ...]


STEADY_METHODS = (
    MethodConfig("Vanilla PINN", "losses_{problem}_pinn_{seed}.txt"),
    MethodConfig("SA-PINN", "losses_{problem}_sa_{seed}.txt"),
    MethodConfig("RBA-PINN", "losses_{problem}_rba_{seed}.txt"),
    MethodConfig("LoReW-PINN", "losses_{problem}_lorew_pinn_{seed}.txt"),
)

TIME_METHODS = (
    MethodConfig("Vanilla PINN", "losses_{problem}_pinn_{seed}.txt"),
    MethodConfig("SA-PINN", "losses_{problem}_sa_{seed}.txt"),
    MethodConfig("RBA-PINN", "losses_{problem}_rba_{seed}.txt"),
    MethodConfig("C-LoReW-PINN", "losses_{problem}_clorew_pinn_{seed}.txt"),
)

PROBLEMS = {
    "Poisson": ProblemConfig(
        name="Poisson",
        base_dir=PROJECT_ROOT / "Poisson" / "Reproducible",
        seeds=(1018, 1314, 2026, 2138, 3412, 3624, 4138, 4396, 5035, 5752),
        methods=STEADY_METHODS,
    ),
    "Helmholtz": ProblemConfig(
        name="Helmholtz",
        base_dir=PROJECT_ROOT / "Helmholtz" / "Reproducible",
        seeds=(1234, 1648, 2138, 2341, 3053, 3624, 4083, 4399, 5035, 5472),
        methods=STEADY_METHODS,
    ),
    "Reaction-Diffusion": ProblemConfig(
        name="Reaction-Diffusion",
        base_dir=PROJECT_ROOT / "RD" / "Reproducible",
        seeds=(1018, 1428, 2026, 2138, 3053, 3521, 4083, 4396, 5035, 5472),
        methods=TIME_METHODS,
    ),
    "Klein-Gordon": ProblemConfig(
        name="Klein-Gordon",
        base_dir=PROJECT_ROOT / "KG" / "Reproducible",
        seeds=(1018, 1234, 2048, 2341, 3624, 3899, 4123, 4396, 5231, 5472),
        methods=TIME_METHODS,
    ),
    "Burgers": ProblemConfig(
        name="Burgers",
        base_dir=PROJECT_ROOT / "Burgers" / "Reproducible",
        seeds=(1018, 1234, 2026, 2566, 3412, 3872, 4123, 4256, 5483, 5555),
        methods=TIME_METHODS,
    ),
    "Allen-Cahn": ProblemConfig(
        name="Allen-Cahn",
        base_dir=PROJECT_ROOT / "AC" / "Reproducible",
        seeds=(1018, 1234, 2428, 2832, 3542, 3625, 4235, 4864, 5483, 5656),
        methods=(
            MethodConfig("Vanilla PINN", "losses_pinn_fourier_{seed}.txt"),
            MethodConfig("SA-PINN", "losses_sa_fourier_{seed}.txt"),
            MethodConfig("RBA-PINN", "losses_rba_fourier_{seed}.txt"),
            MethodConfig("C-LoReW-PINN", "losses_clorew_pinn_fourier_{seed}.txt"),
        ),
    ),
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
            "figure.dpi": 150,
        }
    )


def problem_token(problem: ProblemConfig) -> str:
    return {
        "Poisson": "poisson",
        "Helmholtz": "helmholtz",
        "Reaction-Diffusion": "rd",
        "Klein-Gordon": "kg",
        "Burgers": "burgers",
        "Allen-Cahn": "ac",
    }[problem.name]


def method_file(problem: ProblemConfig, method: MethodConfig, seed: int) -> Path:
    file_name = method.pattern.format(problem=problem_token(problem), seed=seed)
    return problem.base_dir / "logs" / file_name


def load_method_runs(
    problem: ProblemConfig,
    method: MethodConfig,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[int], list[Path]]:
    value_col = METRIC_COLUMNS[metric]
    runs = []
    used_seeds = []
    used_paths = []

    for seed in problem.seeds:
        path = method_file(problem, method, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing required log file: {path}")

        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] <= value_col:
            raise ValueError(f"Unexpected log format: {path}")

        finite = np.isfinite(data[:, 0]) & np.isfinite(data[:, value_col])
        data = data[finite][:, [0, value_col]]
        if data.size == 0:
            raise ValueError(f"No finite data points found in: {path}")

        runs.append(data)
        used_seeds.append(seed)
        used_paths.append(path)

    common_iters = runs[0][:, 0]
    for run in runs[1:]:
        common_iters = np.intersect1d(common_iters, run[:, 0])

    if common_iters.size == 0:
        raise ValueError(f"No common iterations for {problem.name} / {method.label}")

    aligned = []
    for run in runs:
        iter_to_value = {row[0]: row[1] for row in run}
        aligned.append([iter_to_value[it] for it in common_iters])

    return common_iters, np.asarray(aligned, dtype=np.float64), used_seeds, used_paths


def summarize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_runs = values.shape[0]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)
    sem = std / np.sqrt(n_runs)
    lower = np.clip(mean - sem, Y_FLOOR, None)
    upper = np.clip(mean + sem, Y_FLOOR, None)
    return mean, std, sem, lower, upper


def downsample(
    iterations: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if iterations.size <= MAX_PLOT_POINTS:
        return iterations, mean, lower, upper

    indices = np.linspace(0, iterations.size - 1, MAX_PLOT_POINTS, dtype=int)
    indices = np.unique(indices)
    return iterations[indices], mean[indices], lower[indices], upper[indices]


def x_formatter(value: float, _position: int) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 1000:
        return f"{int(round(value / 1000))}k"
    return f"{int(value)}"


def estimate_x_step(max_iter: float) -> int:
    if max_iter >= 250000:
        return 100000
    if max_iter >= 100000:
        return 50000
    if max_iter >= 50000:
        return 20000
    if max_iter >= 30000:
        return 10000
    return 5000


def style_axis(ax: plt.Axes, metric: str, max_iter: float) -> None:
    ax.set_yscale("log")
    ax.set_xlim(0, max_iter * 1.02)
    ax.set_xlabel("Iterations", fontsize=9.5)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=9.5)
    ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(estimate_x_step(max_iter)))
    ax.grid(True, which="major", color="#D8D8D8", linestyle="--", linewidth=0.55, alpha=0.72)
    ax.grid(True, which="minor", axis="y", color="#ECECEC", linestyle=":", linewidth=0.45, alpha=0.55)
    ax.tick_params(axis="both", labelsize=8.2, colors="#333333", width=0.75, length=3.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def add_panel_label(ax: plt.Axes, metric: str) -> None:
    ax.text(
        0.5,
        -0.34,
        PANEL_LABELS[metric],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        clip_on=False,
    )


def plot_metric_axis(
    ax: plt.Axes,
    problem: ProblemConfig,
    metric: str,
    summary_rows: list[dict[str, str]],
    figure_name: str,
) -> tuple[list[plt.Line2D], list[str]]:
    max_iter = 0.0
    y_min = np.inf
    y_max = 0.0
    handles = []
    labels = []

    for method in problem.methods:
        iterations, values, used_seeds, used_paths = load_method_runs(problem, method, metric)
        mean, std, sem, lower, upper = summarize_values(values)
        max_iter = max(max_iter, float(iterations[-1]))
        y_min = min(y_min, float(np.nanmin(mean)))
        y_max = max(y_max, float(np.nanmax(mean)))

        plot_iters, plot_mean, plot_lower, plot_upper = downsample(iterations, mean, lower, upper)
        style = METHOD_STYLES[method.label]
        markevery = max(1, plot_iters.size // 10)
        (line,) = ax.plot(
            plot_iters,
            plot_mean,
            color=style["color"],
            linewidth=1.55,
            marker=style["marker"],
            markersize=3.2,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markevery=markevery,
            label=method.label,
        )
        handles.append(line)
        labels.append(method.label)
        summary_rows.append(
            {
                "figure": figure_name,
                "problem": problem.name,
                "metric": metric,
                "method": method.label,
                "seeds": " ".join(str(seed) for seed in used_seeds),
                "n_runs": str(values.shape[0]),
                "final_iteration": f"{iterations[-1]:.0f}",
                "final_mean": f"{mean[-1]:.10e}",
                "final_std": f"{std[-1]:.10e}",
                "final_sem": f"{sem[-1]:.10e}",
                "final_mean_minus_sem": f"{lower[-1]:.10e}",
                "final_mean_plus_sem": f"{upper[-1]:.10e}",
                "source_dir": str(problem.base_dir.relative_to(PROJECT_ROOT)),
                "source_files": " ".join(path.name for path in used_paths),
            }
        )

    style_axis(ax, metric=metric, max_iter=max_iter)
    ax.set_ylim(max(y_min / 1.25, Y_FLOOR), y_max * 1.22)
    add_panel_label(ax, metric)
    return handles, labels


def add_legend(fig: plt.Figure, handles: list[plt.Line2D], labels: list[str]) -> None:
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(labels),
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        fontsize=8.7,
        handlelength=2.2,
        columnspacing=1.25,
        borderpad=0.38,
        labelspacing=0.35,
    )
    legend.get_frame().set_edgecolor("#D0D0D0")
    legend.get_frame().set_linewidth(0.65)
    legend.get_frame().set_facecolor("#FFFFFF")


def save_figure(fig: plt.Figure, output_stem: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT_DIR / f"{output_stem}.pdf", OUTPUT_DIR / f"{output_stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths


def plot_problem_convergence(problem: ProblemConfig, summary_rows: list[dict[str, str]]) -> list[Path]:
    output_stem = OUTPUT_STEMS[problem.name]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95), constrained_layout=False)
    fig.patch.set_facecolor("#FFFFFF")

    all_handles: list[plt.Line2D] = []
    all_labels: list[str] = []
    for ax, metric in zip(axes, ("l2", "linf")):
        ax.set_facecolor("#FFFFFF")
        handles, labels = plot_metric_axis(
            ax=ax,
            problem=problem,
            metric=metric,
            summary_rows=summary_rows,
            figure_name=output_stem,
        )
        if not all_handles:
            all_handles = handles
            all_labels = labels

    add_legend(fig, all_handles, all_labels)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.78, bottom=0.30, wspace=0.30)
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def write_summary(rows: Iterable[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "figure",
        "problem",
        "metric",
        "method",
        "seeds",
        "n_runs",
        "final_iteration",
        "final_mean",
        "final_std",
        "final_sem",
        "final_mean_minus_sem",
        "final_mean_plus_sem",
        "source_dir",
        "source_files",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        choices=("all", *PROBLEMS.keys()),
        default="all",
        help="Problem to plot. Use quotes for names containing spaces.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    selected_problems = PROBLEMS.values() if args.problem == "all" else (PROBLEMS[args.problem],)

    summary_rows: list[dict[str, str]] = []
    saved_paths: list[Path] = []
    for problem in selected_problems:
        saved_paths.extend(plot_problem_convergence(problem, summary_rows))

    write_summary(summary_rows)

    for path in saved_paths:
        print(f"Saved: {path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
