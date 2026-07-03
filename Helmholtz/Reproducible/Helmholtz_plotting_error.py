import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.max_open_warning'] = 4


SEEDS = [1234, 1648, 2341, 2138, 3053, 3624, 4083, 4399, 5035, 5472]
EPOCH_STEP = 100
NUM_PLOT_POINTS = 8
Y_FLOOR = 1e-8

METHODS = {
    'Vanilla PINN': {
        'pattern': 'losses_helmholtz_pinn_{seed}.txt',
        'color': '#7F7F7F',
    },
    'SA-PINN': {
        'pattern': 'losses_helmholtz_sa_{seed}.txt',
        'color': '#74C69D',
    },
    'RBA-PINN': {
        'pattern': 'losses_helmholtz_rba_{seed}.txt',
        'color': '#64B5F6',
    },
    'LoReW-PINN': {
        'pattern': 'losses_helmholtz_lorew_pinn_{seed}.txt',
        'color': '#F28482',
    },
}

METRICS = {
    'L2RE': {
        'column': 1,
        'output': 'Helmholtz_4x50_plotting_error_L2RE.png',
        'ylabel': r'Relative $L^2$ error',
    },
    'L_inf': {
        'column': 2,
        'output': 'Helmholtz_4x50_plotting_error_Linf.png',
        'ylabel': r'$L^{\infty}$ error',
    },
}


def thousands_formatter(x, pos):
    if x >= 1000:
        return f'{int(x):,}'
    return f'{int(x)}'


def load_runs(
    base_dir,
    pattern,
    seeds,
    value_col,
    epoch_step,
):
    runs = []
    used_seeds = []
    for seed in seeds:
        path = os.path.join(base_dir, 'logs', pattern.format(seed=seed))
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing required file: {path}')

        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] <= value_col:
            raise ValueError(f'Unexpected file format: {path}')

        data = data[:, [0, value_col]]
        data = data[np.mod(data[:, 0], epoch_step) == 0]
        if data.size == 0:
            raise ValueError(f'No iteration points match epoch_step={epoch_step}: {path}')

        runs.append(data)
        used_seeds.append(seed)

    common_iters = runs[0][:, 0]
    for run in runs[1:]:
        common_iters = np.intersect1d(common_iters, run[:, 0])

    if common_iters.size == 0:
        raise ValueError('No common iteration points found across runs.')

    aligned_values = []
    for run in runs:
        iter_to_value = {row[0]: row[1] for row in run}
        aligned_values.append([iter_to_value[it] for it in common_iters])

    values = np.asarray(aligned_values, dtype=np.float64)
    sample_count = values.shape[0]
    mean_values = values.mean(axis=0)
    std_values = values.std(axis=0, ddof=1) if sample_count > 1 else np.zeros_like(mean_values)
    se_values = std_values / np.sqrt(sample_count)
    lower = np.clip(mean_values - se_values, Y_FLOOR, None)
    upper = mean_values + se_values
    spread_values = se_values

    return common_iters, mean_values, spread_values, lower, upper, used_seeds


def select_evenly_spaced_points(iterations, mean_values, lower, upper, num_points):
    if num_points is None or num_points <= 0 or iterations.size <= num_points:
        return iterations, mean_values, lower, upper

    target_iters = np.linspace(iterations[0], iterations[-1], num_points)
    indices = [int(np.argmin(np.abs(iterations - target))) for target in target_iters]
    indices = np.asarray(sorted(set(indices)), dtype=int)

    return iterations[indices], mean_values[indices], lower[indices], upper[indices]


def style_axis(ax, ylabel, max_iter):
    ax.set_yscale('log')
    ax.set_xlim(0, max_iter * 1.02)
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(10000))

    ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.28)
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.45, alpha=0.18)
    ax.tick_params(axis='both', labelsize=13, width=1.2, length=5)

    for spine in ax.spines.values():
        spine.set_color('#111111')
        spine.set_linewidth(1.4)


def plot_single_metric(
    base_dir,
    metric_name,
    metric_cfg,
    seeds,
    epoch_step,
    show,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    # Use a logarithmic y-axis because the methods differ by orders of magnitude.
    ax.set_yscale('log')

    max_iter = 0
    y_min = np.inf
    y_max = 0.0

    for label, cfg in METHODS.items():
        iterations, mean_values, spread_values, lower, upper, used_seeds = load_runs(
            base_dir=base_dir,
            pattern=cfg['pattern'],
            seeds=seeds,
            value_col=metric_cfg['column'],
            epoch_step=epoch_step,
        )
        max_iter = max(max_iter, int(iterations[-1]))
        iterations, mean_values, lower, upper = select_evenly_spaced_points(
            iterations,
            mean_values,
            lower,
            upper,
            NUM_PLOT_POINTS,
        )
        y_min = min(y_min, float(np.min(mean_values)))
        y_max = max(y_max, float(np.max(mean_values)))

        ax.plot(
            iterations,
            mean_values,
            color=cfg['color'],
            linewidth=2.0,
            marker='o',
            markersize=5.8,
            markeredgewidth=0.0,
            label=label,
        )
        print(
            f'{metric_name} {label}: seeds={used_seeds}, '
            f'final center={mean_values[-1]:.6e}, final interval=[{lower[-1]:.6e}, {upper[-1]:.6e}], '
            f'final standard error={spread_values[-1]:.6e}'
        )

    style_axis(ax, metric_cfg.get('ylabel', metric_name), max_iter)
    ax.set_ylim(max(y_min / 1.25, Y_FLOOR), y_max * 1.25)
    ax.legend(
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        fontsize=12,
        edgecolor='#DDDDDD',
    )

    fig.tight_layout()
    figure_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, metric_cfg['output'])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plotting_error(
    base_dir=None,
    seeds=None,
    epoch_step=EPOCH_STEP,
    show=True,
):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    if seeds is None:
        seeds = SEEDS

    saved_paths = {}
    for metric_name, metric_cfg in METRICS.items():
        saved_paths[metric_name] = plot_single_metric(
            base_dir=base_dir,
            metric_name=metric_name,
            metric_cfg=metric_cfg,
            seeds=seeds,
            epoch_step=epoch_step,
            show=show,
        )
    return saved_paths


if __name__ == '__main__':
    paths = plotting_error(show=True)
    for metric, path in paths.items():
        print(f'Saved {metric} figure to: {path}')
