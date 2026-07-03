import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.max_open_warning'] = 4


SEEDS = [1018, 1234, 2428, 2832, 3542, 3625, 4235, 4864, 5483, 5855]
NUM_DISPLAY_POINTS = 18

METHODS = {
    'PINN-Fourier': {
        'pattern': 'losses_pinn_fourier_{seed}.txt',
        'color': '#7F7F7F',
        'marker': 'o',
    },
    'PINN-Enhanced': {
        'pattern': 'losses_pinn_enhanced_{seed}.txt',
        'color': '#4D4D4D',
        'marker': 'o',
    },
    'SA-Fourier': {
        'pattern': 'losses_sa_fourier_{seed}.txt',
        'color': '#74C69D',
        'marker': '^',
    },
    'SA-Enhanced': {
        'pattern': 'losses_sa_enhanced_{seed}.txt',
        'color': '#2D8F68',
        'marker': '^',
    },
    'RBA-Fourier': {
        'pattern': 'losses_rba_fourier_{seed}.txt',
        'color': '#64B5F6',
        'marker': 'D',
    },
    'RBA-Enhanced': {
        'pattern': 'losses_rba_enhanced_{seed}.txt',
        'color': '#2E86C1',
        'marker': 'D',
    },
    'C-LoReW-Fourier': {
        'pattern': 'losses_clorew_pinn_fourier_{seed}.txt',
        'color': '#F28482',
        'marker': 's',
    },
    'C-LoReW-Enhanced': {
        'pattern': 'losses_clorew_pinn_enhanced_{seed}.txt',
        'color': '#C94F4D',
        'marker': 's',
    },
    'LoReW-Fourier': {
        'pattern': 'losses_lorew_pinn_fourier_{seed}.txt',
        'color': '#F28482',
        'marker': 'P',
    },
    'LoReW-Enhanced': {
        'pattern': 'losses_lorew_pinn_enhanced_{seed}.txt',
        'color': '#C94F4D',
        'marker': 'P',
    },
}


def thousands_formatter(x, pos):
    if x >= 1000:
        return f'{int(x):,}'
    return f'{int(x)}'


def estimate_x_step(max_iter):
    if max_iter >= 250000:
        return 50000
    if max_iter >= 100000:
        return 20000
    if max_iter >= 50000:
        return 10000
    return 5000


def load_method_runs(base_dir, pattern, seeds, value_col):
    runs = []
    for seed in seeds:
        path = os.path.join(base_dir, 'logs', pattern.format(seed=seed))
        if not os.path.exists(path):
            continue

        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] <= value_col:
            raise ValueError(f'Unexpected file format: {path}')
        runs.append(data[:, [0, value_col]])

    if not runs:
        return None

    common_iters = runs[0][:, 0]
    for run in runs[1:]:
        common_iters = np.intersect1d(common_iters, run[:, 0])

    if common_iters.size == 0:
        raise ValueError('No common iteration points found across runs.')

    aligned_values = []
    for run in runs:
        iter_to_value = {row[0]: row[1] for row in run}
        aligned_values.append([iter_to_value[it] for it in common_iters])

    aligned_values = np.asarray(aligned_values, dtype=np.float64)
    log_values = np.log10(np.clip(aligned_values, 1e-16, None))
    mean_log_values = log_values.mean(axis=0)
    std_log_values = log_values.std(axis=0, ddof=0)
    return common_iters, mean_log_values, std_log_values


def sample_display_points(iterations, mean_values, lower, upper, num_points):
    if len(iterations) <= num_points:
        return iterations, mean_values, lower, upper

    sample_idx = np.linspace(0, len(iterations) - 1, num_points, dtype=int)
    sample_idx = np.unique(sample_idx)
    return (
        iterations[sample_idx],
        mean_values[sample_idx],
        lower[sample_idx],
        upper[sample_idx],
    )


def style_axis(ax, max_iter, y_min, y_max, ylabel, title):
    x_max_plot = max_iter * 1.04
    ax.set_yscale('log')
    ax.set_xlim(0, x_max_plot)
    ax.set_ylim(max(y_min / 1.35, 1e-6), y_max * 1.18)
    ax.set_xlabel('Iterations', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(estimate_x_step(max_iter)))

    ax.grid(True, which='major', color='#D4D4D4', linestyle='--', linewidth=0.75, alpha=0.65)
    ax.grid(True, which='minor', axis='y', color='#ECECEC', linestyle=':', linewidth=0.5, alpha=0.35)
    ax.tick_params(axis='both', labelsize=11, colors='#4F4F4F', width=0.9, length=5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')


def plot_metric(ax, base_dir, value_col, ylabel, title):
    max_iter = 0
    global_min = np.inf
    global_max = 0.0

    for label, cfg in METHODS.items():
        loaded = load_method_runs(
            base_dir,
            cfg['pattern'],
            SEEDS,
            value_col=value_col,
        )
        if loaded is None:
            continue

        iterations, mean_log_values, std_log_values = loaded
        max_iter = max(max_iter, int(iterations[-1]))
        mean_values = 10 ** mean_log_values
        lower = 10 ** (mean_log_values - std_log_values)
        upper = 10 ** (mean_log_values + std_log_values)
        plot_iters, plot_mean, plot_lower, plot_upper = sample_display_points(
            iterations,
            mean_values,
            lower,
            upper,
            NUM_DISPLAY_POINTS,
        )

        global_min = min(global_min, float(np.min(plot_lower)))
        global_max = max(global_max, float(np.max(plot_upper)))

        ax.plot(
            plot_iters,
            plot_mean,
            color=cfg['color'],
            linewidth=2.1,
            label=label,
            marker=cfg['marker'],
            markersize=6.5,
            markerfacecolor=cfg['color'],
            markeredgecolor='white',
            markeredgewidth=0.9,
        )
        ax.fill_between(
            plot_iters,
            plot_lower,
            plot_upper,
            color=cfg['color'],
            alpha=0.14,
            linewidth=0,
        )

    if max_iter == 0:
        raise FileNotFoundError(f'No loss files found in {base_dir}')

    style_axis(ax, max_iter, global_min, global_max, ylabel, title)


def plot_comparison(base_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.7), sharex=True)
    fig.patch.set_facecolor('#FFFFFF')
    for ax in axes:
        ax.set_facecolor('#FCFCFC')

    plot_metric(
        axes[0],
        base_dir,
        value_col=1,
        ylabel=r'Relative $L^2$ error',
        title='',
    )
    plot_metric(
        axes[1],
        base_dir,
        value_col=2,
        ylabel=r'$L^{\infty}$ error',
        title='',
    )

    axes[0].text(
        0.5,
        -0.20,
        '(a)',
        transform=axes[0].transAxes,
        ha='center',
        va='top',
        fontsize=12,
        color='#4F4F4F',
    )
    axes[1].text(
        0.5,
        -0.20,
        '(b)',
        transform=axes[1].transAxes,
        ha='center',
        va='top',
        fontsize=12,
        color='#4F4F4F',
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        fontsize=10.5,
        ncol=min(5, len(labels)),
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=2.2,
    )
    legend.get_frame().set_edgecolor('#D7D7D7')
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_facecolor('#FFFFFF')

    fig.suptitle('AC Equation', fontsize=15, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84, bottom=0.24, wspace=0.22)

    figure_dir = os.path.join(base_dir, 'figures')

    os.makedirs(figure_dir, exist_ok=True)

    output_path = os.path.join(figure_dir, 'AC_Method_Comparison_Combined_L2_Linf.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return output_path


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    saved_path = plot_comparison(current_dir)
    print(f'Saved figure to: {saved_path}')
