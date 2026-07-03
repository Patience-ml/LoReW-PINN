import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.max_open_warning'] = 4


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'output' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USE_BEST_IF_AVAILABLE = True

DATASETS = {
    'Poisson': {
        'logs_dir': ROOT / 'Poisson' / 'Reproducible' / 'LoReW_hp_sensitivity' / 'logs',
        'h_pattern': 'poisson_lorew_h_sensitivity_summary_*.csv',
        'p_pattern': 'poisson_lorew_p_sensitivity_summary_*.csv',
        'color': '#64B5F6',
        'marker': 'o',
    },
    'Helmholtz': {
        'logs_dir': ROOT / 'Helmholtz' / 'Reproducible' / 'LoReW_hp_sensitivity' / 'logs',
        'h_pattern': 'helmholtz_lorew_h_sensitivity_summary_*.csv',
        'p_pattern': 'helmholtz_lorew_p_sensitivity_summary_*.csv',
        'color': '#F28482',
        'marker': 's',
    },
}


def format_milli_tick(value, _position):
    if value <= 0:
        return ''
    scaled = value * 1e3
    if abs(scaled - round(scaled)) < 1e-8:
        return str(int(round(scaled)))
    return f'{scaled:.1f}'.rstrip('0').rstrip('.')


def latest_file(directory, pattern):
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'No summary file found: {directory / pattern}')
    return files[-1]


def read_summary(path, x_key):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            metric_key = 'best_l2_rel' if USE_BEST_IF_AVAILABLE and row.get('best_l2_rel') else 'final_l2_rel'
            rows.append({
                'x': float(row[x_key]),
                'y': float(row[metric_key]),
                'metric_key': metric_key,
                'seed': row.get('seed', ''),
            })

    if not rows:
        raise ValueError(f'No data rows found in {path}')

    rows = sorted(rows, key=lambda item: item['x'])
    metric_keys = sorted(set(item['metric_key'] for item in rows))
    metric_label = 'best' if metric_keys == ['best_l2_rel'] else 'final'
    if len(metric_keys) > 1:
        metric_label = 'mixed'
    return rows, metric_label


def plot_one_axis(ax, sweep_key, xlabel):
    for dataset_name, config in DATASETS.items():
        pattern = config[f'{sweep_key}_pattern']
        summary_path = latest_file(config['logs_dir'], pattern)
        rows, metric_label = read_summary(summary_path, f'kernel_{sweep_key}')
        x = np.array([item['x'] for item in rows])
        y = np.array([item['y'] for item in rows])
        label = dataset_name
        ax.plot(
            x,
            y,
            marker=config['marker'],
            markersize=6,
            linewidth=2.0,
            color=config['color'],
            label=label,
        )

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(r'Relative $L^2$ error', fontsize=14)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_milli_tick))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.offsetText.set_visible(False)
    ax.text(
        0.0,
        1.015,
        r'$\times 10^{-3}$',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=12,
    )
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.45)
    ax.tick_params(axis='both', labelsize=12, direction='in', width=0.9, length=4)
    ax.legend(fontsize=10, frameon=True, loc='best')


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    plot_one_axis(axes[0], 'h', r'Kernel bandwidth $h$')
    axes[0].set_title(r'Sensitivity to $h$', fontsize=15, pad=8)
    axes[0].text(
        0.5,
        -0.24,
        r'(a) Sensitivity to $h$',
        transform=axes[0].transAxes,
        ha='center',
        va='top',
        fontsize=14,
    )

    plot_one_axis(axes[1], 'p', r'Kernel exponent $p$')
    axes[1].set_title(r'Sensitivity to $p$', fontsize=15, pad=8)
    axes[1].text(
        0.5,
        -0.24,
        r'(b) Sensitivity to $p$',
        transform=axes[1].transAxes,
        ha='center',
        va='top',
        fontsize=14,
    )

    fig.tight_layout(w_pad=2.8)

    png_path = OUTPUT_DIR / 'lorew_hp_sensitivity_poisson_helmholtz.png'
    pdf_path = OUTPUT_DIR / 'lorew_hp_sensitivity_poisson_helmholtz.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved figure to: {png_path}')
    print(f'Saved figure to: {pdf_path}')


if __name__ == '__main__':
    main()
