from pathlib import Path
import re

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

PDE_CONFIGS = [
    {
        'label': 'Reaction-diffusion',
        'short_label': 'RD',
        'root': ROOT / 'RD' / 'Reproducible' / 'Ablation' / 'logs',
        'seed': 2138,
        'logs': {
            'LoReW-PINN': 'RD_LoReW_PINN_train_log_2138.txt',
            'CR-PINN': 'RD_CR_PINN_train_log_2138.txt',
            'C-LoReW-PINN': 'RD_C_LoReW_PINN_train_log_2138.txt',
        },
    },
    {
        'label': 'Klein-Gordon',
        'short_label': 'KG',
        'root': ROOT / 'KG' / 'Reproducible' / 'Ablation' / 'logs',
        'seed': 5472,
        'logs': {
            'LoReW-PINN': 'KG_LoReW_PINN_train_log_5472.txt',
            'CR-PINN': 'KG_CR_PINN_train_log_5472.txt',
            'C-LoReW-PINN': 'KG_C_LoReW_PINN_train_log_5472.txt',
        },
    },
    {
        'label': 'Burgers',
        'short_label': 'Burgers',
        'root': ROOT / 'Burgers' / 'Reproducible' / 'Ablation' / 'logs',
        'seed': 2566,
        'logs': {
            'LoReW-PINN': 'Burgers_LoReW_PINN_train_log_2566.txt',
            'CR-PINN': 'Burgers_CR_PINN_train_log_2566.txt',
            'C-LoReW-PINN': 'Burgers_C_LoReW_PINN_train_log_2566.txt',
        },
    },
    {
        'label': 'Allen-Cahn',
        'short_label': 'AC',
        'root': ROOT / 'AC' / 'Reproducible' / 'Ablation' / 'logs',
        'seed': 1234,
        'logs': {
            'LoReW-PINN': 'AC_LoReW_fourier_train_log_1234.txt',
            'CR-PINN': 'AC_CR_fourier_train_log_1234.txt',
            'C-LoReW-PINN': 'AC_C_LoReW_fourier_train_log_1234.txt',
        },
    },
]

METHODS = ['LoReW-PINN', 'CR-PINN', 'C-LoReW-PINN']
METHOD_COLORS = {
    'LoReW-PINN': '#64B5F6',
    'CR-PINN': '#F4A261',
    'C-LoReW-PINN': '#F28482',
}
METHOD_HATCHES = {
    'LoReW-PINN': '',
    'CR-PINN': '',
    'C-LoReW-PINN': '',
}


def extract_final_l2(log_path):
    if not log_path.exists():
        raise FileNotFoundError(f'Missing ablation log: {log_path}')

    text = log_path.read_text(encoding='utf-8', errors='ignore')
    matches = re.findall(r'Final Relative L2 Error:\s*([0-9]+\.?[0-9]*e[+-][0-9]+)', text)
    if matches:
        return float(matches[-1])

    iter_matches = re.findall(r'Rel_L2:\s*([0-9]+\.?[0-9]*e[+-][0-9]+)', text)
    if iter_matches:
        return float(iter_matches[-1])

    raise ValueError(f'Cannot find relative L2 error in {log_path}')


def collect_results():
    rows = []
    for pde in PDE_CONFIGS:
        for method in METHODS:
            value = extract_final_l2(pde['root'] / pde['logs'][method])
            rows.append({
                'pde': pde['label'],
                'short_label': pde['short_label'],
                'seed': pde['seed'],
                'method': method,
                'relative_l2_error': value,
            })
    return rows


def save_summary(rows):
    summary_path = OUTPUT_DIR / 'clorew_ablation_time_dependent_summary.csv'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('pde,short_label,seed,method,relative_l2_error\n')
        for row in rows:
            f.write(
                f"{row['pde']},{row['short_label']},{row['seed']},"
                f"{row['method']},{row['relative_l2_error']:.10e}\n"
            )
    return summary_path


def sci_tick(value, _position):
    if value <= 0:
        return ''
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10 ** exponent)
    if abs(mantissa - 1.0) < 1e-8:
        return rf'$10^{{{exponent}}}$'
    if abs(mantissa - 2.0) < 1e-8:
        return rf'$2\times10^{{{exponent}}}$'
    if abs(mantissa - 5.0) < 1e-8:
        return rf'$5\times10^{{{exponent}}}$'
    return ''


def plot(rows):
    labels = [item['label'] for item in PDE_CONFIGS]
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.2, 4.7))

    for method_idx, method in enumerate(METHODS):
        offsets = x + (method_idx - 1) * width
        values = []
        for pde in PDE_CONFIGS:
            match = next(
                row for row in rows
                if row['short_label'] == pde['short_label'] and row['method'] == method
            )
            values.append(match['relative_l2_error'])

        ax.bar(
            offsets,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor='white',
            linewidth=0.8,
            hatch=METHOD_HATCHES[method],
            label=method,
            zorder=3,
        )

    ax.set_yscale('log')
    ax.set_ylabel(r'Relative $L^2$ error', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_xlabel('PDE benchmark', fontsize=15)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_tick))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='both', labelsize=12, direction='in', width=0.9, length=4)
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7, alpha=0.45, zorder=0)
    ax.legend(fontsize=11, frameon=True, loc='upper right')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_zorder(10)

    fig.tight_layout()

    png_path = OUTPUT_DIR / 'clorew_ablation_time_dependent.png'
    pdf_path = OUTPUT_DIR / 'clorew_ablation_time_dependent.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    return png_path, pdf_path


def main():
    rows = collect_results()
    summary_path = save_summary(rows)
    png_path, pdf_path = plot(rows)

    print(f'Saved summary to: {summary_path}')
    print(f'Saved figure to: {png_path}')
    print(f'Saved figure to: {pdf_path}')


if __name__ == '__main__':
    main()
