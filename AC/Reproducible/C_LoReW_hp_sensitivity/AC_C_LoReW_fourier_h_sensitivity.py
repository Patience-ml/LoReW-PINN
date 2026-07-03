import csv
import os
import sys
import types

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.max_open_warning'] = 4


file_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(file_path)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)


def install_pydoe_fallback():
    try:
        import pyDOE  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    def lhs(n, samples=None, random_state=None, **_kwargs):
        samples = n if samples is None else samples
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        cut = np.linspace(0.0, 1.0, samples + 1)
        u = rng.rand(samples, n)
        a = cut[:samples]
        b = cut[1:samples + 1]
        points = u * (b - a)[:, None] + a[:, None]
        result = np.zeros_like(points)
        for dim in range(n):
            result[:, dim] = points[rng.permutation(samples), dim]
        return result

    pydoe_module = types.ModuleType('pyDOE')
    pydoe_module.lhs = lhs
    sys.modules['pyDOE'] = pydoe_module


install_pydoe_fallback()
import AC_C_LoReW_fourier as base


LOG_DIR = os.path.join(file_path, 'logs')
MODEL_DIR = os.path.join(file_path, 'models')
FIGURE_DIR = os.path.join(file_path, 'figures')
for output_dir in (LOG_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)

base.LOG_DIR = LOG_DIR
base.MODEL_DIR = MODEL_DIR
base.FIGURE_DIR = FIGURE_DIR
base.layers = [2] + 6 * [128] + [1]

seed = 4235


def configure_base_seed(seed_value):
    base.seed = seed_value
    base.seed_torch(seed_value)
    if base.torch.cuda.is_available():
        base.torch.cuda.empty_cache()
    idx_x = np.random.choice(base.x0.shape[0], base.N_u, replace=False)
    base.x_u = base.x0[idx_x, :]
    base.t_u = np.zeros((base.N_u, 1))
    base.u_u = base.Exact0[idx_x, 0:1]
    base.X_u = np.hstack((base.x_u, base.t_u))
    base.u = base.u_u
    base.X_f = base.lb + (base.ub - base.lb) * base.lhs(2, base.N_f, random_state=seed_value)
    base.X_f = np.vstack((base.X_f, base.X_u))


configure_base_seed(seed)


def format_tag(value):
    return ('%.5g' % value).replace('-', 'm').replace('.', 'p')


# A common temporal C-LoReW sweep for the time-dependent PDEs.
# Current defaults are KG: h=0.02, AC/RD: h=0.05, Burgers: h=0.10.
H_VALUES = [0.02 + 0.02 * i for i in range(7)]
FIXED_KERNEL_P = 2.0
HP_CONFIGS = [
    {
        'tag': f'h_sweep_h{format_tag(h)}',
        'kernel_h': h,
        'kernel_p': FIXED_KERNEL_P,
    }
    for h in H_VALUES
]


def log_message(message, train_log_path):
    print(message)
    with open(train_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(str(message) + '\n')


def get_output_names(hp_config):
    tag = hp_config['tag']
    train_log_path = os.path.join(LOG_DIR, f'AC_C_LoReW_fourier_h_sensitivity_{tag}_train_log_{seed}.txt')
    loss_file_name = f'losses_ac_clorew_fourier_h_sensitivity_{tag}_{seed}.txt'
    model_name = f'ac_clorew_fourier_h_sensitivity_{tag}_model_{seed}'
    return tag, train_log_path, loss_file_name, model_name


def load_completed_result(hp_config):
    tag, train_log_path, loss_file_name, model_name = get_output_names(hp_config)
    loss_path = os.path.join(LOG_DIR, loss_file_name)
    model_path = os.path.join(MODEL_DIR, model_name + '.pt')
    if not (os.path.exists(train_log_path) and os.path.exists(loss_path) and os.path.exists(model_path)):
        return None

    final_l2_rel = None
    final_l_inf = None
    total_time = None
    with open(train_log_path, 'r', encoding='utf-8') as log_file:
        for line in log_file:
            line = line.strip()
            if line.startswith('Training finished. Total time:'):
                total_time = line.split(': ', 1)[1]
            elif line.startswith('Final Relative L2 Error:'):
                final_l2_rel = float(line.split(':', 1)[1])
            elif line.startswith('Final L_inf Error:'):
                final_l_inf = float(line.split(':', 1)[1])

    if final_l2_rel is None or final_l_inf is None or total_time is None:
        return None

    return {
        'tag': tag,
        'seed': seed,
        'kernel_h': hp_config['kernel_h'],
        'kernel_p': hp_config['kernel_p'],
        'final_l2_rel': final_l2_rel,
        'final_l_inf': final_l_inf,
        'total_time': total_time,
        'loss_file': loss_file_name,
        'model_file': model_name + '.pt',
        'train_log': os.path.basename(train_log_path),
    }


def apply_kernel_config(model, hp_config):
    model.kernel_h = hp_config['kernel_h']
    model.kernel_p = hp_config['kernel_p']
    model.kernel_weights = model.build_kernel_weights(model.t_r.detach())
    model.kernel_col_mass = base.torch.sum(model.kernel_weights, dim=0, keepdim=True).T + model.kernel_eps


def run_one_config(hp_config):
    if base.torch.cuda.is_available():
        base.torch.cuda.empty_cache()

    tag, train_log_path, loss_file_name, model_name = get_output_names(hp_config)

    base.train_log_path = train_log_path
    base.file_name = loss_file_name

    open(train_log_path, 'w', encoding='utf-8').close()
    log_message('=' * 70, train_log_path)
    log_message('Initializing C-LoReW-Fourier h-sensitivity run on the Allen-Cahn equation...', train_log_path)
    log_message(
        'Config: h = %.5f, p = %.5f, seed = %d'
        % (hp_config['kernel_h'], hp_config['kernel_p'], seed),
        train_log_path,
    )
    log_message('Only kernel_h is changed from the base AC C-LoReW-Fourier script; kernel_p is fixed at 2.0.', train_log_path)
    log_message('Starting training...', train_log_path)

    base.seed_torch(seed)
    model = base.PINN(base.X_u, base.u, base.X_f, base.lb, base.ub, base.N_u, base.N_u, savept=model_name)
    apply_kernel_config(model, hp_config)
    model.train()
    log_message('Training finished. Total time: %s' % base.format_elapsed_time(model.exec_time), train_log_path)

    model.dnn.eval()
    with base.torch.no_grad():
        u_pred = model.net_u(model.xx, model.tt)
    sol = base.tonp(u_pred).reshape(base.nt, base.nx).T

    exact_flat = model.Exact.flatten()
    pred_flat = sol.flatten()
    final_l2_rel = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
    final_l_inf = np.linalg.norm(exact_flat - pred_flat, np.inf) / np.linalg.norm(exact_flat, np.inf)

    log_message('-' * 50, train_log_path)
    log_message('Final test-set results', train_log_path)
    log_message(f'Final Relative L2 Error: {final_l2_rel:.5e}', train_log_path)
    log_message(f'Final L_inf Error:       {final_l_inf:.5e}', train_log_path)
    log_message('-' * 50, train_log_path)

    return {
        'tag': tag,
        'seed': seed,
        'kernel_h': hp_config['kernel_h'],
        'kernel_p': hp_config['kernel_p'],
        'final_l2_rel': final_l2_rel,
        'final_l_inf': final_l_inf,
        'total_time': base.format_elapsed_time(model.exec_time),
        'loss_file': loss_file_name,
        'model_file': model_name + '.pt',
        'train_log': os.path.basename(train_log_path),
    }


def save_summary(results):
    summary_path = os.path.join(LOG_DIR, f'ac_clorew_fourier_h_sensitivity_summary_{seed}.csv')
    fieldnames = [
        'tag',
        'seed',
        'kernel_h',
        'kernel_p',
        'final_l2_rel',
        'final_l_inf',
        'total_time',
        'loss_file',
        'model_file',
        'train_log',
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(results, key=lambda item: item['final_l2_rel']):
            writer.writerow(row)

    print('=' * 70)
    print(f'Saved summary to: {summary_path}')
    print('Final Relative L2 Error for each h value:')
    for row in sorted(results, key=lambda item: item['kernel_h']):
        print(
            'h=%.5f | p=%.5f, final_L2RE=%.5e, final_Linf=%.5e, time=%s'
            % (
                row['kernel_h'],
                row['kernel_p'],
                row['final_l2_rel'],
                row['final_l_inf'],
                row['total_time'],
            )
        )
    min_row = min(results, key=lambda item: item['final_l2_rel'])
    print(
        'Lowest final error: h=%.5f, p=%.5f, final_L2RE=%.5e'
        % (
            min_row['kernel_h'],
            min_row['kernel_p'],
            min_row['final_l2_rel'],
        )
    )
    print('=' * 70)
    return summary_path


def plot_l2_comparison(results):
    plt.figure(figsize=(9, 5.5))
    for row in results:
        loss_path = os.path.join(LOG_DIR, row['loss_file'])
        if not os.path.exists(loss_path):
            continue
        data = np.loadtxt(loss_path)
        if data.ndim != 2 or data.shape[0] == 0:
            continue
        plt.plot(data[:, 0], data[:, 1], linewidth=1.4, label=row['tag'])

    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.45)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative $L^2$ Error', fontsize=12)
    plt.title(r'AC C-LoReW-Fourier Sensitivity to $h$ ($p=2.0$)', fontsize=14)
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, f'AC_C_LoReW_fourier_h_sensitivity_L2_{seed}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved L2 comparison figure to: {output_path}')


if __name__ == "__main__":
    print('Running C-LoReW-Fourier h-sensitivity on the Allen-Cahn equation.')
    print('All settings are inherited from AC_C_LoReW_fourier.py except kernel_h.')
    print(f'Fixed kernel_p = {FIXED_KERNEL_P}')
    print(f'Seed: {seed}')
    print(f'Number of configurations: {len(HP_CONFIGS)}')

    all_results = []
    for config_id, hp_config in enumerate(HP_CONFIGS, start=1):
        print('=' * 70)
        print(
            'Configuration %d/%d: %s (h=%.5f, p=%.5f)'
            % (
                config_id,
                len(HP_CONFIGS),
                hp_config['tag'],
                hp_config['kernel_h'],
                hp_config['kernel_p'],
            )
        )
        completed_result = load_completed_result(hp_config)
        if completed_result is not None:
            print(
                'Skipping completed configuration: %s, final_L2RE=%.5e, final_Linf=%.5e'
                % (
                    completed_result['tag'],
                    completed_result['final_l2_rel'],
                    completed_result['final_l_inf'],
                )
            )
            all_results.append(completed_result)
            continue

        all_results.append(run_one_config(hp_config))

    save_summary(all_results)
    plot_l2_comparison(all_results)
