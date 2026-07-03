import os
from collections import OrderedDict

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.max_open_warning'] = 4


SEED = 5472
LAYERS = [2] + 4 * [50] + [1]
CMAP_SOLUTION = 'rainbow'
CMAP_ERROR = 'rainbow'

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(FILE_DIR, 'models')
FIGURE_DIR = os.path.join(FILE_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh()

        layer_list = []
        for i in range(self.depth - 1):
            layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(layer.weight)
            layer_list.append((f'layer_{i}', layer))
            layer_list.append((f'activation_{i}', self.activation))

        layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(layer.weight)
        layer_list.append((f'layer_{self.depth - 1}', layer))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


METHODS = [
    {
        'name': 'Vanilla PINN',
        'model_file': f'helmholtz_pinn_model_{SEED}.pt',
    },
    {
        'name': 'SA-PINN',
        'model_file': f'helmholtz_sa_model_{SEED}.pt',
    },
    {
        'name': 'RBA-PINN',
        'model_file': f'helmholtz_rba_model_{SEED}.pt',
    },
    {
        'name': 'LoReW-PINN',
        'model_file': f'helmholtz_lorew_pinn_model_{SEED}.pt',
    },
]


def exact_u_np(x, y):
    return np.sin(4.0 * np.pi * x) * np.sin(np.pi * y)


def load_helmholtz_data():
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    nx, ny = 1001, 1001

    x = np.linspace(x_min, x_max, nx)[:, None]
    y = np.linspace(y_min, y_max, ny)[:, None]
    x_grid, y_grid = np.meshgrid(x.flatten(), y.flatten())

    exact = exact_u_np(x_grid, y_grid).T
    x_star = np.hstack((x_grid.flatten()[:, None], y_grid.flatten()[:, None]))
    return x, y, x_star, exact


def load_prediction(model_file, x_star, output_shape, batch_size=200000):
    model_path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Cannot find model checkpoint: {model_path}')

    model = DNN(LAYERS).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, x_star.shape[0], batch_size):
            batch = torch.tensor(
                x_star[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )
            predictions.append(model(batch).detach().cpu().numpy())

    ny, nx = output_shape
    return np.vstack(predictions).reshape(nx, ny).T


def format_axes(ax, show_ylabel=True):
    ax.set_xlabel(r'$x$', fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r'$y$', fontsize=13)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize=11, width=0.8, length=3)


def add_panel_label(ax, label, y=-0.34):
    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=14,
    )


def main(show=True):
    x, y, x_star, exact = load_helmholtz_data()
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    predictions = {}
    errors = {}
    l2_errors = {}
    linf_errors = {}

    for method in METHODS:
        pred = load_prediction(method['model_file'], x_star, exact.shape)
        err = np.abs(exact - pred)
        predictions[method['name']] = pred
        errors[method['name']] = err
        l2_errors[method['name']] = np.linalg.norm((exact - pred).ravel(), 2) / np.linalg.norm(exact.ravel(), 2)
        linf_errors[method['name']] = np.linalg.norm((exact - pred).ravel(), np.inf)

    solution_vmin = min([exact.min()] + [pred.min() for pred in predictions.values()])
    solution_vmax = max([exact.max()] + [pred.max() for pred in predictions.values()])

    fig = plt.figure(figsize=(21.5, 6.2))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1, 1, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.42,
        hspace=0.45,
    )

    exact_ax = fig.add_subplot(grid[0, 0])
    exact_im = exact_ax.imshow(
        exact,
        cmap=CMAP_SOLUTION,
        aspect='auto',
        extent=extent,
        origin='lower',
        vmin=solution_vmin,
        vmax=solution_vmax,
    )
    exact_ax.set_title(r'Reference solution: $u$', fontsize=14, pad=8)
    exact_ax.set_xlabel(r'$x$', fontsize=13)
    exact_ax.set_ylabel(r'$y$', fontsize=13)
    exact_ax.tick_params(axis='both', labelsize=11, width=0.8, length=3)
    add_panel_label(exact_ax, 'Ground truth')
    exact_cbar = fig.colorbar(exact_im, ax=exact_ax, fraction=0.046, pad=0.025)
    exact_cbar.ax.tick_params(labelsize=9)

    pred_axes = []
    err_axes = []

    for col, method in enumerate(METHODS, start=1):
        name = method['name']
        method_label = f'({chr(ord("a") + col - 1)}) {name}'

        pred_ax = fig.add_subplot(grid[0, col])
        pred_im = pred_ax.imshow(
            predictions[name],
            cmap=CMAP_SOLUTION,
            aspect='auto',
            extent=extent,
            origin='lower',
            vmin=solution_vmin,
            vmax=solution_vmax,
        )
        pred_ax.set_title(r'Predicted solution: $\hat{u}$', fontsize=14, pad=8)
        format_axes(pred_ax, show_ylabel=False)
        pred_ax.set_xlabel('')
        pred_cbar = fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.025)
        pred_cbar.ax.tick_params(labelsize=9)
        pred_axes.append(pred_ax)

        err_ax = fig.add_subplot(grid[1, col])
        err_im = err_ax.imshow(
            errors[name],
            cmap=CMAP_ERROR,
            aspect='auto',
            extent=extent,
            origin='lower',
            vmin=0.0,
            vmax=errors[name].max(),
        )
        err_ax.set_title(r'Absolute error: $\vert u-\hat{u}\vert$', fontsize=13, pad=8)
        format_axes(err_ax, show_ylabel=False)
        err_cbar = fig.colorbar(err_im, ax=err_ax, fraction=0.046, pad=0.025)
        err_cbar.ax.tick_params(labelsize=9)
        add_panel_label(err_ax, method_label)
        err_axes.append(err_ax)

    fig.canvas.draw()
    top_pos = pred_axes[0].get_position()
    bottom_pos = err_axes[0].get_position()
    exact_pos = exact_ax.get_position()
    exact_cbar_pos = exact_cbar.ax.get_position()
    exact_y0 = 0.5 * (top_pos.y1 + bottom_pos.y0 - exact_pos.height)
    exact_ax.set_position([exact_pos.x0, exact_y0, exact_pos.width, exact_pos.height])
    exact_cbar.ax.set_position([exact_cbar_pos.x0, exact_y0, exact_cbar_pos.width, exact_pos.height])
    exact_ax.set_title(r'Reference solution: $u$', fontsize=14, pad=8)
    fig.canvas.draw()

    output_png = os.path.join(FIGURE_DIR, 'Helmholtz_plotting_comparison.png')
    output_pdf = os.path.join(FIGURE_DIR, 'Helmholtz_plotting_comparison.pdf')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    print(f'Saved figure to: {output_png}')
    print(f'Saved figure to: {output_pdf}')

    for name in predictions:
        print(f'{name}: L2RE={l2_errors[name]:.6e}, L_inf={linf_errors[name]:.6e}')

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main(show=True)
