# LoReW-PINN

Code accompanying the manuscript:

**LoReW-PINN: Local Residual Weighting for Physics-Informed Neural Networks with Non-uniform Residual Fields**

This repository contains the reproducible experiment scripts for LoReW-PINN and C-LoReW-PINN, together with the baseline PINN variants used in the paper.

## Overview

LoReW-PINN assigns adaptive residual weights using a local kernel-normalized residual scale. For time-dependent PDEs, C-LoReW-PINN further combines local residual weighting with causal temporal gating. The repository includes the six PDE benchmarks reported in the manuscript:

- Poisson equation
- Helmholtz equation
- Reaction-diffusion equation
- Klein-Gordon equation
- Burgers' equation
- Allen-Cahn equation

## Repository structure

```text
LoReW-PINN/
  Poisson/Reproducible/      Poisson benchmark scripts
  Helmholtz/Reproducible/    Helmholtz benchmark scripts
  RD/Reproducible/           Reaction-diffusion benchmark scripts
  KG/Reproducible/           Klein-Gordon benchmark scripts
  Burgers/Reproducible/      Burgers benchmark scripts and reference data
  AC/Reproducible/           Allen-Cahn benchmark scripts and reference data
  */Reproducible/residual_visualization/
                             Residual-field evolution scripts used in the appendix
  docs/                      Scripts for paper-level figures
  paper_figures/             Generated figures used in the manuscript
```

Each benchmark folder contains the scripts needed for the paper's main comparisons. Sensitivity, ablation, runtime, residual-field, and generalization-gap analyses are included where applicable.

## Installation

The checked-in dependencies are pinned to the environment used to verify this release:

- Python 3.13.5
- PyTorch 2.11.0 with CUDA 12.6 (`2.11.0+cu126`)
- NumPy 2.4.3
- SciPy 1.17.1
- Matplotlib 3.10.8
- pyDOE 0.9.7

A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pinned `torch==2.11.0` requirement is hardware-agnostic. Install the PyTorch build that matches your CUDA version before installing the remaining requirements if GPU acceleration is used. Small numerical differences may still occur across PyTorch, CUDA, and GPU versions despite deterministic seeds.

## Quick start

Run a single benchmark script from the repository root, for example:

```bash
python Poisson/Reproducible/Poisson_LoReW_PINN.py
python Burgers/Reproducible/Burgers_C_LoReW_PINN.py
```

The scripts create local `logs/`, `models/`, and `figures/` folders inside each benchmark directory. These generated files are ignored by Git.

## Paper figures

After running the benchmark scripts, the plotting scripts in each benchmark folder can regenerate convergence and solution-comparison figures. The `docs/` folder contains scripts that assemble paper-level summary figures from the experiment logs:

```bash
python docs/plot_lorew_paper_figures.py
python docs/plot_lorew_hp_sensitivity.py
python docs/plot_clorew_hp_sensitivity.py
python docs/plot_clorew_ablation_time_dependent.py
```

The `paper_figures/` directory stores the generated figures used in the manuscript for reference.

## Residual-field evolution figures

The appendix residual-field figures use raw absolute residuals, a separate full-range color scale for each panel, and no percentile clipping. The final manuscript figures were generated with the following settings:

```bash
python Poisson/Reproducible/residual_visualization/Poisson_residual_visualization.py --seed 3412 --checkpoints 1000 5000 15000 --value-mode raw --scale-mode panel --clip-percentiles 0 100 --output-name poisson_residual_evolution_seed3412_raw_panel_fullrange
python Helmholtz/Reproducible/residual_visualization/Helmholtz_residual_visualization.py --seed 5472 --checkpoints 1000 5000 15000 --value-mode raw --scale-mode panel --clip-percentiles 0 100 --output-name helmholtz_residual_evolution_seed5472_raw_panel_fullrange
python RD/Reproducible/residual_visualization/RD_residual_visualization.py --seed 3053 --checkpoints 1000 5000 15000 --value-mode raw --scale-mode panel --clip-percentiles 0 100 --output-name rd_residual_evolution_seed3053_raw_panel_fullrange
python KG/Reproducible/residual_visualization/KG_residual_visualization.py --seed 1018 --checkpoints 2000 20000 30000 --value-mode raw --scale-mode panel --clip-percentiles 0 100 --output-name kg_residual_evolution_seed1018_epochs2000_20000_30000_raw_panel_fullrange
python Burgers/Reproducible/residual_visualization/Burgers_residual_visualization.py --seed 5555 --checkpoints 2000 20000 40000 --value-mode raw --scale-mode panel --clip-percentiles 0 100 --no-final-models --output-name burgers_residual_evolution_seed5555_epochs2000_20000_40000_raw_panel_fullrange
python AC/Reproducible/residual_visualization/AC_residual_visualization.py --seed 2832 --checkpoints 20000 50000 300000 --value-mode raw --scale-mode panel --clip-percentiles 0 100 --no-final-models --output-name ac_residual_evolution_seed2832_epochs20000_50000_300000_raw_panel_fullrange
```

The Burgers and Allen-Cahn commands train through their final checkpoints because trained model files are not stored in the repository. These runs are computationally expensive. Generated residual arrays and figures are written to `data/` and `figures/` inside the corresponding `residual_visualization/` directory and are ignored by Git.

## Data

Most benchmarks use manufactured analytical solutions and generate collocation points internally. Burgers' equation and Allen-Cahn include the small reference data files required by the scripts:

- `Burgers/burgers_shock.mat`
- `AC/AC.mat`

## Reproducibility notes

The main manuscript reports multi-seed statistics. Several scripts run for many iterations and may take substantial time on CPU. For faster checks, reduce the number of optimization steps inside the corresponding script before running full experiments.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Citation

If you use this code, please cite the accompanying manuscript. A `CITATION.cff` file is included for GitHub citation metadata.
