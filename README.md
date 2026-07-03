# LoReW-PINN

Code accompanying the manuscript:

**LoReW-PINN: Local Residual Weighting for Adaptive Physics-Informed Neural Network Training**

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
  docs/                      Scripts for paper-level figures
  paper_figures/             Generated figures used in the manuscript
```

Each benchmark folder contains scripts for vanilla PINN, SA-PINN, RBA-PINN, LoReW-PINN, and/or C-LoReW-PINN as appropriate for the PDE type. Time-dependent benchmarks also include ablation and hyperparameter-sensitivity scripts.

## Installation

The experiments were developed with Python and PyTorch. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if GPU acceleration is used.

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

## Data

Most benchmarks use manufactured analytical solutions and generate collocation points internally. Burgers' equation and Allen-Cahn include the small reference data files required by the scripts:

- `Burgers/burgers_shock.mat`
- `AC/AC.mat`

## Reproducibility notes

The main manuscript reports multi-seed statistics. Several scripts run for many iterations and may take substantial time on CPU. For faster checks, reduce the number of optimization steps inside the corresponding script before running full experiments.

## License

No open-source license has been specified yet. Add a license file before public release if reuse terms should be explicit.

## Citation

If you use this code, please cite the accompanying manuscript. A `CITATION.cff` file is included for GitHub citation metadata.
