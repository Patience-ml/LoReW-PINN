# Paper figures

This folder contains the generated figures and summary CSV files used in the manuscript. They are included as reference outputs so readers can compare regenerated figures with the submitted version.

The six `*_residual_field_evolution.png` files are the appendix residual-field diagnostics. They show raw absolute physics residuals at selected training iterations. Each panel uses its own full data range, with no percentile clipping. The corresponding generation scripts are stored under each benchmark's `Reproducible/residual_visualization/` directory, and the exact commands are listed in the root `README.md`.

Earlier residual-weight visualizations have been replaced by these residual-field diagnostics and are not part of the current manuscript.

Training logs, trained models, residual arrays, and per-benchmark generated figures are intentionally excluded from version control.
