# AO DL Turbulence Compensation

PyTorch project workspace for deep-learning-assisted adaptive optics turbulence compensation.

The first training target is phase compensation for off-axis holography reconstructed complex optical fields. The initial weak-turbulence reference is the 2x magnification, 40 degree temperature-difference optical field.

## Layout

- `src/`: training, dataset, model, loss, and evaluation code.
- `configs/`: training configuration files.
- `docs/`: method notes, literature notes, and experiment records.
- `data/raw_matlab_exports/`: MATLAB-exported intensity and phase data. Data files are ignored by Git.
- `data/processed/`: preprocessed training caches. Files are ignored by Git.
- `checkpoints/`: model checkpoints. Files are ignored by Git.
- `outputs/`: figures, metrics, and logs. Files are ignored by Git.

## Data Policy

Large data files, training outputs, and model weights should stay out of Git. Keep code, configuration, and small documentation files under version control.
