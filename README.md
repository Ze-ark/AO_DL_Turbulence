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

## Quick Start

Create a project-local Python 3.13 environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Generate a small MATLAB smoke-test dataset:

```matlab
addpath(fullfile(pwd, 'matlab'));
simulate_gaussian_turbulence_dataset( ...
    fullfile(pwd, 'data', 'raw_matlab_exports', 'sim_gaussian_v1.h5'), ...
    50, ...
    64);
```

Generate the planned v1 dataset by increasing the arguments to `1000` frames and `256` resolution.

Export the real 2x off-axis hologram validation set through the MATLAB reconstruction pipeline:

```matlab
cd('C:\Users\Lintianze\OneDrive\Desktop\AO_DL_Turbulence')
addpath(fullfile(pwd, 'matlab'))

export_real_offaxis_validation_dataset( ...
    fullfile(pwd, 'data', 'real_validation', 'real_offaxis_2x_validation.h5'), ...
    'E:\加扩束镜\2倍放大', ...
    200, ...
    [256 256])
```

Train and evaluate:

```powershell
.\.venv\Scripts\python.exe -m src.train_compensation --config configs/sim_gaussian_v1.yaml
.\.venv\Scripts\python.exe -m src.evaluate_compensation --config configs/sim_gaussian_v1.yaml
.\.venv\Scripts\python.exe -m src.validate_real_experiment --config configs/sim_gaussian_v1.yaml
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
