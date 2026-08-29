# Environment

## Status

The repository records the software dependencies visible in the source code. Exact package builds and hardware-dependent runtimes must be frozen after a complete reproduction on the final experimental machine. They are not inferred from old cache files or checkpoint names.

## Recommended Python environment

- Python 3.10 or 3.11
- PyTorch 2.x
- NumPy 1.24 or later
- h5py 3.8 or later
- matplotlib 3.7 or later
- tqdm 4.65 or later

The source also uses Python standard-library modules including `argparse`, `dataclasses`, `itertools`, `math`, `pathlib`, `re`, `subprocess`, `sys`, and `time`.

PyTorch automatically selects CUDA when available in the existing scripts. CPU execution is supported, but full training and large Monte Carlo evaluation may be substantially slower.

## MATLAB environment

The MATLAB code uses matrix eigendecomposition, polynomial/root operations, HDF5 I/O, plotting, and random complex Gaussian simulation. Before publication, record the output of:

```matlab
version
ver
```

The final environment record should identify:

- MATLAB release;
- operating system;
- installed toolboxes used by the scripts;
- CVX version and solver if optional l1-SVD experiments are enabled.

## Final environment capture

After reproducing the paper on the final machine, save:

```bash
python --version
python -m pip freeze
nvidia-smi
```

Also record:

- CPU model;
- GPU model and VRAM;
- RAM;
- CUDA version;
- cuDNN version;
- training time for DMN, MLC, and SubspaceNet;
- approximate runtime for B1-B4 and C1-C2;
- peak disk usage.

## Current verification boundary

Source syntax and file correspondence can be checked without the numerical dependencies. Full numerical verification requires installing the Python packages and running MATLAB; therefore no unverified package combination is described as the original experimental environment.
