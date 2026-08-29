# Generated files

Generated artifacts are not stored in the source repository. This document describes their intended locations and roles.

## Training artifacts

| Producer | Primary output | Consumer |
|---|---|---|
| `python_Ours/Ours_Generate_allpairs.py` | `python_Ours/Ours_train.h5` | `python_Ours/Ours_train.py` |
| `python_Ours/Ours_train.py` | `python_Ours/ours_model.pth` | B3, B4, C1-C5 DMN tests |
| `python_CNN/CNN_Generate_allpairs.py` | `python_CNN/CNN_train.h5` | `python_CNN/CNN_train.py` |
| `python_CNN/CNN_train.py` | `python_CNN/cnn_model.pth` | C1-C5 MLC tests |
| `python_SubspaceNet/SubspaceNet_Generate_train.py` | `python_SubspaceNet/Subspace_train_rho.h5` | `python_SubspaceNet/train_SubspaceNet.py` |
| `python_SubspaceNet/train_SubspaceNet.py` | `python_SubspaceNet/subspace_model.pth` | C1-C5 SubspaceNet tests |

## Paper experiment artifacts

| Experiment | Generated directory | Main files |
|---|---|---|
| B1 | `data/B1/` | projector-deviation HDF5, parameters, and analyzed MAT results |
| B2 | `data/B2/` | coherence-gap HDF5, parameters, and analyzed MAT results |
| B3 | `data/B3/` | sample-projector responses and learned-Q responses |
| B4 | `data/B4/` | common samples and ablation learned-Q outputs |
| C1 | `data/C1/` | common observations and per-method RMSE/PoR results |
| C2 | `data/C2/` | common observations and per-method RMSE/PoR results |

## Extended experiment artifacts

Extended experiments use `data/B5/`, `data/B6/`, and `data/C3/` through `data/C5/`.

## Files excluded from source control

- `*.h5` and `*.hdf5`
- `*.mat`
- `*.pth` and `*.pt`
- `*.npz` and `*.npy`
- generated `*.png`, `*.pdf`, and `*.fig`
- training logs and cache files

Deleting generated artifacts does not remove unique source data because all simulations are produced by the included code. Retaining checkpoints can still be convenient for rapid inference, but checkpoints are not required for a from-scratch reproduction.
