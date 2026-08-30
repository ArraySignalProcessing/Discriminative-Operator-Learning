# Third-party software and baseline attribution

## Python packages

Python dependencies are listed in `requirements.txt` and `environment.yml`. They are installed from their normal package channels and are not vendored in this repository.

## MATLAB

MATLAB is required for simulation generation, classical baselines, and the existing plotting workflow. The exact MATLAB release and toolboxes should be recorded after final verification.

## CVX

`matlab/function/l1_SVD_DoA_est.m` uses CVX. CVX is optional because the main data-generation scripts expose switches that keep l1-SVD disabled in the manuscript workflow.

The CVX distribution, commercial solver binaries, and licenses are not included in this repository. Users who enable l1-SVD must install [CVX](https://cvxr.com/cvx/) separately and comply with its licensing terms.

## Baseline implementations

The repository contains local implementations and experiment adapters for the classical and learning-based baselines used in the manuscript. They are integrated with the common HDF5 datasets, array convention, metrics, and plotting workflow of this project; external repositories are not vendored as submodules or copied as complete distributions.

### MLC

The grid-based multi-label classifier stored under the historical directory name `python_CNN` follows the method and DeepCNN architecture described in:

- G. K. Papageorgiou, M. Sellathurai, and Y. C. Eldar, "Deep Networks for Direction-of-Arrival Estimation in Low SNR," *IEEE Transactions on Signal Processing*, vol. 69, pp. 3714--3729, 2021. [doi:10.1109/TSP.2021.3089927](https://doi.org/10.1109/TSP.2021.3089927)

The DeepCNN implementation in the [ShlezingerLab SubspaceNet repository](https://github.com/ShlezingerLab/SubspaceNet) was used as an architectural reference. The local implementation uses this project's covariance format, angle grid, data generator, training loop, and evaluation interface.

### SubspaceNet

The SubspaceNet baseline follows:

- D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun, and N. Shlezinger, "SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation," *IEEE Transactions on Vehicular Technology*, vol. 74, no. 3, pp. 4962--4976, 2025. [doi:10.1109/TVT.2024.3496119](https://doi.org/10.1109/TVT.2024.3496119)

Its convolution/deconvolution architecture and differentiable Root-MUSIC flow follow the corresponding [official implementation](https://github.com/ShlezingerLab/SubspaceNet). The local code adapts the model to this project's HDF5 data, steering convention, training settings, checkpoint format, and evaluation scripts.

## License boundary

The repository-level MIT license covers project-authored source code and documentation. It does not replace the licenses or usage terms of separately installed Python packages, MATLAB, CVX, or external software. The papers and repositories listed above should be cited when reporting results obtained with the corresponding baselines.
