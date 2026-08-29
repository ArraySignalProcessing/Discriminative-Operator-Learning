# Third-party software

## Python packages

Python dependencies are listed in `requirements.txt` and `environment.yml`. They are installed from their normal package channels and are not vendored in this repository.

## MATLAB

MATLAB is required for simulation generation, classical baselines, and the existing plotting workflow. The exact MATLAB release and toolboxes should be recorded after final verification.

## CVX

`matlab/function/l1_SVD_DoA_est.m` uses CVX. CVX is optional because the main data-generation scripts expose switches that keep l1-SVD disabled in the manuscript workflow.

The CVX distribution, commercial solver binaries, and licenses are not included in this repository. Users who enable l1-SVD must install CVX separately and comply with its licensing terms.

## Baseline implementations

The repository contains project implementations of the classical DOA baselines and learning-based comparison pipelines used by the experiments. Before public release, the authors should review source headers and original-method licenses/citation requirements for any code derived from external implementations.
