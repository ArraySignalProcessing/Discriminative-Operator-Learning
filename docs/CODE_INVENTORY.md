# Code inventory

This inventory classifies the retained source code without changing source filenames or implementations.

## Classification

- **CORE**: directly supports the current manuscript experiments B1-B4 or C1-C2.
- **EXTENDED**: supports B5-B6, C3-C5, K=3 studies, or alternative Root-Ours inference.
- **SHARED**: common training, simulation, metric, plotting, or utility code used across experiments.
- **OPTIONAL**: code used only when an optional baseline or third-party dependency is enabled.

## Proposed method (`python_Ours`)

### Core and shared

| File | Class | Role |
|---|---|---|
| `Ours_Generate_allpairs.py` | CORE | Generates the 202,440 all-pairs population-covariance training set. |
| `Ours_Generate_train.py` | SHARED | Random training-data generator for alternative training studies. |
| `Ours_train.py` | CORE | Defines and trains the proposed discrimination-operator network. |
| `ours_test_common.py` | SHARED | Common loading, spectrum, peak detection, metrics, and test I/O. |
| `expB3_operator_response_export_Q.py` | CORE | Exports learned operators for B3. |
| `trainB4_operator_ablation.py` | CORE | Trains B4 projector-fitting and null-only ablations. |
| `expB4_operator_ablation_export_Q.py` | CORE | Exports the B4 ablation operators. |
| `testC1_close_source.py` | CORE | Grid-backend proposed-method evaluation for C1. |
| `testC2_coherent_sources.py` | CORE | Grid-backend proposed-method evaluation for C2. |

### Extended

| File | Class | Role |
|---|---|---|
| `expB5_backend_consistency.py` | EXTENDED | B5 grid/root backend comparison. |
| `Ours_Generate_train_K3_coherent.py` | EXTENDED | K=3 correlated-source training generation. |
| `Ours_Generate_train_K3_full_output5.py` | EXTENDED | Alternative K=3/output-width training generation. |
| `Ours_train_K3_coherent.py` | EXTENDED | K=3 coherent-model training wrapper. |
| `Ours_train_K3_full_output5.py` | EXTENDED | Alternative K=3/output-width model training. |
| `expB6_k123_composite_spectrum_export_Q.py` | EXTENDED | B6 learned-operator export for K=1,2,3. |
| `testC1_close_source_root.py` | EXTENDED | Root-Ours evaluation for C1. |
| `testC2_coherent_sources_root.py` | EXTENDED | Root-Ours evaluation for C2. |
| `testC3_array_mismatch.py` | EXTENDED | Grid-backend proposed-method evaluation for C3. |
| `testC3_array_mismatch_root.py` | EXTENDED | Root-Ours evaluation for C3. |
| `testC4_snr_scan.py` | EXTENDED | Grid-backend proposed-method evaluation for C4. |
| `testC4_snr_scan_root.py` | EXTENDED | Root-Ours evaluation for C4. |
| `testC5_snapshot_scan.py` | EXTENDED | Grid-backend proposed-method evaluation for C5. |
| `testC5_snapshot_scan_root.py` | EXTENDED | Root-Ours evaluation for C5. |

## MLC (`python_CNN`)

The source directory retains the historical name `python_CNN`, while the manuscript uses the name MLC.

| File | Class | Role |
|---|---|---|
| `CNN_Generate_allpairs.py` | CORE | Generates the all-pairs MLC training set. |
| `CNN_Generate_train.py` | SHARED | Random MLC training-data generator. |
| `CNN_train.py` | CORE | Trains the MLC baseline. |
| `testC1_close_source.py` | CORE | MLC evaluation for C1. |
| `testC2_coherent_sources.py` | CORE | MLC evaluation for C2. |
| `testC3_array_mismatch.py` | EXTENDED | MLC evaluation for C3. |
| `testC4_snr_scan.py` | EXTENDED | MLC evaluation for C4. |
| `testC5_snapshot_scan.py` | EXTENDED | MLC evaluation for C5. |

## SubspaceNet (`python_SubspaceNet`)

| File | Class | Role |
|---|---|---|
| `SubspaceNet_Generate_train.py` | CORE | Generates snapshot-derived SubspaceNet training features. |
| `train_SubspaceNet.py` | CORE | Trains the SubspaceNet baseline. |
| `testC1_close_source.py` | CORE | SubspaceNet evaluation for C1. |
| `testC2_coherent_sources.py` | CORE | SubspaceNet evaluation for C2. |
| `testC3_array_mismatch.py` | EXTENDED | SubspaceNet evaluation for C3. |
| `testC4_snr_scan.py` | EXTENDED | SubspaceNet evaluation for C4. |
| `testC5_snapshot_scan.py` | EXTENDED | SubspaceNet evaluation for C5. |

## MATLAB experiment scripts

### Current manuscript

| Experiment | Core scripts |
|---|---|
| B1 | `generateB1_projection_deviation_data.m`, `analyzeB1_projection_deviation.m`, `plotB1_projection_deviation.m` |
| B2 | `generateB2_coherence_gap_data.m`, `analyzeB2_coherence_gap.m`, `plotB2_coherence_gap.m` |
| B3 | `generateB3_operator_response_data.m`, `plotB3_operator_response.m` |
| B4 | `generateB4_operator_ablation.m`, `plotB4_operator_ablation.m` |
| C1 | `generateC1_close_source_data.m`, `baselineC1_close_source.m`, `plotC1_close_source.m` |
| C2 | `generateC2_coherent_sources_data.m`, `baselineC2_coherent_sources.m`, `plotC2_coherent_sources.m` |

`format_ieee_axes.m` and `export_ieee_figure.m` are shared figure-formatting utilities.

### Extended experiments

| Experiment | Extended scripts |
|---|---|
| B5 | `generateB5_backend_consistency_data.m`, `plotB5_backend_consistency.m` |
| B6 | `generateB6_k123_composite_spectrum_data.m`, `plotB6_k123_composite_spectrum.m`, `plotB6_k123_interactive_slices.m` |
| C3 | `generateC3_array_mismatch_data.m`, `baselineC3_array_mismatch.m`, `plotC3_array_mismatch.m` |
| C4 | `generateC4_snr_scan_data.m`, `baselineC4_snr_scan.m`, `plotC4_snr_scan.m` |
| C5 | `generateC5_snapshot_scan_data.m`, `baselineC5_snapshot_scan.m`, `plotC5_snapshot_scan.m` |

## MATLAB helper functions

| File | Class | Role |
|---|---|---|
| `ang_spec.m` | SHARED | Steering/spectrum helper. |
| `build_tau_corr.m` | SHARED | Builds multi-lag correlation features for SubspaceNet. |
| `conv_cov2vec.m` | SHARED | Covariance-format conversion. |
| `conv2matcom.m` | SHARED | Complex-matrix conversion. |
| `ESPRIT_doa.m` | SHARED | ESPRIT baseline. |
| `match_doa_error.m` | SHARED | Permutation-aware DOA matching/error. |
| `Q_mat.m` | SHARED | Matrix construction helper. |
| `shrinkage_covariance.m` | EXTENDED | Shrinkage covariance used in robustness baselines. |
| `toeplitz_covariance_reconstruct.m` | EXTENDED | Toeplitz reconstruction baseline helper. |
| `unit_ESPRIT.m` | EXTENDED | Unitary ESPRIT implementation. |
| `unit_ESPRIT_fast.m` | EXTENDED | Accelerated unitary ESPRIT implementation. |
| `l1_SVD_DoA_est.m` | OPTIONAL | l1-SVD baseline; requires separately installed CVX. |
| `optimize_threshold_l1SVD.m` | OPTIONAL | l1-SVD threshold tuning; requires CVX path. |

## Excluded artifact classes

The curated source repository intentionally excludes generated HDF5/MAT data, checkpoints, NumPy histories, cache files, generated figures, manuscript files, and the vendored CVX distribution. Their absence does not remove experimental source logic.
