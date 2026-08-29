# Extended experiments

The experiments in this document are included as supplementary investigations. They are not used for the main figures or quantitative claims of the current manuscript.

## B5: inference-backend consistency

Purpose: compare increasingly fine grid inference against the Root-MUSIC-type backend for the learned operator.

Scripts:

1. `matlab/generateB5_backend_consistency_data.m`
2. `python_Ours/expB5_backend_consistency.py`
3. `matlab/plotB5_backend_consistency.m`

## B6: source-count extension

Purpose: explore learned-operator spectra for one, two, and three sources.

Scripts:

1. `python_Ours/Ours_Generate_train_K3_coherent.py`
2. `python_Ours/Ours_Generate_train_K3_full_output5.py`
3. `python_Ours/Ours_train_K3_coherent.py`
4. `python_Ours/Ours_train_K3_full_output5.py`
5. `matlab/generateB6_k123_composite_spectrum_data.m`
6. `python_Ours/expB6_k123_composite_spectrum_export_Q.py`
7. `matlab/plotB6_k123_composite_spectrum.m`
8. `matlab/plotB6_k123_interactive_slices.m`

Release status: exploratory. Before presenting B6 as fully reproducible, confirm which K=3 training script and checkpoint produced the final B6 results. The export script's historical default checkpoint name does not uniquely identify the retained development checkpoint.

## C3: array mismatch

Purpose: test robustness to sensor-position perturbations, gain/phase errors, and mutual coupling.

Scripts:

- `matlab/generateC3_array_mismatch_data.m`
- `matlab/baselineC3_array_mismatch.m`
- `python_Ours/testC3_array_mismatch.py`
- `python_Ours/testC3_array_mismatch_root.py`
- `python_CNN/testC3_array_mismatch.py`
- `python_SubspaceNet/testC3_array_mismatch.py`
- `matlab/plotC3_array_mismatch.m`

## C4: SNR sweep

Purpose: conventional robustness evaluation over SNR.

Scripts:

- `matlab/generateC4_snr_scan_data.m`
- `matlab/baselineC4_snr_scan.m`
- `python_Ours/testC4_snr_scan.py`
- `python_Ours/testC4_snr_scan_root.py`
- `python_CNN/testC4_snr_scan.py`
- `python_SubspaceNet/testC4_snr_scan.py`
- `matlab/plotC4_snr_scan.m`

## C5: snapshot-count sweep

Purpose: conventional finite-sample robustness evaluation.

Scripts:

- `matlab/generateC5_snapshot_scan_data.m`
- `matlab/baselineC5_snapshot_scan.m`
- `python_Ours/testC5_snapshot_scan.py`
- `python_Ours/testC5_snapshot_scan_root.py`
- `python_CNN/testC5_snapshot_scan.py`
- `python_SubspaceNet/testC5_snapshot_scan.py`
- `matlab/plotC5_snapshot_scan.m`

## Root-Ours

The `testC*_root.py` scripts apply a Root-MUSIC-type polynomial backend to the learned operator. They are retained to study backend behavior but are separate from the default grid-spectrum inference used for the proposed-method curves in the current manuscript.
