"""test_subspace_1C.py
Topic 1C - RMSE & PoR vs angular separation.
Read ../data/C1/C1_CloseSource.h5 and write ../data/C1/C1_SubspaceNet.h5.
"""

import argparse
import itertools
import os
import re

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_SubspaceNet import SubspaceNet, root_music_torch


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='subspace_model.pth', help='Path to PyTorch model .pth')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

MODEL_PATH = args.model
DATA_PATH = '../data/C1/C1_CloseSource.h5'
OUT_PATH = '../data/C1/C1_SubspaceNet.h5'
K = 2
THRESHOLD = 1.0 # degrees, for PoR calculation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_group(group_name: str) -> float:
    match = re.search(r'Sep_(\d+)deg', group_name)
    return float(match.group(1)) if match else float('nan')


def normalize_angle_batch(angles: np.ndarray, k: int = K) -> np.ndarray:
    if angles.ndim == 1:
        angles = angles.reshape(-1, 1)
    if angles.shape[0] == k:
        return np.sort(angles.astype(np.float32), axis=0).T
    if angles.shape[1] >= k:
        return np.sort(angles[:, :k].astype(np.float32), axis=1)
    raise ValueError(f'Cannot infer sample axis in angles array, shape={angles.shape}')


def normalize_tau_corr(tau_corr: np.ndarray) -> np.ndarray:
    if tau_corr.ndim != 4:
        raise ValueError(f'Expected 4D tau_corr data, shape={tau_corr.shape}')
    if tau_corr.shape[2] == 2 * tau_corr.shape[3]:
        return tau_corr.astype(np.float32)
    if tau_corr.shape[1] == 2 * tau_corr.shape[0]:
        return tau_corr.transpose(3, 2, 1, 0).astype(np.float32)
    if tau_corr.shape[-1] == 2 * tau_corr.shape[-2]:
        return tau_corr.transpose(0, 1, 3, 2).astype(np.float32)
    raise ValueError(f'Cannot normalize tau_corr layout, shape={tau_corr.shape}')


def best_permutation_diff(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return per-sample errors after minimum-SSE source permutation matching."""
    if est.shape != gt.shape:
        raise ValueError(f'est shape {est.shape} does not match gt shape {gt.shape}')

    best_est = None
    best_diff = None
    best_se = None
    for perm in itertools.permutations(range(est.shape[1])):
        est_perm = est[:, perm]
        diff = est_perm - gt
        se = np.sum(diff ** 2, axis=1)
        if best_diff is None:
            best_est = est_perm.copy()
            best_diff = diff.copy()
            best_se = se
            continue
        mask = se < best_se
        best_est[mask] = est_perm[mask]
        best_diff[mask] = diff[mask]
        best_se[mask] = se[mask]
    return best_est, best_diff


def plot_all_predictions(x_list, gt_list, est_list) -> None:
    plot_dir = os.path.join(os.path.dirname(__file__), 'Plot_result')
    os.makedirs(plot_dir, exist_ok=True)
    n = len(x_list)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, (x_value, gt, est) in enumerate(zip(x_list, gt_list, est_list)):
        ax = axes[i]
        sample_idx = np.arange(1, gt.shape[0] + 1)
        ax.hlines(gt[0, 0], sample_idx[0], sample_idx[-1], colors='blue', linestyles='--', linewidth=2, label='GT src 1')
        ax.hlines(gt[0, 1], sample_idx[0], sample_idx[-1], colors='orange', linestyles='--', linewidth=2, label='GT src 2')
        ax.scatter(sample_idx, est[:, 0], s=20, alpha=0.7, color='blue', marker='o')
        ax.scatter(sample_idx, est[:, 1], s=20, alpha=0.7, color='orange', marker='o')
        ax.set_title(f'Delta={x_value:g} deg')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.text(0.5, 0.02, 'Sample', ha='center')
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'C1_prediction.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.close(fig)


print(f'Loading model from {MODEL_PATH}...')
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
k = int(checkpoint.get('k', K))
output_unit = checkpoint.get('output_unit', 'deg')
sort_output = bool(checkpoint.get('sort_output', True))
model = SubspaceNet(N=int(checkpoint['N']), tau=int(checkpoint['tau']), eps=float(checkpoint.get('eps', 1.0)),
                    M=int(checkpoint.get('M', k)), diff_method=checkpoint.get('diff_method', 'root_music'),
                    output_unit=output_unit, sort_output=sort_output).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f'Model loaded. output_unit={output_unit}, sort_output={sort_output}\n')

print(f'Opening {DATA_PATH}')
with h5py.File(DATA_PATH, 'r') as f:
    groups = sorted([g for g in f.keys() if not g.startswith('.')], key=parse_group)
    print(f'Found {len(groups)} groups: {groups}\n')
    x_vals, rmse_vals, por_vals = [], [], []
    plot_x, plot_gts, plot_ests = [], [], []

    for grp in groups:
        x_value = parse_group(grp)
        tau_batch = normalize_tau_corr(np.array(f[grp]['tau_corr']))
        gt = normalize_angle_batch(np.array(f[grp]['angles']), k)
        predictions = []
        with torch.no_grad():
            for start in range(0, tau_batch.shape[0], args.batch_size):
                batch = torch.from_numpy(tau_batch[start:start + args.batch_size]).float().to(DEVICE)
                doa_pred, _, _, _ = model(batch)
                predictions.append(doa_pred.cpu().numpy())
        est = np.vstack(predictions).astype(np.float32)
        if output_unit == 'rad':
            est = np.rad2deg(est)
        elif output_unit != 'deg':
            raise ValueError(f'Unsupported SubspaceNet output_unit: {output_unit}')

        est, diff = best_permutation_diff(est, gt)
        bias = diff
        print(f'  GT:{gt[0]} Pred:{est[0]} Bias:{bias[0]}')
        rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1) / k)))
        por = float(np.mean(np.max(np.abs(diff), axis=1) <= THRESHOLD) * 100.0)
        x_vals.append(x_value)
        rmse_vals.append(rmse)
        por_vals.append(por)
        plot_x.append(x_value)
        plot_gts.append(gt)
        plot_ests.append(est)
        print(f'Delta={x_value:g} deg: RMSE={rmse:.3f} deg, PoR={por:.1f}%')

order = np.argsort(x_vals)
plot_all_predictions(plot_x, plot_gts, plot_ests)
with h5py.File(OUT_PATH, 'w') as fout:
    fout.create_dataset('SubspaceNet_RMSE', data=np.array(rmse_vals, dtype=np.float32)[order])
    fout.create_dataset('SubspaceNet_PoR', data=np.array(por_vals, dtype=np.float32)[order] / 100.0)
    fout.create_dataset('delta_theta_vec', data=np.array(x_vals, dtype=np.float32)[order])
print(f'Results saved to {OUT_PATH}')
