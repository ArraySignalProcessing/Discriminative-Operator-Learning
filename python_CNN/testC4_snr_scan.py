"""
test_cnn_1A.py
==============
Topic 1A: evaluate the trained CNN on the SNR-scan test set.
Reads ../data/C4/C4_SnrScan.h5 and writes ../data/C4/C4_CNN.h5.
"""
# Unified MATLAB benchmark uses T=200 for Topic 1A.

import h5py
import numpy as np
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import os
import argparse
import itertools

DATA_PATH = '../data/C4/C4_SnrScan.h5'
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cnn_model.pth', help='Path to PyTorch model .pth')
args = parser.parse_args()
MODEL_PATH = args.model
OUT_PATH  = '../data/C4/C4_CNN.h5'
K = 2
ANGLE_GRID = np.arange(-60, 60.5, 0.5)
RESOLUTION_THRESHOLD = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_covariance_batch(r_sam: np.ndarray) -> np.ndarray:
    """Return covariance samples in paper layout: (N, 3, M, M)."""
    def with_phase(two_channel: np.ndarray) -> np.ndarray:
        complex_cov = two_channel[:, 0] + 1j * two_channel[:, 1]
        phase = np.angle(complex_cov).astype(np.float32)
        return np.concatenate([two_channel.astype(np.float32), phase[:, None]], axis=1)

    if r_sam.ndim != 4:
        raise ValueError(f'Expected 4D covariance array, got shape={r_sam.shape}')

    if r_sam.shape[1] == 3:
        return r_sam.transpose(0, 1, 3, 2).astype(np.float32)

    if r_sam.shape[1] == 2:
        # MATLAB HDF5 data is read as (N, C, col, row); restore matrix axes.
        return with_phase(r_sam.transpose(0, 1, 3, 2))

    if r_sam.shape[2] == 2:
        # Legacy MATLAB layout: (M, M, 2, N).
        return with_phase(r_sam.transpose(3, 2, 0, 1))

    if r_sam.shape[-1] == 2:
        # Alternate layout: (N, M, M, 2).
        return with_phase(r_sam.transpose(0, 3, 1, 2))

    raise ValueError(f'Cannot locate channel axis in sam array, shape={r_sam.shape}')


def normalize_angle_batch(angles: np.ndarray) -> np.ndarray:
    """Return sorted DOA labels in shape (N, K)."""
    if angles.ndim != 2:
        raise ValueError(f'Expected 2D angle array, got shape={angles.shape}')

    if angles.shape[0] == K:
        return np.sort(angles, axis=0).T

    if angles.shape[1] == K:
        return np.sort(angles, axis=1)

    raise ValueError(f'Cannot infer sample axis in angles array, shape={angles.shape}')


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


def parse_snr_group(group_name: str) -> float:
    """Parse HDF5 group names like SNR_10dB or SNR_min10dB."""
    match = re.search(r'SNR_(?:min)?(\d+)dB', group_name)
    if not match:
        return float('nan')

    snr_value = float(match.group(1))
    return -snr_value if 'min' in group_name else snr_value


def plot_all_predictions(snr_list, gt_list, est_list) -> None:
    """Save all SNRs in one figure with 3-column layout."""
    plot_dir = os.path.join(os.path.dirname(__file__), 'Plot_result')
    os.makedirs(plot_dir, exist_ok=True)
    n = len(snr_list)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, (snr, gt, est) in enumerate(zip(snr_list, gt_list, est_list)):
        ax = axes[i]
        sample_idx = np.arange(1, gt.shape[0] + 1)
        ax.hlines(gt[0, 0], sample_idx[0], sample_idx[-1], colors='blue', linestyles='--', linewidth=2, label='GT src 1')
        ax.hlines(gt[0, 1], sample_idx[0], sample_idx[-1], colors='orange', linestyles='--', linewidth=2, label='GT src 2')
        ax.scatter(sample_idx, est[:, 0], s=20, alpha=0.7, color='blue', marker='o')
        ax.scatter(sample_idx, est[:, 1], s=20, alpha=0.7, color='orange', marker='o')
        ax.set_title(f'SNR={snr:g}dB')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend()
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.text(0.5, 0.02, 'Sample', ha='center')
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'C4_prediction.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.close(fig)

class DeepCNN(nn.Module):
    def __init__(self, grid_size=241):
        super(DeepCNN, self).__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, grid_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x

print(f'Loading model from {MODEL_PATH}...')
model = DeepCNN(grid_size=len(ANGLE_GRID)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print('Model loaded.\n')

print(f'Opening test data: {DATA_PATH}')
with h5py.File(DATA_PATH, 'r') as f:
    groups = [g for g in f.keys() if not g.startswith('.')]
    groups = sorted(groups, key=parse_snr_group)
    print(f'Found {len(groups)} groups: {groups}\n')

    snr_vals = []
    rmse_vals = []
    por_vals = []
    plot_snrs = []
    plot_gts = []
    plot_ests = []

    for grp in groups:
        snr = parse_snr_group(grp)
        if np.isnan(snr):
            print(f'Warning: cannot parse SNR from group {grp}')

        R_sam = np.array(f[grp]['sam'])
        R_batch = normalize_covariance_batch(R_sam)
        N_mc = R_batch.shape[0]
        R_batch_tensor = torch.from_numpy(R_batch).float().to(DEVICE)

        gt = normalize_angle_batch(np.array(f[grp]['angles']))

        with torch.no_grad():
            probs = model(R_batch_tensor).cpu().numpy()  # (N_mc, grid_size)
        topk_idx = np.argpartition(probs, -K, axis=1)[:, -K:]
        est = ANGLE_GRID[topk_idx]

        est, diff = best_permutation_diff(est, gt)
        bias = diff
        print(f'  GT:{gt[0]} Pred:{est[0]} Bias:{bias[0]}')
        plot_snrs.append(snr)
        plot_gts.append(gt)
        plot_ests.append(est)

        # RMSE
        se = np.sum(diff**2, axis=1)
        rmse = np.sqrt(np.mean(se / K))

        # PoR
        max_err = np.max(np.abs(diff), axis=1)
        por = np.mean(max_err <= RESOLUTION_THRESHOLD) * 100

        snr_vals.append(snr)
        rmse_vals.append(rmse)
        por_vals.append(por)

        print(f'SNR={snr:5g} dB: RMSE={rmse:.3f} deg, PoR={por:5.1f}%')

plot_all_predictions(plot_snrs, plot_gts, plot_ests)

order = np.argsort(snr_vals)
snr_sorted = np.array(snr_vals)[order]
rmse_sorted = np.array(rmse_vals)[order]
por_sorted  = np.array(por_vals)[order]

with h5py.File(OUT_PATH, 'w') as fout:
    fout.create_dataset('CNN_RMSE', data=rmse_sorted)
    fout.create_dataset('CNN_PoR',  data=por_sorted / 100.0)  # Convert percentage to decimal
    fout.create_dataset('SNR_vec',  data=snr_sorted)

print(f'\nResults saved to {OUT_PATH}')
print('Done.')
