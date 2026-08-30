"""
testC2_coherent_sources.py
C2: coherent-source rho scan, RMSE and PoR vs rho.
Reads ../data/C2/C2_CoherentSources.h5 and writes
../data/C2/C2_CNN.h5.
"""
import h5py, numpy as np, torch, re, itertools
import matplotlib.pyplot as plt
import os
import argparse
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cnn_model.pth', help='Path to PyTorch model .pth')
args = parser.parse_args()
MODEL_PATH = args.model
DATA_PATH = '../data/C2/C2_CoherentSources.h5'
OUT_PATH  = '../data/C2/C2_CNN.h5'
K = 2
THRESHOLD = 1.0
ANGLE_GRID = np.arange(-60, 60.5, 0.5)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_rho_group(group_name: str) -> float:
    """Parse HDF5 group names like Rho_0p10 or Rho_1p00."""
    match = re.search(r'Rho_([0-9]+p[0-9]+|1p00)', group_name)
    if not match:
        return float('nan')
    return float(match.group(1).replace('p', '.'))


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
        return with_phase(r_sam.transpose(3, 2, 0, 1))

    if r_sam.shape[-1] == 2:
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


def plot_all_predictions(rho_list, gt_list, est_list) -> None:
    """Save all rho values in one figure with a 3-column layout."""
    plot_dir = os.path.join(os.path.dirname(__file__), 'Plot_result')
    os.makedirs(plot_dir, exist_ok=True)
    n = len(rho_list)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, (rho, gt, est) in enumerate(zip(rho_list, gt_list, est_list)):
        ax = axes[i]
        sample_idx = np.arange(1, gt.shape[0] + 1)
        ax.hlines(gt[0, 0], sample_idx[0], sample_idx[-1], colors='blue', linestyles='--', linewidth=2, label='GT src 1')
        ax.hlines(gt[0, 1], sample_idx[0], sample_idx[-1], colors='orange', linestyles='--', linewidth=2, label='GT src 2')
        ax.scatter(sample_idx, est[:, 0], s=20, alpha=0.7, color='blue', marker='o')
        ax.scatter(sample_idx, est[:, 1], s=20, alpha=0.7, color='orange', marker='o')
        ax.set_title(f'rho={rho:.2f}')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend()
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.text(0.5, 0.02, 'Sample', ha='center')
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'C2_prediction.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.close(fig)

class DeepCNN(nn.Module):
    def __init__(self, grid_size=241):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 2)
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
model = DeepCNN(len(ANGLE_GRID)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print(f'Opening {DATA_PATH}')
with h5py.File(DATA_PATH, 'r') as f:
    groups = [g for g in f.keys() if not g.startswith('.')]
    groups = sorted(groups, key=parse_rho_group)
    print(f'Found {len(groups)} groups: {groups}\n')
    rho_vals, rmse_vals, por_vals = [], [], []
    plot_rhos = []
    plot_gts = []
    plot_ests = []

    for grp in groups:
        rho = parse_rho_group(grp)
        R_sam = np.array(f[grp]['sam'])
        R_batch = normalize_covariance_batch(R_sam)
        R_tensor = torch.from_numpy(R_batch).float().to(DEVICE)
        gt = normalize_angle_batch(np.array(f[grp]['angles']))

        with torch.no_grad():
            probs = model(R_tensor).cpu().numpy()
        topk = np.argpartition(probs, -K, axis=1)[:, -K:]
        est = ANGLE_GRID[topk]

        est, diff = best_permutation_diff(est, gt)
        bias = diff
        print(f'  GT:{gt[0]} Pred:{est[0]} Bias:{bias[0]}')
        plot_rhos.append(rho)
        plot_gts.append(gt)
        plot_ests.append(est)
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1) / K))
        por = np.mean(np.max(np.abs(diff), axis=1) <= THRESHOLD) * 100
        rho_vals.append(rho)
        rmse_vals.append(rmse)
        por_vals.append(por)
        print(f'rho={rho:5.2f}: RMSE={rmse:.3f} deg, PoR={por:.1f}%')

    order = np.argsort(rho_vals)
    plot_all_predictions(plot_rhos, plot_gts, plot_ests)
with h5py.File(OUT_PATH, 'w') as fout:
    fout.create_dataset('rho_vec', data=np.array(rho_vals)[order])
    fout.create_dataset('CNN_RMSE', data=np.array(rmse_vals)[order])
    fout.create_dataset('CNN_PoR', data=np.array(por_vals)[order] / 100.0)  # Convert percentage to decimal
print(f'Results saved to {OUT_PATH}')
