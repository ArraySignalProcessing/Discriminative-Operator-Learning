"""
Ours_train_revised.py

Train the proposed discriminative operator learning network.

Compatible HDF5 formats:
    X: covariance channels, shape (N, 2, M, M)
    Y: multi-hot DOA labels, shape (N, grid_size)
    angle_grid: fixed DOA grid in degrees

The network outputs a complex factor L, constructs Q = L L^H, trace-normalizes Q,
and optimizes a MUSIC-style response

    q(theta) = a(theta)^H Q a(theta) / (a(theta)^H a(theta)).

Revised loss design:
    1) Null loss: force q(theta_k) -> 0 at true DOAs.
    2) Margin loss: force q(theta) >= margin at all non-true grid points.
    3) Trace regularization: small numerical stabilizer.

Compared with the previous script, this version removes the 0.75-degree neighborhood
background mask. Only the exact true grid points are excluded from the background
margin loss. This makes the learned nulls sharper and is more suitable for small-angle
separation experiments.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


EPS = 1e-8


def normalize_covariance_array(covariance: np.ndarray) -> np.ndarray:
    """Normalize covariance layouts to [N, 2, M, M]."""
    if covariance.ndim == 4:
        if covariance.shape[1] == 2:
            return covariance.astype(np.float32)
        if covariance.shape[2] == 2:
            return covariance.transpose(3, 2, 0, 1).astype(np.float32)
        if covariance.shape[-1] == 2:
            return covariance.transpose(0, 3, 1, 2).astype(np.float32)

    if covariance.ndim == 5:
        if covariance.shape[2] == 2:
            return covariance.reshape(
                -1,
                covariance.shape[2],
                covariance.shape[3],
                covariance.shape[4],
            ).astype(np.float32)
        if covariance.shape[-1] == 2:
            reordered = covariance.transpose(0, 1, 4, 2, 3)
            return reordered.reshape(
                -1,
                reordered.shape[2],
                reordered.shape[3],
                reordered.shape[4],
            ).astype(np.float32)

    raise ValueError(f'Cannot normalize covariance layout, shape={covariance.shape}')


def labels_to_angles(Y: np.ndarray, angle_grid: np.ndarray, k: int) -> np.ndarray:
    """Convert multi-hot labels to sorted angle values."""
    angles = np.zeros((Y.shape[0], k), dtype=np.float32)
    for i in range(Y.shape[0]):
        idx = np.flatnonzero(Y[i] > 0.5)
        if idx.size < k:
            idx = np.argsort(Y[i])[-k:]
        idx = idx[:k]
        angles[i] = np.sort(angle_grid[idx])
    return angles


def normalize_angles_array(angles: np.ndarray, k: int) -> np.ndarray:
    """Normalize angle layouts to [N, K]."""
    if angles.ndim != 2:
        raise ValueError(f'Expected angles to be 2D, got {angles.shape}')
    if angles.shape[1] == k:
        return np.sort(angles.astype(np.float32), axis=1)
    if angles.shape[0] == k:
        return np.sort(angles.astype(np.float32), axis=0).T
    raise ValueError(f'Cannot infer angles layout, shape={angles.shape}, k={k}')


class OperatorDataset(Dataset):
    def __init__(self, h5_path: str, k: int = 2):
        with h5py.File(h5_path, 'r') as f:
            if 'X' in f:
                X = f['X'][:]
            elif 'theor' in f:
                X = f['theor'][:]
            elif 'sam' in f:
                X = f['sam'][:]
            else:
                raise KeyError('HDF5 must contain X, theor, or sam')

            self.X = normalize_covariance_array(X)

            if 'angle_grid' in f:
                self.angle_grid = f['angle_grid'][:].astype(np.float32)
            else:
                self.angle_grid = np.arange(-60.0, 60.5, 0.5, dtype=np.float32)

            if 'angles' in f:
                self.angles = normalize_angles_array(f['angles'][:], k)
            elif 'Y' in f:
                self.angles = labels_to_angles(f['Y'][:], self.angle_grid, k)
            else:
                raise KeyError('HDF5 must contain angles or Y labels')

        if self.X.shape[0] != self.angles.shape[0]:
            raise ValueError(f'X samples {self.X.shape[0]} != angle samples {self.angles.shape[0]}')

        print(f'Loaded {len(self.X)} samples from {h5_path}')
        print(f'X shape: {self.X.shape}, angles shape: {self.angles.shape}')
        print(f'Angle grid: {self.angle_grid[0]} to {self.angle_grid[-1]} deg, size={self.angle_grid.size}')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.angles[idx]).float()


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class DiscriminativeOperatorNet(nn.Module):
    def __init__(self, m: int = 10, rank: int = 8):
        super().__init__()
        self.m = m
        self.rank = rank
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
        )
        self.out = nn.Conv2d(256, 2 * rank, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def trace_normalize_input(x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, M, M]. Channel 0 is the real covariance component.
        tr = torch.diagonal(x[:, 0], dim1=-2, dim2=-1).sum(dim=-1).view(-1, 1, 1, 1)
        scale = tr / x.shape[-1]
        return x / (scale.abs() + EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trace_normalize_input(x)
        feat = self.features(x)
        z = self.out(feat)              # [B, 2r, M, M]
        z = z.mean(dim=3)               # [B, 2r, M]
        z = z.permute(0, 2, 1)          # [B, M, 2r]
        real = z[:, :, :self.rank]
        imag = z[:, :, self.rank:]
        return torch.complex(real, imag)  # [B, M, r]

    def make_Q(self, x: torch.Tensor) -> torch.Tensor:
        L = self.forward(x)
        Q = L @ L.conj().transpose(-1, -2)   # [B, M, M], Hermitian PSD
        tr = torch.diagonal(Q, dim1=-2, dim2=-1).real.sum(dim=-1).view(-1, 1, 1)
        return Q * (self.m / (tr + EPS))


def steering_vectors_torch(angles_deg: torch.Tensor, m: int, d: float = 0.5) -> torch.Tensor:
    """Return steering vectors with shape [..., M]."""
    theta = angles_deg * torch.pi / 180.0
    pos = torch.arange(m, device=angles_deg.device, dtype=angles_deg.dtype) * d
    phase = 2.0 * torch.pi * pos.view(*([1] * angles_deg.ndim), m) * torch.sin(theta).unsqueeze(-1)
    return torch.exp(1j * phase)


def response(Q: torch.Tensor, angles_deg: torch.Tensor, d: float = 0.5) -> torch.Tensor:
    """
    Q: [B, M, M]
    angles_deg: [B, K] or [G]
    Returns q = a^H Q a / M with shape [B, K] or [B, G].
    """
    _, M, _ = Q.shape

    if angles_deg.ndim == 1:
        a = steering_vectors_torch(angles_deg, M, d=d).to(Q.dtype)  # [G, M]
        Qa = torch.einsum('bmn,gn->bgm', Q, a)
        return torch.einsum('gm,bgm->bg', a.conj(), Qa).real / M

    if angles_deg.ndim == 2:
        a = steering_vectors_torch(angles_deg, M, d=d).to(Q.dtype)  # [B, K, M]
        Qa = torch.einsum('bmn,bkn->bkm', Q, a)
        return torch.einsum('bkm,bkm->bk', a.conj(), Qa).real / M

    raise ValueError(f'Unsupported angle tensor shape {angles_deg.shape}')


@dataclass
class LossParts:
    total: torch.Tensor
    null: torch.Tensor
    margin: torch.Tensor
    trace_reg: torch.Tensor


def operator_loss(
    Q: torch.Tensor,
    true_angles: torch.Tensor,
    bg_grid: torch.Tensor,
    margin: float,
    lambda_margin: float,
    lambda_trace: float,
    true_tol: float,
) -> LossParts:
    """
    Discriminative operator loss.

    Only exact true-grid points are removed from the background margin term.
    No neighborhood mask is used. This avoids encouraging a wide low-response valley
    around closely spaced sources.
    """
    q_true = response(Q, true_angles)          # [B, K]
    q_bg = response(Q, bg_grid)                # [B, G]

    diff = torch.abs(bg_grid.view(1, 1, -1) - true_angles.unsqueeze(-1))
    bg_mask = (diff.min(dim=1).values > true_tol).float()  # [B, G]

    loss_null = (q_true ** 2).mean()
    hinge = torch.relu(margin - q_bg) ** 2
    loss_margin = (hinge * bg_mask).sum() / (bg_mask.sum() + EPS)

    tr = torch.diagonal(Q, dim1=-2, dim2=-1).real.sum(dim=-1)
    loss_trace = ((tr - Q.shape[-1]) ** 2).mean()

    total = loss_null + lambda_margin * loss_margin + lambda_trace * loss_trace
    return LossParts(total, loss_null, loss_margin, loss_trace)


def train() -> None:
    parser = argparse.ArgumentParser(description='Train discriminative noise-projection operator network')
    parser.add_argument('--data', default='Ours_train.h5', help='HDF5 training data')
    parser.add_argument('--output', default='ours_model.pth')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--d', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--lambda_margin', type=float, default=1.0)
    parser.add_argument('--lambda_trace', type=float, default=1e-4)
    parser.add_argument('--bg_step', type=float, default=0.5, help='background grid step; should match training angle grid by default')
    parser.add_argument('--true_tol', type=float, default=1e-4, help='tolerance for excluding exact true DOA grid points')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = OperatorDataset(str(data_path), k=args.k)
    m = dataset.X.shape[-1]

    full_bg_grid_np = np.arange(
        float(dataset.angle_grid[0]),
        float(dataset.angle_grid[-1]) + 0.5 * args.bg_step,
        args.bg_step,
        dtype=np.float32,
    )
    bg_grid = torch.from_numpy(full_bg_grid_np).float().to(device)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = DiscriminativeOperatorNet(m=m, rank=args.rank).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=8,
    )

    best_val = float('inf')
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        model.train()
        sums = np.zeros(4, dtype=np.float64)
        count = 0

        for X, ang in tqdm(train_loader, desc=f'Epoch {epoch:3d}/{args.epochs}', leave=False, unit='batch'):
            X = X.to(device, non_blocking=True)
            ang = ang.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            Q = model.make_Q(X)
            parts = operator_loss(
                Q,
                ang,
                bg_grid,
                margin=args.margin,
                lambda_margin=args.lambda_margin,
                lambda_trace=args.lambda_trace,
                true_tol=args.true_tol,
            )
            parts.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            b = X.size(0)
            sums += b * np.array([
                parts.total.item(),
                parts.null.item(),
                parts.margin.item(),
                parts.trace_reg.item(),
            ])
            count += b

        train_avg = sums / max(count, 1)

        model.eval()
        sums = np.zeros(4, dtype=np.float64)
        count = 0

        with torch.no_grad():
            for X, ang in val_loader:
                X = X.to(device, non_blocking=True)
                ang = ang.to(device, non_blocking=True)
                Q = model.make_Q(X)
                parts = operator_loss(
                    Q,
                    ang,
                    bg_grid,
                    margin=args.margin,
                    lambda_margin=args.lambda_margin,
                    lambda_trace=args.lambda_trace,
                    true_tol=args.true_tol,
                )
                b = X.size(0)
                sums += b * np.array([
                    parts.total.item(),
                    parts.null.item(),
                    parts.margin.item(),
                    parts.trace_reg.item(),
                ])
                count += b

        val_avg = sums / max(count, 1)
        scheduler.step(val_avg[0])
        history.append([epoch, *train_avg.tolist(), *val_avg.tolist()])

        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train total/null/margin/trace='
            f'{train_avg[0]:.6g}/{train_avg[1]:.6g}/{train_avg[2]:.6g}/{train_avg[3]:.6g} | '
            f'Val total/null/margin/trace='
            f'{val_avg[0]:.6g}/{val_avg[1]:.6g}/{val_avg[2]:.6g}/{val_avg[3]:.6g} | '
            f'Time={time.perf_counter() - t0:.1f}s'
        )

        if val_avg[0] < best_val:
            best_val = float(val_avg[0])
            stale = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'm': m,
                'rank': args.rank,
                'angle_grid': dataset.angle_grid,
                'bg_grid': full_bg_grid_np,
                'args': vars(args),
            }, args.output)
            print(f'  -> Best model saved to {args.output}')
        else:
            stale += 1
            if stale >= args.patience:
                print(f'Early stopping at epoch {epoch}. Best val loss={best_val:.6g}')
                break

    history_path = args.output.replace('.pth', '_history.npz')
    np.savez(history_path, history=np.asarray(history, dtype=np.float64))
    print(f'Training finished. Best val loss={best_val:.6g}')
    print(f'History saved to {history_path}')


if __name__ == '__main__':
    train()
