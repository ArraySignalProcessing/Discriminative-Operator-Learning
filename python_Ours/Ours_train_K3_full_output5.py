"""
Ours_train_K3_full_output5.py

Train the proposed discriminative operator model on full-grid K=3 data with a
fixed output width of 5. The output width is the number of complex columns in
the network-produced factor L; it is an implementation width, not a source
count and not a classical MUSIC subspace dimension.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


EPS = 1e-8


def normalize_covariance_array(covariance: np.ndarray) -> np.ndarray:
    if covariance.ndim == 4:
        if covariance.shape[1] == 2:
            return covariance.astype(np.float32)
        if covariance.shape[2] == 2:
            return covariance.transpose(3, 2, 0, 1).astype(np.float32)
        if covariance.shape[-1] == 2:
            return covariance.transpose(0, 3, 1, 2).astype(np.float32)
    raise ValueError(f"Cannot normalize covariance layout, shape={covariance.shape}")


def labels_to_angles(Y: np.ndarray, angle_grid: np.ndarray, k: int) -> np.ndarray:
    angles = np.zeros((Y.shape[0], k), dtype=np.float32)
    for i in range(Y.shape[0]):
        idx = np.flatnonzero(Y[i] > 0.5)
        if idx.size < k:
            idx = np.argsort(Y[i])[-k:]
        angles[i] = np.sort(angle_grid[idx[:k]])
    return angles


def normalize_angles_array(angles: np.ndarray, k: int) -> np.ndarray:
    if angles.ndim != 2:
        raise ValueError(f"Expected angles to be 2D, got {angles.shape}")
    if angles.shape[1] == k:
        return np.sort(angles.astype(np.float32), axis=1)
    if angles.shape[0] == k:
        return np.sort(angles.astype(np.float32), axis=0).T
    raise ValueError(f"Cannot infer angles layout, shape={angles.shape}, k={k}")


class OperatorDataset(Dataset):
    def __init__(self, h5_path: str, k: int):
        with h5py.File(h5_path, "r") as f:
            if "X" in f:
                X = f["X"][:]
            elif "theor" in f:
                X = f["theor"][:]
            elif "sam" in f:
                X = f["sam"][:]
            else:
                raise KeyError("HDF5 must contain X, theor, or sam")

            self.X = normalize_covariance_array(X)

            if "angle_grid" in f:
                self.angle_grid = f["angle_grid"][:].astype(np.float32)
            else:
                self.angle_grid = np.arange(-60.0, 60.5, 0.5, dtype=np.float32)

            if "angles" in f:
                self.angles = normalize_angles_array(f["angles"][:], k)
            elif "Y" in f:
                self.angles = labels_to_angles(f["Y"][:], self.angle_grid, k)
            else:
                raise KeyError("HDF5 must contain angles or Y labels")

        if self.X.shape[0] != self.angles.shape[0]:
            raise ValueError(f"X samples {self.X.shape[0]} != angle samples {self.angles.shape[0]}")

        print(f"Loaded {len(self.X)} samples from {h5_path}")
        print(f"X shape: {self.X.shape}, angles shape: {self.angles.shape}")
        print(f"Angle grid: {self.angle_grid[0]} to {self.angle_grid[-1]} deg, size={self.angle_grid.size}")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
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
    def __init__(self, m: int = 10, output_width: int = 5):
        super().__init__()
        self.m = m
        self.output_width = output_width
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
        self.out = nn.Conv2d(256, 2 * output_width, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def trace_normalize_input(x: torch.Tensor) -> torch.Tensor:
        tr = torch.diagonal(x[:, 0], dim1=-2, dim2=-1).sum(dim=-1).view(-1, 1, 1, 1)
        scale = tr / x.shape[-1]
        return x / (scale.abs() + EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trace_normalize_input(x)
        feat = self.features(x)
        z = self.out(feat)
        z = z.mean(dim=3)
        z = z.permute(0, 2, 1)
        real = z[:, :, :self.output_width]
        imag = z[:, :, self.output_width:]
        return torch.complex(real, imag)

    def make_Q(self, x: torch.Tensor) -> torch.Tensor:
        L = self.forward(x)
        Q = L @ L.conj().transpose(-1, -2)
        tr = torch.diagonal(Q, dim1=-2, dim2=-1).real.sum(dim=-1).view(-1, 1, 1)
        return Q * (self.m / (tr + EPS))


def steering_vectors_torch(angles_deg: torch.Tensor, m: int, d: float = 0.5) -> torch.Tensor:
    theta = angles_deg * torch.pi / 180.0
    pos = torch.arange(m, device=angles_deg.device, dtype=angles_deg.dtype) * d
    phase = 2.0 * torch.pi * pos.view(*([1] * angles_deg.ndim), m) * torch.sin(theta).unsqueeze(-1)
    return torch.exp(1j * phase)


def response(Q: torch.Tensor, angles_deg: torch.Tensor, d: float = 0.5) -> torch.Tensor:
    _, m, _ = Q.shape
    if angles_deg.ndim == 1:
        a = steering_vectors_torch(angles_deg, m, d=d).to(Q.dtype)
        Qa = torch.einsum("bmn,gn->bgm", Q, a)
        return torch.einsum("gm,bgm->bg", a.conj(), Qa).real / m
    if angles_deg.ndim == 2:
        a = steering_vectors_torch(angles_deg, m, d=d).to(Q.dtype)
        Qa = torch.einsum("bmn,bkn->bkm", Q, a)
        return torch.einsum("bkm,bkm->bk", a.conj(), Qa).real / m
    raise ValueError(f"Unsupported angle tensor shape {angles_deg.shape}")


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
    q_true = response(Q, true_angles)
    q_bg = response(Q, bg_grid)

    diff = torch.abs(bg_grid.view(1, 1, -1) - true_angles.unsqueeze(-1))
    bg_mask = (diff.min(dim=1).values > true_tol).float()

    loss_null = (q_true ** 2).mean()
    hinge = torch.relu(margin - q_bg) ** 2
    loss_margin = (hinge * bg_mask).sum() / (bg_mask.sum() + EPS)

    tr = torch.diagonal(Q, dim1=-2, dim2=-1).real.sum(dim=-1)
    loss_trace = ((tr - Q.shape[-1]) ** 2).mean()
    total = loss_null + lambda_margin * loss_margin + lambda_trace * loss_trace
    return LossParts(total, loss_null, loss_margin, loss_trace)


def train() -> None:
    parser = argparse.ArgumentParser(description="Train K=3 full-grid output-width-5 discriminative operator model")
    parser.add_argument("--data", default="Ours_train_K3_full_output5.h5")
    parser.add_argument("--output", default="ours_model_K3_full_output5.pth")
    parser.add_argument("--output_width", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--d", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda_margin", type=float, default=1.0)
    parser.add_argument("--lambda_trace", type=float, default=1e-4)
    parser.add_argument("--bg_step", type=float, default=0.5)
    parser.add_argument("--true_tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    model = DiscriminativeOperatorNet(m=m, output_width=args.output_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=8,
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        sums = np.zeros(4, dtype=np.float64)
        count = 0

        for X, ang in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False, unit="batch"):
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
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append([epoch, *train_avg.tolist(), *val_avg.tolist(), current_lr])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train total/null/margin/trace="
            f"{train_avg[0]:.6g}/{train_avg[1]:.6g}/{train_avg[2]:.6g}/{train_avg[3]:.6g} | "
            f"Val total/null/margin/trace="
            f"{val_avg[0]:.6g}/{val_avg[1]:.6g}/{val_avg[2]:.6g}/{val_avg[3]:.6g} | "
            f"LR={current_lr:.3g} | Time={time.perf_counter() - t0:.1f}s"
        )

        if val_avg[0] < best_val:
            best_val = float(val_avg[0])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "m": m,
                    "output_width": args.output_width,
                    "angle_grid": dataset.angle_grid,
                    "bg_grid": full_bg_grid_np,
                    "args": vars(args),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                args.output,
            )
            print(f"  -> Best model saved to {args.output}")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best val loss={best_val:.6g}")
                break

    history_path = args.output.replace(".pth", "_history.npz")
    history_columns = np.asarray([
        "epoch",
        "train_total",
        "train_null",
        "train_margin",
        "train_trace",
        "val_total",
        "val_null",
        "val_margin",
        "val_trace",
        "lr",
    ])
    np.savez(
        history_path,
        history=np.asarray(history, dtype=np.float64),
        columns=history_columns,
        best_val=np.asarray(best_val, dtype=np.float64),
        best_epoch=np.asarray(best_epoch, dtype=np.int64),
        output_width=np.asarray(args.output_width, dtype=np.int64),
    )
    print(f"Training finished. Best epoch={best_epoch}, best val loss={best_val:.6g}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    train()
