#!/usr/bin/env python3
"""Paper-aligned SubspaceNet baseline for DoA estimation.

This module follows the Shmuel et al. SubspaceNet / Deep Root-MUSIC flow:

    multi-lag autocorrelation -> autoencoder surrogate covariance -> Root-MUSIC -> DOA

The project-wide steering convention is

    a(theta) = exp(+j*pi*n*sin(theta))

so the Root-MUSIC angle recovery uses sin(theta)=angle(z)/pi for the selected polynomial roots.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

EPS = 1e-12


def polyroot_torch_desc(coeffs: torch.Tensor) -> torch.Tensor:
    """Return polynomial roots for coefficients in descending order."""
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be a 1D tensor")

    degree = coeffs.numel() - 1
    if degree < 1:
        return torch.empty(0, dtype=torch.complex128, device=coeffs.device)

    lead = coeffs[0]
    if torch.abs(lead) < EPS:
        lead = lead + coeffs.new_tensor(EPS)
    normalized = coeffs / lead

    companion = torch.zeros((degree, degree), dtype=torch.complex128, device=coeffs.device)
    companion[0, :] = -normalized[1:].to(torch.complex128)
    if degree > 1:
        companion[1:, :-1] = torch.eye(degree - 1, dtype=torch.complex128, device=coeffs.device)

    eigvals, _ = torch.linalg.eig(companion)
    return eigvals


def root_music_torch(
    Rz: torch.Tensor,
    k: int,
    output_unit: str = "rad",
    sort_output: bool = False,
) -> torch.Tensor:
    """Differentiable Root-MUSIC for a ULA with half-wavelength spacing.

    The paper implementation returns angles in radians.  ``output_unit`` and
    ``sort_output`` are kept only to load old local checkpoints faithfully.
    """
    batch_size, n_sensors = Rz.shape[0], Rz.shape[-1]
    device = Rz.device
    doas_batch = []

    for b in range(batch_size):
        R = Rz[b]
        eigvals, eigvecs = torch.linalg.eig(R)
        idx = torch.argsort(eigvals.real, descending=True)
        Un = eigvecs[:, idx[k:]]
        F = Un @ Un.conj().T

        coeffs = torch.zeros(2 * n_sensors - 1, dtype=torch.complex128, device=device)
        for lag in range(-(n_sensors - 1), n_sensors):
            coeffs[lag + (n_sensors - 1)] = torch.diagonal(F, offset=lag).sum().to(torch.complex128)

        roots = polyroot_torch_desc(coeffs.flip(0))
        inside = torch.abs(roots) <= 1.0
        candidates_pool = roots[inside]
        if candidates_pool.numel() < k:
            candidates_pool = roots

        dist_to_unit = torch.abs(1.0 - torch.abs(candidates_pool))
        _, sorted_idx = torch.sort(dist_to_unit)
        candidates = candidates_pool[sorted_idx[:k]]

        angles_rad = torch.angle(candidates)
        sin_theta = angles_rad / np.pi
        sin_theta = torch.clamp(sin_theta, -0.9999, 0.9999)
        doas_rad = torch.arcsin(sin_theta)
        if sort_output:
            doas_rad = torch.sort(doas_rad).values

        if output_unit == "rad":
            doas = doas_rad
        elif output_unit == "deg":
            doas = doas_rad * 180.0 / np.pi
        else:
            raise ValueError(f"Unsupported Root-MUSIC output unit: {output_unit}")
        doas_batch.append(doas)

    return torch.stack(doas_batch, dim=0)


def periodic_error(ang_true: torch.Tensor, ang_pred: torch.Tensor, period: float = np.pi) -> torch.Tensor:
    diff = (ang_true - ang_pred) % period
    return torch.where(diff > period / 2.0, diff - period, diff)


def rmspe_loss(doa_true: torch.Tensor, doa_pred: torch.Tensor, period: float = np.pi) -> torch.Tensor:
    """Permutation-invariant RMSPE loss in radians, matching SubspaceNet."""
    batch_size, k = doa_true.shape
    if doa_pred.shape[1] != k:
        pad = torch.zeros(batch_size, k - doa_pred.shape[1], device=doa_true.device)
        doa_pred = torch.cat([doa_pred, pad], dim=1)

    per_perm_rmspe = []
    for perm in itertools.permutations(range(k)):
        perm_tensor = torch.tensor(perm, device=doa_true.device)
        err = periodic_error(doa_true, doa_pred[:, perm_tensor], period=period)
        per_perm_rmspe.append(torch.linalg.norm(err, dim=1) / np.sqrt(k))

    best_rmspe = torch.stack(per_perm_rmspe, dim=1).min(dim=1).values
    return best_rmspe.mean()


class SubspaceNet(nn.Module):
    """Official SubspaceNet architecture with the local HDF5 interface.

    The convolution/deconvolution stack follows the ShlezingerLab SubspaceNet
    implementation: kernel-size 2 layers, anti-rectifier activations, dropout,
    Gram diagonal loading, and a differentiable Root-MUSIC head.
    """

    def __init__(self, N: int, tau: int = 5, eps: float = 1.0, M: int = 2,
                 diff_method: str = "root_music", output_unit: str = "rad",
                 sort_output: bool = False):
        super().__init__()
        self.N = N
        self.tau = tau
        self.eps = eps
        self.M = M
        self.diff_method = diff_method
        self.output_unit = output_unit
        self.sort_output = sort_output

        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()

    def anti_rectifier(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.ReLU(x), self.ReLU(-x)), dim=1)

    def surrogate_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Map tau features ``[B,tau,2N,N]`` to surrogate covariance ``Rz``."""
        self.N = x.shape[-1]
        x = self.anti_rectifier(self.conv1(x))
        x = self.anti_rectifier(self.conv2(x))
        x = self.anti_rectifier(self.conv3(x))
        x = self.anti_rectifier(self.deconv2(x))
        x = self.anti_rectifier(self.deconv3(x))
        x = self.DropOut(x)
        x = self.deconv4(x).squeeze(1)

        real = x[:, :self.N, :]
        imag = x[:, self.N:, :]
        K_mat = torch.complex(real, imag)
        Rz = K_mat @ K_mat.conj().transpose(-1, -2)
        Rz = Rz + self.eps * torch.eye(self.N, dtype=Rz.dtype, device=Rz.device)
        return Rz

    def forward(self, x: torch.Tensor):
        """Official-style forward: return Root-MUSIC output and surrogate covariance."""
        Rz = self.surrogate_covariance(x)
        if not self.diff_method.startswith("root_music"):
            raise ValueError(f"Unsupported SubspaceNet diff_method: {self.diff_method}")
        doa_prediction = root_music_torch(Rz, self.M, output_unit=self.output_unit, sort_output=self.sort_output)
        return doa_prediction, None, None, Rz


class DeepRootMusicDataset(Dataset):
    """HDF5 dataset containing tau-correlation features and DOA labels."""

    @staticmethod
    def normalize_tau_corr(x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tau_corr input, got {x.shape}")
        if x.shape[2] == 2 * x.shape[3]:
            return x.astype(np.float32)
        if x.shape[1] == 2 * x.shape[0]:
            return x.transpose(3, 2, 1, 0).astype(np.float32)
        if x.shape[-1] == 2 * x.shape[-2]:
            return x.transpose(0, 1, 3, 2).astype(np.float32)
        raise ValueError(f"Cannot normalize tau_corr layout, shape={x.shape}")

    def __init__(self, h5_path: str, k: int = 2):
        with h5py.File(h5_path, "r") as f:
            if "tau_corr" in f:
                raw_x = f["tau_corr"][:]
            elif "X" in f:
                raw_x = f["X"][:]
            else:
                raise KeyError("HDF5 must contain 'tau_corr' or 'X'")
            self.x = self.normalize_tau_corr(raw_x)

            if "angles" in f:
                self.angles = f["angles"][:].astype(np.float32)
            elif "Y" in f:
                angle_grid = f["angle_grid"][:] if "angle_grid" in f else np.arange(-60.0, 60.5, 0.5, dtype=np.float32)
                labels = f["Y"][:]
                self.angles = np.zeros((labels.shape[0], k), dtype=np.float32)
                for row_idx, row in enumerate(labels):
                    idx = np.flatnonzero(row > 0.5)
                    if idx.size < k:
                        idx = np.argsort(row)[-k:]
                    self.angles[row_idx] = np.sort(angle_grid[idx[:k]])
            else:
                raise KeyError("HDF5 must contain 'angles' or 'Y'")

            self.angle_grid = f["angle_grid"][:] if "angle_grid" in f else None

        if self.angles.ndim == 1:
            self.angles = self.angles.reshape(-1, 1)
        if self.angles.shape[1] != k:
            self.angles = self.angles[:, :k]

        _, self.channels, height, self.N = self.x.shape
        if height != 2 * self.N:
            raise ValueError(f"Expected tau input [B,tau,2N,N], got {self.x.shape}")

        print(f"Loaded {len(self)} samples from {h5_path}")
        print(f"  X/tau_corr: shape {self.x.shape}, dtype {self.x.dtype}")
        print(f"  angles: shape {self.angles.shape}, range [{self.angles.min():.1f}, {self.angles.max():.1f}]")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.angles[idx]).float()


def train(data_path: str, output_path: str,
          epochs: int, batch_size: int, lr: float, weight_decay: float,
          val_split: float, patience: int, k: int, eps: float, seed: int) -> None:
    data_path = str(Path(data_path).resolve() if Path(data_path).is_absolute() else Path(__file__).resolve().parent / data_path)
    output_path = str(Path(output_path).resolve() if Path(output_path).is_absolute() else Path(__file__).resolve().parent / output_path)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DeepRootMusicDataset(data_path, k=k)
    n_sensors = dataset.N
    channels = dataset.channels

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())

    model = SubspaceNet(N=n_sensors, tau=channels, eps=eps, M=k, diff_method="root_music",
                        output_unit="rad", sort_output=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.2)

    best_val_loss = float("inf")
    stale = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for cov, ang_true in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False, unit="batch"):
            cov = cov.to(device, non_blocking=True)
            ang_true = torch.deg2rad(ang_true.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            ang_pred, _, _, _ = model(cov)
            loss = rmspe_loss(ang_true, ang_pred, period=np.pi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * cov.size(0)
            n_train += cov.size(0)

        avg_train_loss = total_train_loss / max(n_train, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for cov, ang_true in val_loader:
                cov = cov.to(device, non_blocking=True)
                ang_true = torch.deg2rad(ang_true.to(device, non_blocking=True))
                ang_pred, _, _, _ = model(cov)
                loss = rmspe_loss(ang_true, ang_pred, period=np.pi)
                total_val_loss += loss.item() * cov.size(0)
                n_val += cov.size(0)

        avg_val_loss = total_val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs} | Train RMSPE: {avg_train_loss:.6f} | Val RMSPE: {avg_val_loss:.6f} | Time: {time.perf_counter() - t0:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "SubspaceNet",
                "N": n_sensors,
                "tau": channels,
                "k": k,
                "M": k,
                "eps": eps,
                "diff_method": "root_music",
                "output_unit": "rad",
                "sort_output": False,
                "angle_grid": dataset.angle_grid,
                "args": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "val_split": val_split,
                    "patience": patience,
                    "scheduler": "StepLR",
                    "step_size": 80,
                    "gamma": 0.2,
                    "seed": seed,
                },
            }, output_path)
            print(f"  -> Best model saved (val RMSPE = {best_val_loss:.6f})")
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"Early stopping after {epoch} epochs. Best val RMSPE: {best_val_loss:.6f}")
                break

    out_dir = Path(output_path).resolve().parent
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train RMSPE")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val RMSPE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSPE (rad)")
    plt.title("SubspaceNet Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "deep_root_music_training_loss.png", dpi=200)
    plt.close()
    print(f"Training finished. Best validation RMSPE = {best_val_loss:.6f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Deep Root-MUSIC baseline")
    parser.add_argument("--data", type=str, default="Subspace_train_rho.h5")
    parser.add_argument("--output", type=str, default="subspace_model.pth")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-9)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(data_path=args.data,
          output_path=args.output,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          weight_decay=args.weight_decay,
          val_split=args.val_split,
          patience=args.patience,
          k=args.k,
          eps=args.eps,
          seed=args.seed)


if __name__ == "__main__":
    main()
