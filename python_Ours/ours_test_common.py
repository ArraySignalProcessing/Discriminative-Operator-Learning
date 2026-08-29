"""Shared Ours and Root-Ours test backends for C1-C5 experiments."""

from __future__ import annotations

import argparse
import itertools
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


K_DEFAULT = 2
RESOLUTION_THRESHOLD = 1.0 # degrees, for PoR calculation
DEFAULT_ANGLE_GRID = np.arange(-60.0, 60.0 + 0.5, 0.5, dtype=np.float32)
EPS = 1e-8
SPEC_EPS = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TopicConfig:
    topic: str
    data_path: str
    ours_out_path: str
    root_out_path: str
    x_dataset: str
    x_label: str
    plot_suffix: str


TOPIC_CONFIGS = {
    "C4": TopicConfig(
        topic="C4",
        data_path="../data/C4/C4_SnrScan.h5",
        ours_out_path="../data/C4/C4_Ours.h5",
        root_out_path="../data/C4/C4_RootOurs.h5",
        x_dataset="SNR_vec",
        x_label="SNR",
        plot_suffix="C4",
    ),
    "C5": TopicConfig(
        topic="C5",
        data_path="../data/C5/C5_SnapshotScan.h5",
        ours_out_path="../data/C5/C5_Ours.h5",
        root_out_path="../data/C5/C5_RootOurs.h5",
        x_dataset="T_vec",
        x_label="T",
        plot_suffix="C5",
    ),
    "C1": TopicConfig(
        topic="C1",
        data_path="../data/C1/C1_CloseSource.h5",
        ours_out_path="../data/C1/C1_Ours.h5",
        root_out_path="../data/C1/C1_RootOurs.h5",
        x_dataset="delta_theta_vec",
        x_label="DeltaTheta",
        plot_suffix="C1",
    ),
    "C2": TopicConfig(
        topic="C2",
        data_path="../data/C2/C2_CoherentSources.h5",
        ours_out_path="../data/C2/C2_Ours.h5",
        root_out_path="../data/C2/C2_RootOurs.h5",
        x_dataset="rho_vec",
        x_label="rho",
        plot_suffix="C2",
    ),
    "C3": TopicConfig(
        topic="C3",
        data_path="../data/C3/C3_ArrayMismatch.h5",
        ours_out_path="../data/C3/C3_Ours.h5",
        root_out_path="../data/C3/C3_RootOurs.h5",
        x_dataset="rho_all",
        x_label="rho",
        plot_suffix="C3",
    ),
}


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
            return covariance.reshape(-1, covariance.shape[2], covariance.shape[3], covariance.shape[4]).astype(np.float32)
        if covariance.shape[-1] == 2:
            reordered = covariance.transpose(0, 1, 4, 2, 3)
            return reordered.reshape(-1, reordered.shape[2], reordered.shape[3], reordered.shape[4]).astype(np.float32)

    raise ValueError(f"Cannot normalize covariance layout, shape={covariance.shape}")


def labels_to_angles(Y: np.ndarray, angle_grid: np.ndarray, k: int) -> np.ndarray:
    angles = np.zeros((Y.shape[0], k), dtype=np.float32)
    for i in range(Y.shape[0]):
        idx = np.flatnonzero(Y[i] > 0.5)
        if idx.size < k:
            idx = np.argsort(Y[i])[-k:]
        idx = idx[:k]
        angles[i] = np.sort(angle_grid[idx])
    return angles


def normalize_angles_array(angles: np.ndarray, k: int) -> np.ndarray:
    if angles.ndim != 2:
        raise ValueError(f"Expected angles to be 2D, got shape={angles.shape}")
    if angles.shape[1] == k:
        return np.sort(angles.astype(np.float32), axis=1)
    if angles.shape[0] == k:
        return np.sort(angles.astype(np.float32), axis=0).T
    raise ValueError(f"Cannot infer angles layout, shape={angles.shape}, k={k}")


def load_gt_angles(h5_group: h5py.Group, angle_grid: np.ndarray, k: int) -> np.ndarray:
    if "angles" in h5_group:
        return normalize_angles_array(h5_group["angles"][:], k)
    if "Y" in h5_group:
        return labels_to_angles(h5_group["Y"][:], angle_grid, k)
    raise KeyError("Each test group must contain angles or Y labels")


def best_permutation_match(est: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return estimates and errors after minimum-SSE source permutation matching."""
    if est.shape != gt.shape:
        raise ValueError(f"est shape {est.shape} does not match gt shape {gt.shape}")

    best_est = None
    best_diff = None
    best_score = None
    for perm in itertools.permutations(range(est.shape[1])):
        est_perm = est[:, perm]
        diff = est_perm - gt
        finite = np.isfinite(diff)
        score = np.sum(np.where(finite, diff ** 2, 1e12), axis=1)
        if best_diff is None:
            best_est = est_perm.copy()
            best_diff = diff.copy()
            best_score = score
            continue
        mask = score < best_score
        best_est[mask] = est_perm[mask]
        best_diff[mask] = diff[mask]
        best_score[mask] = score[mask]
    return best_est.astype(np.float32), best_diff.astype(np.float32)


def parse_group_value(group_name: str) -> float:
    if group_name.startswith("SNR_"):
        match = re.search(r"SNR_(?:min)?(\d+)dB", group_name)
        if match:
            value = float(match.group(1))
            return -value if "min" in group_name else value
    if group_name.startswith("T_"):
        return float(group_name.split("_", 1)[1])
    if group_name.startswith("Sep_"):
        match = re.search(r"Sep_([0-9.]+)deg", group_name)
        if match:
            return float(match.group(1))
    if group_name.startswith("Rho_"):
        tag = group_name.split("_", 1)[1].replace("p", ".")
        return float(tag)
    return float("nan")


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
    def __init__(self, m: int = 10, rank: int = 8, output_width: int | None = None):
        super().__init__()
        self.m = m
        self.output_width = int(output_width if output_width is not None else rank)
        self.rank = self.output_width
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
        self.out = nn.Conv2d(256, 2 * self.output_width, kernel_size=1)

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


def load_model_from_checkpoint(model_path: str, fallback_rank: int) -> Tuple[DiscriminativeOperatorNet, np.ndarray, int, int, dict]:
    state = torch.load(model_path, map_location=DEVICE)
    metadata = {}
    if isinstance(state, dict) and "model_state_dict" in state:
        metadata = state
        m = int(state.get("m", 10))
        output_width = int(state.get("output_width", state.get("rank", fallback_rank)))
        angle_grid = np.asarray(state.get("angle_grid", DEFAULT_ANGLE_GRID), dtype=np.float32)
        model = DiscriminativeOperatorNet(m=m, output_width=output_width).to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
    else:
        m = 10
        output_width = int(fallback_rank)
        angle_grid = DEFAULT_ANGLE_GRID.copy()
        model = DiscriminativeOperatorNet(m=m, output_width=output_width).to(DEVICE)
        model.load_state_dict(state)
    model.eval()
    return model, angle_grid, m, output_width, metadata


def steering_matrix_np(angle_grid: np.ndarray, m: int, d: float = 0.5) -> np.ndarray:
    theta = np.deg2rad(angle_grid.astype(np.float32)).reshape(-1, 1)
    pos = (np.arange(m, dtype=np.float32) * d).reshape(1, -1)
    phase = 2.0 * np.pi * pos * np.sin(theta)
    return np.exp(1j * phase).astype(np.complex64)


def response_from_Q(Q: torch.Tensor, A_grid: torch.Tensor) -> torch.Tensor:
    m = Q.shape[-1]
    Qa = torch.einsum("bmn,gn->bgm", Q, A_grid)
    q = torch.einsum("gm,bgm->bg", A_grid.conj(), Qa).real / m
    return torch.clamp(q, min=0.0)


def _plateau_aware_local_maxima(values: np.ndarray) -> list[int]:
    n = values.size
    if n == 0:
        return []
    if n == 1:
        return [0] if np.isfinite(values[0]) else []
    candidates = []
    if np.isfinite(values[0]) and values[0] > values[1]:
        candidates.append(0)
    i = 1
    while i < n - 1:
        if not np.isfinite(values[i]):
            i += 1
            continue
        left = i
        right = i
        while right + 1 < n and np.isfinite(values[right + 1]) and values[right + 1] == values[i]:
            right += 1
        left_val = values[left - 1] if left > 0 else -np.inf
        right_val = values[right + 1] if right + 1 < n else -np.inf
        if values[i] >= left_val and values[i] >= right_val and (values[i] > left_val or values[i] > right_val):
            candidates.append((left + right) // 2)
        i = right + 1
    if np.isfinite(values[-1]) and values[-1] > values[-2]:
        candidates.append(n - 1)
    return sorted(set(candidates))


def pick_k_spatial_spectrum_peaks(spectrum: np.ndarray, angle_grid: np.ndarray, k: int) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.ndim != 1:
        raise ValueError(f"Expected 1D spatial spectrum, got shape={spectrum.shape}")
    if spectrum.size != angle_grid.size:
        raise ValueError(f"Spectrum length {spectrum.size} != angle grid length {angle_grid.size}")
    values = spectrum.copy()
    values[~np.isfinite(values)] = -np.inf
    candidates = _plateau_aware_local_maxima(values)
    ranked = sorted(candidates, key=lambda i: values[i], reverse=True)
    selected = ranked[:k]
    if len(selected) < k:
        for i in np.argsort(values)[::-1]:
            i = int(i)
            if np.isfinite(values[i]) and i not in selected:
                selected.append(i)
            if len(selected) == k:
                break
    if len(selected) < k:
        raise RuntimeError(f"Only found {len(selected)} finite grid points, but K={k}.")
    return np.sort(angle_grid[np.asarray(selected, dtype=int)]).astype(np.float32)


def estimate_grid_batch(model: DiscriminativeOperatorNet,
                        R_batch: np.ndarray,
                        angle_grid: np.ndarray,
                        batch_size: int,
                        k: int,
                        d: float,
                        save_debug_spectrum: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    m = R_batch.shape[-1]
    A_t = torch.from_numpy(steering_matrix_np(angle_grid, m=m, d=d)).to(DEVICE)
    estimates = []
    p_all = [] if save_debug_spectrum else None
    with torch.no_grad():
        for start in range(0, R_batch.shape[0], batch_size):
            end = min(start + batch_size, R_batch.shape[0])
            x = torch.from_numpy(R_batch[start:end]).float().to(DEVICE)
            Q = model.make_Q(x)
            q = response_from_Q(Q, A_t)
            spectrum = (1.0 / (q + SPEC_EPS)).detach().cpu().numpy().astype(np.float32)
            for row_p in spectrum:
                estimates.append(pick_k_spatial_spectrum_peaks(row_p, angle_grid, k=k))
            if save_debug_spectrum:
                p_all.append(spectrum)
    return np.vstack(estimates).astype(np.float32), np.vstack(p_all).astype(np.float32) if save_debug_spectrum else None


def polynomial_coefficients_from_Q(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.complex128)
    Q = 0.5 * (Q + Q.conj().T)
    m = Q.shape[0]
    coeffs_lag = np.zeros(2 * m - 1, dtype=np.complex128)
    for lag in range(-(m - 1), m):
        coeffs_lag[lag + (m - 1)] = np.diagonal(Q, offset=lag).sum()
    poly_desc = coeffs_lag[::-1]
    nz = np.flatnonzero(np.abs(poly_desc) > 1e-12)
    return poly_desc if nz.size == 0 else poly_desc[nz[0]:]


def root_music_from_Q(Q: np.ndarray,
                      k: int,
                      d: float = 0.5,
                      debug: bool = False) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, dict]:
    def pack(doas, selected, roots, finite_roots, inside, ranked):
        if not debug:
            return doas, selected
        return doas, {
            "all_roots": roots,
            "finite_roots": finite_roots,
            "inside_roots": inside,
            "ranked_inside_roots": ranked,
            "selected_roots": selected,
        }

    empty = np.array([], dtype=np.complex128)
    nan_doa = np.full(k, np.nan, dtype=np.float32)
    poly = polynomial_coefficients_from_Q(Q)
    if poly.size == 0 or np.all(np.abs(poly) <= 1e-12):
        return pack(nan_doa, empty, empty, empty, empty, empty)
    roots = np.roots(poly)
    if roots.size == 0:
        return pack(nan_doa, roots, roots, empty, empty, empty)
    finite_roots = roots[np.isfinite(roots.real) & np.isfinite(roots.imag)]
    if finite_roots.size == 0:
        return pack(nan_doa, empty, roots, finite_roots, empty, empty)
    inside = finite_roots[np.abs(finite_roots) < 1.0]
    if inside.size == 0:
        return pack(nan_doa, inside, roots, finite_roots, inside, empty)
    ranked = inside[np.argsort(np.abs(np.abs(inside) - 1.0))]
    phase = np.angle(ranked)
    sin_theta = phase / (2.0 * np.pi * d)
    valid = np.isfinite(sin_theta) & (np.abs(np.real(sin_theta)) <= 1.0)
    selected = ranked[valid][:k]
    if selected.size == 0:
        return pack(nan_doa, selected, roots, finite_roots, inside, ranked)
    phase = np.angle(selected)
    sin_theta = np.real(phase / (2.0 * np.pi * d))
    doas = np.rad2deg(np.arcsin(sin_theta)).astype(np.float32)
    if doas.size < k:
        doas = np.pad(doas, (0, k - doas.size), constant_values=np.nan)
    return pack(np.sort(doas[:k]).astype(np.float32), selected, roots, finite_roots, inside, ranked)


def estimate_root_batch(model: DiscriminativeOperatorNet,
                        R_batch: np.ndarray,
                        batch_size: int,
                        k: int,
                        d: float,
                        save_debug_roots: bool = False) -> Tuple[np.ndarray, list | None]:
    model.eval()
    estimates = []
    debug_roots = [] if save_debug_roots else None
    with torch.no_grad():
        for start in range(0, R_batch.shape[0], batch_size):
            end = min(start + batch_size, R_batch.shape[0])
            x = torch.from_numpy(R_batch[start:end]).float().to(DEVICE)
            Q_batch = model.make_Q(x).detach().cpu().numpy()
            for Q in Q_batch:
                est, roots = root_music_from_Q(Q, k=k, d=d, debug=save_debug_roots)
                estimates.append(est)
                if save_debug_roots:
                    debug_roots.append(roots)
    return np.vstack(estimates).astype(np.float32), debug_roots


def print_checkpoint_metadata(metadata: dict) -> None:
    args = metadata.get("args", {}) if isinstance(metadata, dict) else {}
    if not args:
        return
    print("\nCheckpoint training metadata:")
    for key in ["data", "epochs", "batch_size", "lr", "output_width", "margin", "lambda_margin", "lambda_trace", "bg_step", "true_tol"]:
        if key in args:
            print(f"  {key}: {args[key]}")


def plot_predictions(topic: str, x_label: str, x_list, gt_list, est_list, method: str) -> None:
    plot_dir = os.path.join(os.path.dirname(__file__), "Plot_result")
    os.makedirs(plot_dir, exist_ok=True)
    n = len(x_list)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.asarray(axes).reshape(-1) if n > 1 else np.array([axes])
    for i, (x_value, gt, est) in enumerate(zip(x_list, gt_list, est_list)):
        ax = axes[i]
        sample_idx = np.arange(1, gt.shape[0] + 1)
        ax.hlines(gt[0, 0], sample_idx[0], sample_idx[-1], linestyles="--", linewidth=2, label="GT src 1")
        ax.hlines(gt[0, 1], sample_idx[0], sample_idx[-1], linestyles="--", linewidth=2, label="GT src 2")
        ax.scatter(sample_idx, est[:, 0], s=18, alpha=0.75, marker="o", label="Est src 1")
        ax.scatter(sample_idx, est[:, 1], s=18, alpha=0.75, marker="x", label="Est src 2")
        ax.set_title(f"{x_label}={x_value:g}")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.text(0.5, 0.02, "Sample", ha="center")
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{topic}_{method}_prediction.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction plot saved to {save_path}")


def plot_debug_spatial_spectrum(angle_grid: np.ndarray,
                                spectrum: np.ndarray,
                                gt_angles: np.ndarray,
                                est_angles: np.ndarray,
                                topic: str,
                                group_name: str,
                                sample_index: int) -> None:
    plot_dir = os.path.join(os.path.dirname(__file__), "Plot_result")
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(angle_grid, spectrum, linewidth=1.4, label="Spatial spectrum P(theta), linear")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("P(theta), linear")
    ax.grid(True, alpha=0.25)
    for a in gt_angles:
        ax.axvline(float(a), linestyle="--", linewidth=1.2, label="GT" if a == gt_angles[0] else None)
    for a in est_angles:
        ax.axvline(float(a), linestyle=":", linewidth=1.4, label="Est" if a == est_angles[0] else None)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{topic} debug spatial spectrum: {group_name}, sample {sample_index}")
    fig.tight_layout()
    safe_group = group_name.replace("/", "_")
    save_path = os.path.join(plot_dir, f"debug_spatial_spectrum_{topic}_{safe_group}_sample{sample_index}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Debug spatial spectrum plot saved to {save_path}")


def plot_debug_roots(root_info: dict,
                     gt_angles: np.ndarray,
                     est_angles: np.ndarray,
                     topic: str,
                     group_name: str,
                     sample_index: int,
                     d: float) -> None:
    plot_dir = os.path.join(os.path.dirname(__file__), "Plot_result")
    os.makedirs(plot_dir, exist_ok=True)
    all_roots = np.asarray(root_info.get("finite_roots", []), dtype=np.complex128)
    inside_roots = np.asarray(root_info.get("inside_roots", []), dtype=np.complex128)
    selected_roots = np.asarray(root_info.get("selected_roots", []), dtype=np.complex128)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    phi = np.linspace(0.0, 2.0 * np.pi, 512)
    ax.plot(np.cos(phi), np.sin(phi), "k--", linewidth=1.2, label="Unit circle")
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    if all_roots.size:
        ax.scatter(all_roots.real, all_roots.imag, s=28, c="0.65", label="All finite roots")
    if inside_roots.size:
        ax.scatter(inside_roots.real, inside_roots.imag, s=42, facecolors="none", edgecolors="tab:blue", label="Inside roots")
    if selected_roots.size:
        ax.scatter(selected_roots.real, selected_roots.imag, s=82, marker="x", c="tab:red", linewidths=2.0, label="Selected roots")
    for root in selected_roots:
        phase = np.angle(root)
        sin_theta = phase / (2.0 * np.pi * d)
        if np.isfinite(sin_theta) and abs(float(np.real(sin_theta))) <= 1.0:
            theta = np.rad2deg(np.arcsin(np.real(sin_theta)))
            ax.annotate(f"{theta:.2f} deg", (root.real, root.imag), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{topic} Root-Ours roots: {group_name}, sample {sample_index}\nGT={gt_angles}, Est={est_angles}")
    fig.tight_layout()
    safe_group = group_name.replace("/", "_")
    save_path = os.path.join(plot_dir, f"debug_roots_unit_circle_{topic}_{safe_group}_sample{sample_index}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Debug root plot saved to {save_path}")


def sorted_groups(h5_file: h5py.File) -> list[str]:
    groups = [g for g in h5_file.keys() if not g.startswith(".")]
    return sorted(groups, key=parse_group_value)


def run_grid_test(topic_key: str) -> None:
    config = TOPIC_CONFIGS[topic_key]
    parser = argparse.ArgumentParser(description=f"Test Ours on Topic {topic_key}")
    parser.add_argument("--data", default=config.data_path)
    parser.add_argument("--model", default="ours_model.pth")
    parser.add_argument("--out", default=config.ours_out_path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d", type=float, default=0.5)
    parser.add_argument("--debug_group", default="")
    parser.add_argument("--debug_sample", type=int, default=0)
    parser.add_argument("--save_all_debug_spectra", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading Ours model from {args.model}...")
    model, angle_grid, m, rank, metadata = load_model_from_checkpoint(args.model, args.rank)
    print(f"Model loaded. m={m}, rank={rank}, grid=[{float(angle_grid[0])}, {float(angle_grid[-1])}], G={angle_grid.size}")
    print_checkpoint_metadata(metadata)

    print(f"\nOpening test data: {args.data}")
    x_vals, rmse_vals, por_vals = [], [], []
    plot_x, plot_gts, plot_ests = [], [], []
    debug_payload = {}

    with h5py.File(args.data, "r") as f:
        groups = sorted_groups(f)
        print(f"Found {len(groups)} groups: {groups}\n")
        if args.debug_group and args.debug_group not in groups:
            raise KeyError(f"--debug_group {args.debug_group} not found. Available groups: {groups}")

        for grp in groups:
            x_value = parse_group_value(grp)
            R_batch = normalize_covariance_array(f[grp]["sam"][:])
            gt = load_gt_angles(f[grp], angle_grid, args.k)
            if R_batch.shape[0] != gt.shape[0]:
                raise ValueError(f"Group {grp}: R samples {R_batch.shape[0]} != gt samples {gt.shape[0]}")
            if R_batch.shape[-1] != m:
                raise ValueError(f"Group {grp}: covariance M={R_batch.shape[-1]} does not match model m={m}")

            need_debug = args.save_all_debug_spectra or (args.debug_group == grp)
            print(f"{config.x_label}={x_value:g} | estimating...", flush=True)
            est, p_spectrum = estimate_grid_batch(model, R_batch, angle_grid, args.batch_size, args.k, args.d, need_debug)
            est = -est
            est_matched, diff = best_permutation_match(est, gt)
            rmse = float(np.sqrt(np.nanmean(np.sum(diff ** 2, axis=1) / args.k)))
            por = float(np.nanmean(np.max(np.abs(diff), axis=1) <= RESOLUTION_THRESHOLD) * 100.0)

            x_vals.append(x_value)
            rmse_vals.append(rmse)
            por_vals.append(por)
            plot_x.append(x_value)
            plot_gts.append(gt)
            plot_ests.append(est_matched)
            gt_str = np.array2string(gt[0], precision=2, separator=", ", suppress_small=False)
            pred_str = np.array2string(est_matched[0], precision=2, separator=", ", suppress_small=False)
            print(f"{config.x_label}={x_value:g} | GT={gt_str} | Pred={pred_str} | RMSE={rmse:.3f} deg | PoR={por:5.1f}%", flush=True)

            if need_debug and p_spectrum is not None:
                debug_payload[grp] = {"p_spectrum": p_spectrum, "gt": gt, "est": est_matched}

    order = np.argsort(x_vals)
    x_sorted = np.asarray(x_vals, dtype=np.float32)[order]
    rmse_sorted = np.asarray(rmse_vals, dtype=np.float32)[order]
    por_sorted = np.asarray(por_vals, dtype=np.float32)[order] / 100.0

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with h5py.File(args.out, "w") as fout:
        fout.create_dataset("Ours_RMSE", data=rmse_sorted)
        fout.create_dataset("Ours_PoR", data=por_sorted)
        fout.create_dataset(config.x_dataset, data=x_sorted)
        fout.create_dataset("angle_grid", data=angle_grid)
        fout.attrs["estimation_rule"] = "Q-grid MUSIC backend: P(theta)=1/(a^H Q a / M + eps), sign flip, minimum-error source matching"
        if debug_payload:
            dbg_root = fout.create_group("debug")
            for grp, payload in debug_payload.items():
                g = dbg_root.create_group(grp)
                if args.save_all_debug_spectra:
                    g.create_dataset("p_spectrum", data=payload["p_spectrum"])
                else:
                    idx = int(args.debug_sample)
                    g.create_dataset("p_spectrum", data=payload["p_spectrum"][idx])
                    g.create_dataset("gt_angles", data=payload["gt"][idx])
                    g.create_dataset("est_angles", data=payload["est"][idx])
                    plot_debug_spatial_spectrum(angle_grid, payload["p_spectrum"][idx], payload["gt"][idx], payload["est"][idx], config.topic, grp, idx)

    plot_predictions(config.plot_suffix, config.x_label, plot_x, plot_gts, plot_ests, "ours")
    print(f"\nResults saved to {args.out}")
    print("Done.")


def run_root_test(topic_key: str) -> None:
    config = TOPIC_CONFIGS[topic_key]
    parser = argparse.ArgumentParser(description=f"Test Root-Ours on Topic {topic_key}")
    parser.add_argument("--data", default=config.data_path)
    parser.add_argument("--model", default="ours_model.pth")
    parser.add_argument("--out", default=config.root_out_path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d", type=float, default=0.5)
    parser.add_argument("--debug_group", default="")
    parser.add_argument("--debug_sample", type=int, default=0)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading Root-Ours model from {args.model}...")
    model, angle_grid, m, rank, metadata = load_model_from_checkpoint(args.model, args.rank)
    print(f"Model loaded. m={m}, rank={rank}, grid=[{float(angle_grid[0])}, {float(angle_grid[-1])}], G={angle_grid.size}")
    print_checkpoint_metadata(metadata)

    print(f"\nOpening test data: {args.data}")
    x_vals, rmse_vals, por_vals = [], [], []
    plot_x, plot_gts, plot_ests = [], [], []
    debug_root_info_to_save = None
    debug_est_to_save = None
    debug_gt_to_save = None

    with h5py.File(args.data, "r") as f:
        groups = sorted_groups(f)
        print(f"Found {len(groups)} groups: {groups}\n")
        if args.debug_group and args.debug_group not in groups:
            raise KeyError(f"--debug_group {args.debug_group} not found. Available groups: {groups}")

        for grp in groups:
            x_value = parse_group_value(grp)
            R_batch = normalize_covariance_array(f[grp]["sam"][:])
            gt = load_gt_angles(f[grp], angle_grid, args.k)
            if R_batch.shape[0] != gt.shape[0]:
                raise ValueError(f"Group {grp}: R samples {R_batch.shape[0]} != gt samples {gt.shape[0]}")
            if R_batch.shape[-1] != m:
                raise ValueError(f"Group {grp}: covariance M={R_batch.shape[-1]} does not match model m={m}")

            need_debug = args.debug_group == grp
            print(f"{config.x_label}={x_value:g} | estimating...", flush=True)
            est, roots_debug = estimate_root_batch(model, R_batch, args.batch_size, args.k, args.d, need_debug)
            est = -est
            est_matched, diff = best_permutation_match(est, gt)
            rmse = float(np.sqrt(np.nanmean(np.sum(diff ** 2, axis=1) / args.k)))
            por = float(np.nanmean(np.max(np.abs(diff), axis=1) <= RESOLUTION_THRESHOLD) * 100.0)

            x_vals.append(x_value)
            rmse_vals.append(rmse)
            por_vals.append(por)
            plot_x.append(x_value)
            plot_gts.append(gt)
            plot_ests.append(est_matched)
            gt_str = np.array2string(gt[0], precision=2, separator=", ", suppress_small=False)
            pred_str = np.array2string(est_matched[0], precision=2, separator=", ", suppress_small=False)
            print(f"{config.x_label}={x_value:g} | GT={gt_str} | Pred={pred_str} | RMSE={rmse:.3f} deg | PoR={por:5.1f}%", flush=True)

            if need_debug:
                idx = int(args.debug_sample)
                debug_gt_to_save = gt[idx]
                debug_est_to_save = est_matched[idx]
                debug_root_info_to_save = roots_debug[idx] if roots_debug is not None else {
                    "all_roots": np.array([], dtype=np.complex128),
                    "finite_roots": np.array([], dtype=np.complex128),
                    "inside_roots": np.array([], dtype=np.complex128),
                    "ranked_inside_roots": np.array([], dtype=np.complex128),
                    "selected_roots": np.array([], dtype=np.complex128),
                }
                print("\nRoot-Ours debug sample:")
                print(f"  group: {grp}")
                print(f"  sample: {idx}")
                print(f"  GT angles: {debug_gt_to_save}")
                print(f"  estimated angles: {debug_est_to_save}")
                print("  selected roots nearest unit circle:")
                for r in debug_root_info_to_save["selected_roots"]:
                    phase = np.angle(r)
                    sin_theta = phase / (2.0 * np.pi * args.d)
                    if np.isfinite(sin_theta) and abs(float(np.real(sin_theta))) <= 1.0:
                        theta_text = f"{np.rad2deg(np.arcsin(np.real(sin_theta))):.6f}"
                    else:
                        theta_text = "invalid"
                    print(f"    root={r.real:+.8e}{r.imag:+.8e}j, |z|={abs(r):.8e}, angle={phase:.8e}, doa={theta_text}")
                print(f"  finite roots: {debug_root_info_to_save['finite_roots'].size}, inside roots: {debug_root_info_to_save['inside_roots'].size}\n")

    order = np.argsort(x_vals)
    x_sorted = np.asarray(x_vals, dtype=np.float32)[order]
    rmse_sorted = np.asarray(rmse_vals, dtype=np.float32)[order]
    por_sorted = np.asarray(por_vals, dtype=np.float32)[order] / 100.0

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with h5py.File(args.out, "w") as fout:
        fout.create_dataset("RootOurs_RMSE", data=rmse_sorted)
        fout.create_dataset("RootOurs_PoR", data=por_sorted)
        fout.create_dataset(config.x_dataset, data=x_sorted)
        fout.create_dataset("angle_grid", data=angle_grid)
        fout.attrs["estimation_rule"] = "Root-MUSIC polynomial backend on learned Q, inside-unit-circle root selection, sign flip, minimum-error source matching"
        if debug_root_info_to_save is not None:
            g = fout.create_group("debug")
            g.create_dataset("gt_angles", data=debug_gt_to_save)
            g.create_dataset("est_angles", data=debug_est_to_save)
            for name, roots in debug_root_info_to_save.items():
                roots = np.asarray(roots, dtype=np.complex128)
                g.create_dataset(f"{name}_real", data=np.real(roots))
                g.create_dataset(f"{name}_imag", data=np.imag(roots))
            g.attrs["debug_group"] = args.debug_group
            g.attrs["debug_sample"] = args.debug_sample

    plot_predictions(config.plot_suffix, config.x_label, plot_x, plot_gts, plot_ests, "root_ours")
    if debug_root_info_to_save is not None:
        plot_debug_roots(debug_root_info_to_save, debug_gt_to_save, debug_est_to_save, config.topic, args.debug_group, int(args.debug_sample), args.d)
    print(f"\nResults saved to {args.out}")
    print("Done.")
