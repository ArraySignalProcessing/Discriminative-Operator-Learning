"""
B5 backend convergence test.

The experiment uses one learned operator Q per sample and compares:
  Q-root: Root-MUSIC backend on Q
  Q-grid(delta): grid-search backend on the same Q with grid step delta

The reported metric is the grid-to-root RMSE discrepancy, i.e. the
permutation-invariant RMSE between Q-grid(delta) and Q-root. It isolates
backend compatibility from DOA accuracy against ground truth.
"""

import argparse
import itertools
import os
from typing import Tuple

import h5py
import numpy as np
import torch

from Ours_train import DiscriminativeOperatorNet, normalize_covariance_array


DEFAULT_DATA_PATH = "../data/B5/B5_BackendCompat.h5"
DEFAULT_MODEL_PATH = "ours_model.pth"
DEFAULT_OUT_PATH = "../data/B5/B5_BackendConvergence.h5"

K = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_covariance_batch(r_sam: np.ndarray) -> np.ndarray:
    return normalize_covariance_array(r_sam)


def normalize_angle_batch(angles: np.ndarray, k: int = K) -> np.ndarray:
    if angles.ndim != 2:
        raise ValueError(f"Expected 2D angle array, got shape={angles.shape}")
    if angles.shape[0] == k:
        return np.sort(angles.astype(np.float32), axis=0).T
    if angles.shape[1] == k:
        return np.sort(angles.astype(np.float32), axis=1)
    raise ValueError(f"Cannot infer sample axis in angles array, shape={angles.shape}")


def best_permutation_diff(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if est.shape != ref.shape:
        raise ValueError(f"est shape {est.shape} does not match ref shape {ref.shape}")
    best_diff = None
    best_se = None
    for perm in itertools.permutations(range(est.shape[1])):
        diff = est[:, perm] - ref
        se = np.sum(diff ** 2, axis=1)
        if best_diff is None:
            best_diff = diff.copy()
            best_se = se
            continue
        mask = se < best_se
        best_diff[mask] = diff[mask]
        best_se[mask] = se[mask]
    return best_diff


def rmse_deg(est: np.ndarray, ref: np.ndarray) -> float:
    diff = best_permutation_diff(est, ref)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1) / est.shape[1])))


def export_q(model: DiscriminativeOperatorNet, cov: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, cov.shape[0], batch_size):
            end = min(start + batch_size, cov.shape[0])
            x = torch.from_numpy(cov[start:end]).float().to(DEVICE)
            chunks.append(model.make_Q(x).detach().cpu().numpy().astype(np.complex64))
    return np.concatenate(chunks, axis=0)


def root_music_from_q(Q: np.ndarray, k: int = K) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.asarray(Q, dtype=np.complex128)
    Q = 0.5 * (Q + Q.conj().T)
    m = Q.shape[0]
    coeffs = np.zeros(2 * m - 1, dtype=np.complex128)
    for lag in range(-(m - 1), m):
        coeffs[lag + (m - 1)] = np.diagonal(Q, offset=lag).sum()
    poly = coeffs[::-1]
    nonzero = np.flatnonzero(np.abs(poly) > 1e-12)
    if nonzero.size == 0:
        return np.zeros(k, dtype=np.float32), np.zeros(k, dtype=np.complex64)
    roots = np.roots(poly[nonzero[0] :])
    if roots.size == 0:
        return np.zeros(k, dtype=np.float32), np.zeros(k, dtype=np.complex64)
    inside = roots[np.abs(roots) <= 1.0]
    pool = inside if inside.size >= k else roots
    selected = pool[np.argsort(np.abs(1.0 - np.abs(pool)))[:k]]
    sin_theta = np.clip(np.real(np.angle(selected) / np.pi), -0.9999, 0.9999)
    doa = -np.rad2deg(np.arcsin(sin_theta)).astype(np.float32)
    if doa.size < k:
        doa = np.pad(doa, (0, k - doa.size), mode="constant")
        selected = np.pad(selected, (0, k - selected.size), mode="constant")
    return np.sort(doa[:k]).astype(np.float32), selected[:k].astype(np.complex64)


def steering_matrix(angle_grid: np.ndarray, m: int, d: float = 0.5) -> np.ndarray:
    theta = np.deg2rad(angle_grid).reshape(-1, 1)
    pos = (np.arange(m, dtype=np.float32) * d).reshape(1, -1)
    return np.exp(1j * 2.0 * np.pi * pos * np.sin(theta)).astype(np.complex64)


def spectrum_from_q(Q: np.ndarray, angle_grid: np.ndarray) -> np.ndarray:
    A = steering_matrix(angle_grid, Q.shape[-1])
    QA = np.einsum("nmk,gk->ngm", Q, A)
    q = np.einsum("gm,ngm->ng", A.conj(), QA).real / Q.shape[-1]
    return 1.0 / (np.maximum(q, 0.0) + 1e-6)


def pick_k_spectrum_peaks(
    power: np.ndarray,
    angle_grid: np.ndarray,
    k: int = K,
    min_separation_deg: float = 1.0,
) -> np.ndarray:
    values = power.astype(np.float64)
    candidates = []

    def far_enough(idx: int, selected_idx: list[int]) -> bool:
        if not selected_idx:
            return True
        return bool(np.all(np.abs(angle_grid[idx] - angle_grid[np.array(selected_idx)]) >= min_separation_deg))

    for idx in range(1, values.size - 1):
        if np.isfinite(values[idx]) and values[idx] > values[idx - 1] and values[idx] >= values[idx + 1]:
            candidates.append(idx)
    selected = []
    for idx in sorted(candidates, key=lambda idx: values[idx], reverse=True):
        if far_enough(idx, selected):
            selected.append(int(idx))
        if len(selected) == k:
            break
    if len(selected) < k:
        for idx in np.argsort(values[1:-1])[::-1] + 1:
            idx = int(idx)
            if np.isfinite(values[idx]) and idx not in selected and far_enough(idx, selected):
                selected.append(idx)
            if len(selected) == k:
                break
    if len(selected) < k:
        for idx in np.argsort(values)[::-1]:
            idx = int(idx)
            if np.isfinite(values[idx]) and idx not in selected and far_enough(idx, selected):
                selected.append(idx)
            if len(selected) == k:
                break
    return np.sort(-angle_grid[np.array(selected[:k], dtype=int)]).astype(np.float32)


def estimate_grid(
    Q: np.ndarray,
    grid_step: float,
    grid_batch_size: int,
    angle_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    angle_grid = np.arange(-60.0, 60.0 + 0.5 * grid_step, grid_step, dtype=np.float32)
    estimates = []
    for start in range(0, Q.shape[0], grid_batch_size):
        end = min(start + grid_batch_size, Q.shape[0])
        spectra_parts = []
        for angle_start in range(0, angle_grid.size, angle_batch_size):
            angle_end = min(angle_start + angle_batch_size, angle_grid.size)
            spectra_parts.append(spectrum_from_q(Q[start:end], angle_grid[angle_start:angle_end]))
        spectra = np.concatenate(spectra_parts, axis=1)
        estimates.extend(pick_k_spectrum_peaks(row, angle_grid) for row in spectra)
    est = np.vstack(estimates).astype(np.float32)
    return est, angle_grid


def step_tag(step: float) -> str:
    label = f"{step:.6f}".rstrip("0").rstrip(".")
    return f"step_{label}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grid_batch_size", type=int, default=16)
    parser.add_argument("--angle_batch_size", type=int, default=10000)
    parser.add_argument(
        "--grid_steps",
        type=float,
        nargs="+",
        default=[5.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.075, 0.05, 0.03, 0.02, 0.01],
    )
    args = parser.parse_args()

    with h5py.File(args.data, "r") as f:
        cov = normalize_covariance_batch(np.array(f["sam"]))
        angles = normalize_angle_batch(np.array(f["angles"]))

    model = DiscriminativeOperatorNet(m=cov.shape[-1], rank=args.rank).to(DEVICE)
    state = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    Q = export_q(model, cov, args.batch_size)

    root_est, roots = zip(*(root_music_from_q(q) for q in Q))
    root_est = np.vstack(root_est).astype(np.float32)
    roots = np.vstack(roots).astype(np.complex64)

    grid_steps = np.array(args.grid_steps, dtype=np.float32)
    g2r_rmse = []
    grid_gt_rmse = []
    grid_estimates = {}
    for step in grid_steps:
        grid_est, _ = estimate_grid(Q, float(step), args.grid_batch_size, args.angle_batch_size)
        grid_estimates[float(step)] = grid_est
        g2r = rmse_deg(grid_est, root_est)
        gt = rmse_deg(grid_est, angles)
        g2r_rmse.append(g2r)
        grid_gt_rmse.append(gt)
        print(f"grid step={step:g} deg: G2R RMSE={g2r:.4f} deg, GT RMSE={gt:.4f} deg")

    root_gt_rmse = rmse_deg(root_est, angles)
    print(f"Q-root GT RMSE={root_gt_rmse:.4f} deg")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as fout:
        fout.create_dataset("grid_steps", data=grid_steps)
        fout.create_dataset("G2R_RMSE", data=np.array(g2r_rmse, dtype=np.float32))
        fout.create_dataset("Q_grid_RMSE_gt", data=np.array(grid_gt_rmse, dtype=np.float32))
        fout.create_dataset("Q_root_RMSE_gt", data=np.array([root_gt_rmse], dtype=np.float32))
        fout.create_dataset("Q_root_DOA", data=root_est)
        fout.create_dataset("roots_real", data=roots.real.astype(np.float32))
        fout.create_dataset("roots_imag", data=roots.imag.astype(np.float32))
        group = fout.create_group("Q_grid_DOA")
        for step, est in grid_estimates.items():
            group.create_dataset(step_tag(step), data=est)
        fout.attrs["data"] = args.data
        fout.attrs["model"] = args.model

    print(f"Saved convergence results to {args.out}")


if __name__ == "__main__":
    main()
