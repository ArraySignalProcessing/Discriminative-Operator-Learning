"""
Ours_Generate_train_K3_coherent.py

Generate K=3 coherent-source training data for the proposed Ours model.

This script is intentionally narrow for the B6 source-count-generalization
experiment:
    - K is fixed to 3 by default.
    - DOAs are sampled without replacement from the fixed grid.
    - No minimum angular-separation constraint is imposed.
    - The stored covariance is the ideal/theoretical covariance by default:
          R = A Rs A^H + sigma^2 I
    - Source coherence is modeled with an equicorrelated source covariance:
          Rs = (1-rho) I + rho 11^T

Output format is compatible with Ours_train.py:
    X:          (N, 2, M, M), real/imaginary covariance channels
    Y:          (N, grid_size), multi-hot DOA labels
    angles:     (N, K), true DOAs in degrees
    angle_grid: (grid_size,)
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def steering_matrix(angles_deg: np.ndarray, array_pos: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angles_deg).reshape(1, -1)
    pos = array_pos.reshape(-1, 1)
    return np.exp(1j * 2.0 * np.pi * pos * np.sin(theta)).astype(np.complex64)


def make_source_cov(k: int, coherent: bool, rho: float) -> np.ndarray:
    if coherent and k > 1:
        eye = np.eye(k, dtype=np.complex64)
        ones = np.ones((k, k), dtype=np.complex64)
        return ((1.0 - rho) * eye + rho * ones).astype(np.complex64)
    return np.eye(k, dtype=np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate K=3 coherent ideal-covariance training data for Ours."
    )
    parser.add_argument("--num_samples", type=int, default=300000)
    parser.add_argument("--out", type=str, default="Ours_train_K3_coherent.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=float, default=0.5)
    parser.add_argument("--angle_min", type=float, default=-60.0)
    parser.add_argument("--angle_max", type=float, default=60.0)
    parser.add_argument("--angle_step", type=float, default=0.5)
    parser.add_argument("--snr_min", type=float, default=-15.0)
    parser.add_argument("--snr_max", type=float, default=10.0)
    parser.add_argument("--coherent_prob", type=float, default=0.6)
    parser.add_argument("--coherent_rho_min", type=float, default=0.3)
    parser.add_argument("--coherent_rho_max", type=float, default=0.98)
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be positive")
    if not (0.0 <= args.coherent_prob <= 1.0):
        raise ValueError("--coherent_prob must be in [0, 1]")
    if args.coherent_rho_min < 0.0 or args.coherent_rho_max >= 1.0:
        raise ValueError("coherent rho must satisfy 0 <= rho_min <= rho_max < 1")
    if args.coherent_rho_min > args.coherent_rho_max:
        raise ValueError("--coherent_rho_min cannot exceed --coherent_rho_max")

    rng = np.random.default_rng(args.seed)
    angle_grid = np.arange(
        args.angle_min,
        args.angle_max + 0.5 * args.angle_step,
        args.angle_step,
        dtype=np.float32,
    )
    n_grid = angle_grid.size
    if args.k > n_grid:
        raise ValueError(f"--k={args.k} exceeds grid size {n_grid}")

    array_pos = np.arange(args.m, dtype=np.float32) * args.d

    X = np.zeros((args.num_samples, 2, args.m, args.m), dtype=np.float32)
    Y = np.zeros((args.num_samples, n_grid), dtype=np.float32)
    angles = np.zeros((args.num_samples, args.k), dtype=np.float32)
    snrs = np.zeros(args.num_samples, dtype=np.float32)
    rhos = np.zeros(args.num_samples, dtype=np.float32)
    coherent_flags = np.zeros(args.num_samples, dtype=np.uint8)

    eye_m = np.eye(args.m, dtype=np.complex64)

    for i in range(args.num_samples):
        if i % 5000 == 0:
            print(f"Generating {i}/{args.num_samples}...", flush=True)

        idx = rng.choice(n_grid, size=args.k, replace=False)
        doa = np.sort(angle_grid[idx]).astype(np.float32)
        A = steering_matrix(doa, array_pos)

        snr_db = float(rng.uniform(args.snr_min, args.snr_max))
        sigma2 = float(10.0 ** (-snr_db / 10.0))
        coherent = bool(rng.random() < args.coherent_prob)
        rho = float(rng.uniform(args.coherent_rho_min, args.coherent_rho_max)) if coherent else 0.0
        Rs = make_source_cov(args.k, coherent, rho)

        R = (A @ Rs @ A.conj().T + sigma2 * eye_m).astype(np.complex64)

        X[i, 0] = R.real.astype(np.float32)
        X[i, 1] = R.imag.astype(np.float32)
        Y[i, idx] = 1.0
        angles[i] = doa
        snrs[i] = snr_db
        rhos[i] = rho
        coherent_flags[i] = 1 if coherent else 0

    perm = rng.permutation(args.num_samples)
    X, Y, angles = X[perm], Y[perm], angles[perm]
    snrs, rhos, coherent_flags = snrs[perm], rhos[perm], coherent_flags[perm]

    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("angles", data=angles)
        f.create_dataset("angle_grid", data=angle_grid)
        f.create_dataset("snr_db", data=snrs)
        f.create_dataset("rho", data=rhos)
        f.create_dataset("coherent_flag", data=coherent_flags)
        f.attrs["description"] = "K=3 coherent ideal-covariance training data for Ours"
        f.attrs["covariance_type"] = "ideal_theoretical"
        f.attrs["source_covariance_model"] = "(1-rho)I + rho*11H for coherent samples"
        f.attrs["doa_sampling"] = "grid sampling without replacement; no minimum separation"
        f.attrs["m"] = args.m
        f.attrs["k"] = args.k
        f.attrs["d"] = args.d
        f.attrs["angle_min"] = args.angle_min
        f.attrs["angle_max"] = args.angle_max
        f.attrs["angle_step"] = args.angle_step
        f.attrs["snr_min"] = args.snr_min
        f.attrs["snr_max"] = args.snr_max
        f.attrs["coherent_prob"] = args.coherent_prob
        f.attrs["coherent_rho_min"] = args.coherent_rho_min
        f.attrs["coherent_rho_max"] = args.coherent_rho_max
        f.attrs["seed"] = args.seed

    print(f"Training data saved to {args.out}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, angles shape: {angles.shape}")
    print("covariance_type=ideal_theoretical")
    print("doa_sampling=without replacement, no minimum separation")


if __name__ == "__main__":
    main()
