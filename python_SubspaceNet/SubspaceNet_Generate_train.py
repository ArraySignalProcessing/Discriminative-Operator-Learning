#!/usr/bin/env python3
"""Generate paper-aligned SubspaceNet training data.

The main SubspaceNet input follows Shmuel et al.'s multi-lag empirical
auto-correlation feature:
    tau_corr:   (N, tau_max+1, 2*M, M)
    sam:        (N, 2, M, M), empirical covariance real/imag channels
    Y:          (N, grid_size), nearest-grid multi-hot DOA labels
    angles:     (N, K), continuous true DoAs in degrees
    angle_grid: (grid_size,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def steering_matrix(angles_deg: np.ndarray, array_pos: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angles_deg).reshape(1, -1)
    pos = array_pos.reshape(-1, 1)
    return np.exp(1j * 2.0 * np.pi * pos * np.sin(theta)).astype(np.complex64)


def complex_gaussian(shape, rng: np.random.Generator) -> np.ndarray:
    return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)).astype(np.complex64)


def make_source_cov(k: int, coherent: bool, rho: float) -> np.ndarray:
    if coherent and k == 2:
        return np.array([[1.0, rho], [rho, 1.0]], dtype=np.complex64)
    return np.eye(k, dtype=np.complex64)


def covariance_channels(y_snapshots: np.ndarray) -> np.ndarray:
    """Convert complex snapshots into empirical covariance real/imag channels."""
    _, snapshots = y_snapshots.shape
    cov = (y_snapshots @ y_snapshots.conj().T) / float(snapshots)
    return np.stack([cov.real, cov.imag], axis=0).astype(np.float32)


def build_tau_corr(y_snapshots: np.ndarray, tau_max: int) -> np.ndarray:
    """Build multi-lag autocorrelation features with shape [tau+1, 2M, M]."""
    m, snapshots = y_snapshots.shape
    if tau_max >= snapshots:
        raise ValueError('tau_max must be smaller than the number of snapshots')

    tau_corr = np.zeros((tau_max + 1, 2 * m, m), dtype=np.float32)
    for lag in range(tau_max + 1):
        y1 = y_snapshots[:, :snapshots - lag]
        y2 = y_snapshots[:, lag:snapshots]
        corr = (y1 @ y2.conj().T) / float(snapshots - lag)
        tau_corr[lag, :m, :] = corr.real.astype(np.float32)
        tau_corr[lag, m:, :] = corr.imag.astype(np.float32)
    return tau_corr


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate SubspaceNet training data')
    parser.add_argument('--num_samples', type=int, default=200000, help='number of samples to generate')
    parser.add_argument('--out', type=str, default='Subspace_train_rho.h5', help='output HDF5 path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--snapshots_min', type=int, default=200, help='minimum snapshot count')
    parser.add_argument('--snapshots_max', type=int, default=200, help='maximum snapshot count')
    parser.add_argument('--snr_min', type=float, default=-20.0, help='minimum training SNR in dB')
    parser.add_argument('--snr_max', type=float, default=5.0, help='maximum training SNR in dB')
    parser.add_argument('--coherent_prob', type=float, default=0.2, help='probability of coherent sources')
    parser.add_argument('--coherent_rho_min', type=float, default=0.0, help='minimum coherence rho')
    parser.add_argument('--coherent_rho_max', type=float, default=1.0, help='maximum coherence rho')
    parser.add_argument('--m', type=int, default=10, help='number of sensors')
    parser.add_argument('--k', type=int, default=2, help='number of sources')
    parser.add_argument('--d', type=float, default=0.5, help='inter-element spacing in wavelengths')
    parser.add_argument('--tau_max', type=int, default=4, help='maximum autocorrelation lag')
    parser.add_argument('--angle_min', type=float, default=-60.0)
    parser.add_argument('--angle_max', type=float, default=60.0)
    parser.add_argument('--angle_step', type=float, default=0.5)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    angle_grid = np.arange(args.angle_min, args.angle_max + 0.5 * args.angle_step, args.angle_step, dtype=np.float32)
    array_pos = np.arange(args.m, dtype=np.float32) * args.d
    n_grid = angle_grid.size

    tau_corr = np.zeros((args.num_samples, args.tau_max + 1, 2 * args.m, args.m), dtype=np.float32)
    sam = np.zeros((args.num_samples, 2, args.m, args.m), dtype=np.float32)
    Y = np.zeros((args.num_samples, n_grid), dtype=np.float32)
    angles = np.zeros((args.num_samples, args.k), dtype=np.float32)
    snr_db = np.zeros(args.num_samples, dtype=np.float32)
    snapshots = np.zeros(args.num_samples, dtype=np.int32)
    rho = np.zeros(args.num_samples, dtype=np.float32)

    for i in range(args.num_samples):
        if i % 5000 == 0:
            print(f'Generating {i}/{args.num_samples}...')

        angle_idx = rng.choice(n_grid, size=args.k, replace=False)
        doa = np.sort(angle_grid[angle_idx]).astype(np.float32)
        A = steering_matrix(doa, array_pos)

        snr_val = float(rng.uniform(args.snr_min, args.snr_max))
        sigma2 = float(10.0 ** (-snr_val / 10.0))
        coherent = rng.random() < args.coherent_prob
        rho_val = float(rng.uniform(args.coherent_rho_min, args.coherent_rho_max)) if coherent else 0.0
        snapshots_count = int(rng.integers(args.snapshots_min, args.snapshots_max + 1))
        if coherent and args.k == 2 and rho_val >= 1.0:
            s1 = complex_gaussian((1, snapshots_count), rng)
            source_symbols = np.vstack([s1, s1]).astype(np.complex64)
        else:
            Rs = make_source_cov(args.k, coherent, rho_val)
            source_cholesky = np.linalg.cholesky(Rs + 1e-6 * np.eye(args.k, dtype=np.complex64)).astype(np.complex64)
            source_symbols = source_cholesky @ complex_gaussian((args.k, snapshots_count), rng)
        noise = np.sqrt(sigma2).astype(np.float32) * complex_gaussian((args.m, snapshots_count), rng)
        y_snapshots = A @ source_symbols + noise

        tau_corr[i] = build_tau_corr(y_snapshots, args.tau_max)
        sam[i] = covariance_channels(y_snapshots)
        Y[i, angle_idx] = 1.0
        angles[i] = doa
        snr_db[i] = snr_val
        snapshots[i] = snapshots_count
        rho[i] = rho_val

    perm = rng.permutation(args.num_samples)
    tau_corr = tau_corr[perm]
    sam = sam[perm]
    Y = Y[perm]
    angles = angles[perm]
    snr_db = snr_db[perm]
    snapshots = snapshots[perm]
    rho = rho[perm]

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('tau_corr', data=tau_corr)
        f.create_dataset('sam', data=sam)
        f['X'] = f['tau_corr']
        f.create_dataset('Y', data=Y)
        f.create_dataset('angles', data=angles)
        f.create_dataset('angle_grid', data=angle_grid)
        f.create_dataset('snr_db', data=snr_db)
        f.create_dataset('snapshots', data=snapshots)
        f.create_dataset('rho', data=rho)

    print(f'Training data saved to {out_path}')
    print(f'tau_corr shape:   {tau_corr.shape}')
    print(f'sam shape:        {sam.shape}')
    print(f'Y shape:          {Y.shape}')
    print(f'angles shape:     {angles.shape}')


if __name__ == '__main__':
    main()
