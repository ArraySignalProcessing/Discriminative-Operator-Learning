"""
Ours_Generate_train.py

Training-data generator for the proposed discriminative operator learning method.
It is intentionally compatible with the existing CNN training format:
    X:          (N, 2, M, M), real/imaginary covariance channels
    Y:          (N, grid_size), multi-hot DOA labels
    angles:     (N, K), true DOAs in degrees, added for convenience
    angle_grid: (grid_size,)

By default this script reproduces the CNN theoretical-covariance training data.
Use --use_sample_cov to train on finite-snapshot sample covariance instead.
"""

import argparse
import h5py
import numpy as np


def steering_matrix(angles_deg: np.ndarray, array_pos: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angles_deg).reshape(1, -1)
    pos = array_pos.reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * pos * np.sin(theta)).astype(np.complex64)


def make_source_cov(k: int, coherent: bool, rho: float) -> np.ndarray:
    if coherent and k == 2:
        return np.array([[1.0, rho], [rho, 1.0]], dtype=np.complex64)
    return np.eye(k, dtype=np.complex64)


def complex_gaussian(shape, rng: np.random.Generator) -> np.ndarray:
    return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)).astype(np.complex64)


def sample_covariance(A: np.ndarray, Rs: np.ndarray, sigma2: float, snapshots: int, rng: np.random.Generator) -> np.ndarray:
    """Generate finite-snapshot sample covariance from x(t)=A s(t)+n(t)."""
    k = A.shape[1]
    m = A.shape[0]
    # s = C z where C C^H = Rs. Add tiny jitter for near-singular coherent cases.
    C = np.linalg.cholesky(Rs + 1e-6 * np.eye(k, dtype=np.complex64)).astype(np.complex64)
    z = complex_gaussian((k, snapshots), rng)
    s = C @ z
    n = np.sqrt(sigma2).astype(np.float32) * complex_gaussian((m, snapshots), rng)
    x = A @ s + n
    return (x @ x.conj().T / snapshots).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate training data for the proposed DOA discriminative operator network')
    parser.add_argument('--num_samples', type=int, default=200000)
    parser.add_argument('--out', type=str, default='Ours_train_rho.h5')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--d', type=float, default=0.5)
    parser.add_argument('--angle_min', type=float, default=-60.0)
    parser.add_argument('--angle_max', type=float, default=60.0)
    parser.add_argument('--angle_step', type=float, default=0.5)
    parser.add_argument('--snr_min', type=float, default=-20.0)
    parser.add_argument('--snr_max', type=float, default=5.0)
    parser.add_argument('--coherent_prob', type=float, default=0.2)
    parser.add_argument('--coherent_rho_min', type=float, default=0.0)
    parser.add_argument('--coherent_rho_max', type=float, default=1.0)
    parser.add_argument('--use_sample_cov', action='store_true', help='store finite-snapshot sample covariance instead of theoretical covariance')
    parser.add_argument('--snapshots_min', type=int, default=20)
    parser.add_argument('--snapshots_max', type=int, default=1000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    angle_grid = np.arange(args.angle_min, args.angle_max + 0.5 * args.angle_step, args.angle_step, dtype=np.float32)
    n_grid = angle_grid.size
    array_pos = np.arange(args.m, dtype=np.float32) * args.d

    X = np.zeros((args.num_samples, 2, args.m, args.m), dtype=np.float32)
    Y = np.zeros((args.num_samples, n_grid), dtype=np.float32)
    angles = np.zeros((args.num_samples, args.k), dtype=np.float32)
    snrs = np.zeros(args.num_samples, dtype=np.float32)
    snapshots = np.zeros(args.num_samples, dtype=np.int32)
    rhos = np.zeros(args.num_samples, dtype=np.float32)

    for i in range(args.num_samples):
        if i % 5000 == 0:
            print(f'Generating {i}/{args.num_samples}...')

        idx = rng.choice(n_grid, size=args.k, replace=False)
        doa = np.sort(angle_grid[idx]).astype(np.float32)
        A = steering_matrix(doa, array_pos)

        snr_db = rng.uniform(args.snr_min, args.snr_max)
        sigma2 = float(10.0 ** (-snr_db / 10.0))
        coherent = rng.random() < args.coherent_prob
        rho = float(rng.uniform(args.coherent_rho_min, args.coherent_rho_max)) if coherent else 0.0
        Rs = make_source_cov(args.k, coherent, rho)

        R_the = A @ Rs @ A.conj().T + sigma2 * np.eye(args.m, dtype=np.complex64)
        T = int(rng.integers(args.snapshots_min, args.snapshots_max + 1))
        R = sample_covariance(A, Rs, sigma2, T, rng) if args.use_sample_cov else R_the.astype(np.complex64)

        X[i, 0] = R.real.astype(np.float32)
        X[i, 1] = R.imag.astype(np.float32)
        Y[i, idx] = 1.0
        angles[i] = doa
        snrs[i] = snr_db
        snapshots[i] = T
        rhos[i] = rho

    perm = rng.permutation(args.num_samples)
    X, Y, angles = X[perm], Y[perm], angles[perm]
    snrs, snapshots, rhos = snrs[perm], snapshots[perm], rhos[perm]

    with h5py.File(args.out, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
        f.create_dataset('angles', data=angles)
        f.create_dataset('angle_grid', data=angle_grid)
        f.create_dataset('snr_db', data=snrs)
        f.create_dataset('snapshots', data=snapshots)
        f.create_dataset('rho', data=rhos)

    print(f'Training data saved to {args.out}')
    print(f'X shape: {X.shape}, Y shape: {Y.shape}, angles shape: {angles.shape}')
    print(f'use_sample_cov={args.use_sample_cov}')


if __name__ == '__main__':
    main()
