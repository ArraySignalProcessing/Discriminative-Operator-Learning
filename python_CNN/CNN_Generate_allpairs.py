"""
Generate CNN training data with exhaustive two-source angle-pair coverage.

Default behavior for K=2:
- Enumerate all unordered angle pairs on the DOA grid, without any minimum separation constraint.
- Repeat each pair --samples_per_pair times.
- For each repeated pair, randomly sample SNR and source coherence parameters.

Output:
- X: theoretical covariance, shape (N, 3, M, M)
- Y: multi-hot DOA labels, shape (N, grid_size)
- angles: true DOAs in degrees, shape (N, K)
- angle_grid: fixed angle grid
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def steering_matrix(angles_deg: np.ndarray, array_pos: np.ndarray) -> np.ndarray:
    """Construct steering matrix A for selected DOAs and array positions."""
    theta = np.deg2rad(angles_deg).reshape(1, -1)
    pos = array_pos.reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * pos * np.sin(theta)).astype(np.complex64)


def build_angle_index_plan(n_grid: int, k: int, samples_per_pair: int, rng: np.random.Generator) -> np.ndarray:
    """Return repeated angle-index combinations. For K=2, all unordered pairs are used."""
    if k != 2:
        raise ValueError('Exhaustive all-pair generation is currently defined for k=2 only.')
    i_idx, j_idx = np.triu_indices(n_grid, k=1)
    pairs = np.stack([i_idx, j_idx], axis=1).astype(np.int32)
    plan = np.repeat(pairs, repeats=samples_per_pair, axis=0)
    return plan[rng.permutation(plan.shape[0])]


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate CNN theoretical-covariance training data with all angle-pair coverage')
    parser.add_argument('--out', type=str, default='CNN_train.h5', help='output HDF5 path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--samples_per_pair', type=int, default=7, help='number of random non-ideal realizations per angle pair')
    parser.add_argument('--coherent_prob', type=float, default=0.2, help='probability that a sample is coherent (0-1)')
    parser.add_argument('--coherent_rho_min', type=float, default=0.0, help='minimum coherence rho for coherent samples')
    parser.add_argument('--coherent_rho_max', type=float, default=1.0, help='maximum coherence rho for coherent samples')
    parser.add_argument('--m', type=int, default=10, help='number of sensors')
    parser.add_argument('--k', type=int, default=2, help='number of sources')
    parser.add_argument('--d', type=float, default=0.5, help='inter-element spacing in wavelengths')
    parser.add_argument('--angle_min', type=float, default=-60.0)
    parser.add_argument('--angle_max', type=float, default=60.0)
    parser.add_argument('--angle_step', type=float, default=0.5)
    parser.add_argument('--snr_min', type=float, default=-20.0)
    parser.add_argument('--snr_max', type=float, default=5.0)
    args = parser.parse_args()

    if args.samples_per_pair < 1:
        raise ValueError('samples_per_pair must be at least 1')

    rng = np.random.default_rng(args.seed)
    angle_grid = np.arange(args.angle_min, args.angle_max + 0.5 * args.angle_step, args.angle_step, dtype=np.float32)
    n_grid = int(angle_grid.size)
    pair_plan = build_angle_index_plan(n_grid, args.k, args.samples_per_pair, rng)
    num_samples = int(pair_plan.shape[0])

    X = np.zeros((num_samples, 3, args.m, args.m), dtype=np.float32)
    Y = np.zeros((num_samples, n_grid), dtype=np.float32)
    angles = np.zeros((num_samples, args.k), dtype=np.float32)
    snrs = np.zeros(num_samples, dtype=np.float32)
    rhos = np.zeros(num_samples, dtype=np.float32)

    array_pos = np.arange(args.m, dtype=np.float32) * args.d

    for i, ang_idx in enumerate(pair_plan):
        if i % 5000 == 0:
            print(f'Generating {i}/{num_samples}...')

        ang = angle_grid[ang_idx].astype(np.float32)
        A = steering_matrix(ang, array_pos)

        snr_db = float(rng.uniform(args.snr_min, args.snr_max))
        sigma2 = float(10.0 ** (-snr_db / 10.0))
        coherent = rng.random() < args.coherent_prob
        if coherent:
            rho = float(rng.uniform(args.coherent_rho_min, args.coherent_rho_max))
            source_cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.complex64)
        else:
            rho = 0.0
            source_cov = np.eye(args.k, dtype=np.complex64)

        R_the = A @ source_cov @ A.conj().T + sigma2 * np.eye(args.m, dtype=np.complex64)

        X[i, 0] = R_the.real.astype(np.float32)
        X[i, 1] = R_the.imag.astype(np.float32)
        X[i, 2] = np.angle(R_the).astype(np.float32)
        Y[i, ang_idx] = 1.0
        angles[i] = np.sort(ang)
        snrs[i] = snr_db
        rhos[i] = rho

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
        f.create_dataset('angles', data=angles)
        f.create_dataset('angle_grid', data=angle_grid)
        f.create_dataset('angle_pair_indices', data=pair_plan)
        f.create_dataset('snr_db', data=snrs)
        f.create_dataset('rho', data=rhos)
        f.attrs['samples_per_pair'] = args.samples_per_pair
        f.attrs['all_unordered_angle_pairs'] = True
        f.attrs['min_angle_separation_control'] = False

    print(f'Training data saved to {out_path}')
    print(f'Angle grid size: {n_grid}')
    print(f'Number of unordered angle pairs: {n_grid * (n_grid - 1) // 2}')
    print(f'Samples per pair: {args.samples_per_pair}')
    print(f'Total samples: {num_samples}')
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')
    print(f'Coherent prob: {args.coherent_prob}, rho range: {args.coherent_rho_min} to {args.coherent_rho_max}')


if __name__ == '__main__':
    main()
