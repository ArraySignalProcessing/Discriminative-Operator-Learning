"""CNN_Generate_train.py

Generate the minimal training set used by the current PyTorch trainer:
- X: theoretical covariance, shape (N, 3, M, M)
- Y: multi-hot DOA labels, shape (N, grid_size)
- angle_grid: fixed angle grid
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def steering_matrix(angles_deg: np.ndarray, array_pos: np.ndarray) -> np.ndarray:
    """Construct steering matrix A for selected DOAs and array positions."""
    theta = np.deg2rad(angles_deg).reshape(1, -1)
    pos = array_pos.reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * pos * np.sin(theta)).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate theoretical-covariance training data')
    parser.add_argument('--num_samples', type=int, default=200000, help='number of samples to generate')
    parser.add_argument('--out', type=str, default='CNN_train_rho.h5', help='output HDF5 path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--coherent_prob', type=float, default=0.2, help='probability that a sample is coherent (0-1)')
    parser.add_argument('--coherent_rho_min', type=float, default=0.0, help='minimum coherence rho for coherent samples')
    parser.add_argument('--coherent_rho_max', type=float, default=1.0, help='maximum coherence rho for coherent samples')
    args = parser.parse_args()

    num_samples = args.num_samples
    out = args.out
    np.random.seed(args.seed)

    m = 10
    k = 2
    d = 0.5
    angle_min = -60.0
    angle_max = 60.0
    angle_step = 0.5
    snr_min = -20.0
    snr_max = 5.0

    angle_grid = np.arange(angle_min, angle_max + 0.5 * angle_step, angle_step, dtype=np.float32)
    n_grid = int(angle_grid.size)

    x = np.zeros((num_samples, 3, m, m), dtype=np.float32)
    y = np.zeros((num_samples, n_grid), dtype=np.float32)

    array_pos = np.arange(m, dtype=np.float32) * d

    for i in range(num_samples):
        if i % 5000 == 0:
            print(f'Generating {i}/{num_samples}...')

        ang_idx = np.random.choice(n_grid, size=k, replace=False)
        ang = angle_grid[ang_idx]
        a = steering_matrix(ang, array_pos)

        snr_db = np.random.uniform(snr_min, snr_max)
        sigma2 = 10.0 ** (-snr_db / 10.0)
        if np.random.rand() < args.coherent_prob:
            rho = np.random.uniform(args.coherent_rho_min, args.coherent_rho_max)
            source_cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.complex64)
        else:
            source_cov = np.eye(k, dtype=np.complex64)

        r_the = a @ source_cov @ a.conj().T + sigma2 * np.eye(m, dtype=np.complex64)

        x[i, 0] = r_the.real.astype(np.float32)
        x[i, 1] = r_the.imag.astype(np.float32)
        x[i, 2] = np.angle(r_the).astype(np.float32)
        y[i, ang_idx] = 1.0

    perm = np.random.permutation(num_samples)
    x = x[perm]
    y = y[perm]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as f:
        f.create_dataset('X', data=x)
        f.create_dataset('Y', data=y)
        f.create_dataset('angle_grid', data=angle_grid)

    print(f'Training data saved to {out}')
    print(f'Total samples: {num_samples}')
    print(f'X shape: {x.shape}, Y shape: {y.shape}')
    print(f'Angle grid size: {n_grid}')
    print('Data generation completed.')
    print(f'Coherent prob: {args.coherent_prob}, rho range: {args.coherent_rho_min} to {args.coherent_rho_max}')


if __name__ == '__main__':
    main()
