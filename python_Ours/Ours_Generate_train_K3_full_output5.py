"""
Ours_Generate_train_K3_full_output5.py

Generate full-grid K=3 ideal-covariance training data for the proposed
discriminative operator model. The experiment uses a fixed output width of 5
in the downstream model, but this data file itself only stores covariance
samples and true DOAs.

The generator covers every combination of three distinct angles on the fixed
DOA grid. SNR and source coherence are assigned by deterministic cycles so the
dataset is exactly reproducible without inflating the sample count.

Output HDF5 format:
    X:          (N, 2, M, M), real/imaginary covariance channels
    angles:     (N, 3), sorted true DOAs in degrees
    angle_grid: (G,), fixed DOA grid in degrees
    snr_db:     (N,), SNR assigned to each sample
    rho:        (N,), source coherence assigned to each sample
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import h5py
import numpy as np


def parse_float_list(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated float list")
    return np.asarray(values, dtype=np.float32)


def steering_matrix(angles_deg: np.ndarray, array_pos: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angles_deg).reshape(1, -1)
    pos = array_pos.reshape(-1, 1)
    return np.exp(1j * 2.0 * np.pi * pos * np.sin(theta)).astype(np.complex64)


def make_source_cov(k: int, rho: float) -> np.ndarray:
    if rho <= 0.0:
        return np.eye(k, dtype=np.complex64)
    eye = np.eye(k, dtype=np.complex64)
    ones = np.ones((k, k), dtype=np.complex64)
    return ((1.0 - rho) * eye + rho * ones).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full-grid K=3 ideal-covariance training data for output_width=5."
    )
    parser.add_argument("--out", type=str, default="Ours_train_K3_full_output5.h5")
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=float, default=0.5)
    parser.add_argument("--angle_min", type=float, default=-60.0)
    parser.add_argument("--angle_max", type=float, default=60.0)
    parser.add_argument("--angle_step", type=float, default=0.5)
    parser.add_argument("--snr_set", type=str, default="-20,-15,-10,-5,0,5,10,15")
    parser.add_argument("--rho_set", type=str, default="0,0.5,0.7,0.85,0.95,0.99")
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--output_width_target", type=int, default=5)
    args = parser.parse_args()

    if args.k != 3:
        raise ValueError("This full-combination generator is intended for --k 3")
    if args.m <= 0:
        raise ValueError("--m must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")

    snr_set = parse_float_list(args.snr_set)
    rho_set = parse_float_list(args.rho_set)
    if np.any(rho_set < 0.0) or np.any(rho_set >= 1.0):
        raise ValueError("rho values must satisfy 0 <= rho < 1")

    angle_grid = np.arange(
        args.angle_min,
        args.angle_max + 0.5 * args.angle_step,
        args.angle_step,
        dtype=np.float32,
    )
    n_grid = int(angle_grid.size)
    if args.k > n_grid:
        raise ValueError(f"--k={args.k} exceeds angle grid size {n_grid}")

    num_samples = math.comb(n_grid, args.k)
    array_pos = np.arange(args.m, dtype=np.float32) * args.d
    eye_m = np.eye(args.m, dtype=np.complex64)
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    print(f"Angle grid size: {n_grid}")
    print(f"Total K=3 combinations: {num_samples}")
    print(f"Writing to: {out_path}")

    with h5py.File(out_path, "w") as f:
        x_ds = f.create_dataset(
            "X",
            shape=(num_samples, 2, args.m, args.m),
            dtype=np.float32,
            chunks=(min(args.chunk_size, num_samples), 2, args.m, args.m),
        )
        ang_ds = f.create_dataset(
            "angles",
            shape=(num_samples, args.k),
            dtype=np.float32,
            chunks=(min(args.chunk_size, num_samples), args.k),
        )
        snr_ds = f.create_dataset(
            "snr_db",
            shape=(num_samples,),
            dtype=np.float32,
            chunks=(min(args.chunk_size, num_samples),),
        )
        rho_ds = f.create_dataset(
            "rho",
            shape=(num_samples,),
            dtype=np.float32,
            chunks=(min(args.chunk_size, num_samples),),
        )
        f.create_dataset("angle_grid", data=angle_grid)

        x_buf = np.empty((args.chunk_size, 2, args.m, args.m), dtype=np.float32)
        ang_buf = np.empty((args.chunk_size, args.k), dtype=np.float32)
        snr_buf = np.empty((args.chunk_size,), dtype=np.float32)
        rho_buf = np.empty((args.chunk_size,), dtype=np.float32)

        write_start = 0
        buf_count = 0

        for sample_idx, idx_tuple in enumerate(itertools.combinations(range(n_grid), args.k)):
            doa = angle_grid[np.asarray(idx_tuple, dtype=np.int64)].astype(np.float32)
            snr_db = float(snr_set[sample_idx % snr_set.size])
            rho = float(rho_set[(sample_idx // snr_set.size) % rho_set.size])
            sigma2 = float(10.0 ** (-snr_db / 10.0))

            A = steering_matrix(doa, array_pos)
            Rs = make_source_cov(args.k, rho)
            R = (A @ Rs @ A.conj().T + sigma2 * eye_m).astype(np.complex64)

            x_buf[buf_count, 0] = R.real.astype(np.float32)
            x_buf[buf_count, 1] = R.imag.astype(np.float32)
            ang_buf[buf_count] = doa
            snr_buf[buf_count] = snr_db
            rho_buf[buf_count] = rho
            buf_count += 1

            if buf_count == args.chunk_size:
                write_end = write_start + buf_count
                x_ds[write_start:write_end] = x_buf[:buf_count]
                ang_ds[write_start:write_end] = ang_buf[:buf_count]
                snr_ds[write_start:write_end] = snr_buf[:buf_count]
                rho_ds[write_start:write_end] = rho_buf[:buf_count]
                write_start = write_end
                buf_count = 0
                print(f"Generated {write_start}/{num_samples} samples...", flush=True)

        if buf_count:
            write_end = write_start + buf_count
            x_ds[write_start:write_end] = x_buf[:buf_count]
            ang_ds[write_start:write_end] = ang_buf[:buf_count]
            snr_ds[write_start:write_end] = snr_buf[:buf_count]
            rho_ds[write_start:write_end] = rho_buf[:buf_count]
            write_start = write_end

        f.attrs["description"] = "K=3 full-grid ideal covariance training data"
        f.attrs["covariance_type"] = "ideal_theoretical"
        f.attrs["source_covariance_model"] = "Rs=I for rho=0; Rs=(1-rho)I+rho*11H for rho>0"
        f.attrs["doa_sampling"] = "all combinations of 3 distinct grid angles; no minimum separation"
        f.attrs["output_width_target"] = args.output_width_target
        f.attrs["m"] = args.m
        f.attrs["k"] = args.k
        f.attrs["d"] = args.d
        f.attrs["angle_min"] = args.angle_min
        f.attrs["angle_max"] = args.angle_max
        f.attrs["angle_step"] = args.angle_step
        f.attrs["snr_set"] = np.array2string(snr_set, separator=",")
        f.attrs["rho_set"] = np.array2string(rho_set, separator=",")

    print(f"Training data saved to {out_path}")
    print(f"Samples written: {write_start}")


if __name__ == "__main__":
    main()
