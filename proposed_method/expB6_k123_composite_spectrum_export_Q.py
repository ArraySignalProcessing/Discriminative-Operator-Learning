"""
B6 learned-Q export for K=1/2/3 composite-robustness spectra.

This script applies one trained Ours checkpoint, typically a K=3 model, to
all B6 K=1/2/3 covariance samples. It exports only learned Q matrices; it does
not estimate DOAs or compute RMSE/PoR.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from ours_test_common import (
    DEVICE,
    load_model_from_checkpoint,
    normalize_covariance_array,
)


DEFAULT_DATA_DIR = "../data/B6"
DEFAULT_OUT_DIR = "../data/B6"
DEFAULT_MODEL_PATH = "ours_model_K3_full_output5.pth"


def resolve_path(path: str, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return base_dir / p


def parse_scan_value(group_name: str) -> float:
    name = group_name.strip("/")
    if name.startswith("Theta_"):
        tag = name.split("_", 1)[1].replace("deg", "")
        sign = -1.0 if tag.startswith("min") else 1.0
        tag = tag[3:] if tag.startswith("min") else tag
        return sign * float(tag.replace("p", "."))
    if name.startswith("Delta_"):
        tag = name.split("_", 1)[1].replace("deg", "")
        return float(tag.replace("p", "."))
    return float("inf")


def sample_groups(fin: h5py.File) -> list[str]:
    groups = [name for name, obj in fin.items() if isinstance(obj, h5py.Group) and "sam" in obj]
    return sorted(groups, key=parse_scan_value)


def export_q_group(model, cov: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, cov.shape[0], batch_size):
            end = min(start + batch_size, cov.shape[0])
            x = torch.from_numpy(cov[start:end]).float().to(DEVICE)
            q = model.make_Q(x).detach().cpu().numpy().astype(np.complex64)
            chunks.append(q)
    return np.concatenate(chunks, axis=0)


def copy_if_exists(fin: h5py.File, fout: h5py.File, name: str) -> None:
    if name in fin:
        fout.create_dataset(name, data=np.array(fin[name]))


def export_for_k(
    args: argparse.Namespace,
    script_dir: Path,
    model,
    model_m: int,
    model_output_width: int,
    k_value: int,
) -> None:
    data_dir = resolve_path(args.data_dir, script_dir)
    out_dir = resolve_path(args.out_dir, script_dir)
    data_path = data_dir / f"B6_K123_CompositeSpectrum_QPn_K{k_value}.h5"
    out_path = out_dir / f"B6_K123_CompositeSpectrum_Q_K{k_value}.h5"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing B6 K={k_value} data file: {data_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with h5py.File(data_path, "r") as fin, h5py.File(out_path, "w") as fout:
        for key, value in fin.attrs.items():
            fout.attrs[key] = value
        fout.attrs["source_data"] = str(data_path)
        fout.attrs["model"] = args.model
        fout.attrs["model_output_width"] = model_output_width
        fout.attrs["description"] = "B6 learned Q export from one Ours checkpoint for K=1/2/3 spectra"

        copy_if_exists(fin, fout, "theta_vec")
        copy_if_exists(fin, fout, "delta_theta_vec")
        copy_if_exists(fin, fout, "angle_scan")

        groups = sample_groups(fin)
        print(f"K={k_value}: found {len(groups)} sample groups.")
        for group_name in groups:
            gin = fin[group_name]
            cov = normalize_covariance_array(np.array(gin["sam"]))
            if cov.shape[-1] != model_m:
                raise ValueError(
                    f"{group_name}: data M={cov.shape[-1]} does not match model m={model_m}"
                )
            print(f"K={k_value} {group_name}: exporting Q for {cov.shape[0]} samples...", flush=True)
            q = export_q_group(model, cov, args.batch_size)

            gout = fout.create_group(group_name)
            gout.create_dataset("Q_real", data=q.real.astype(np.float32))
            gout.create_dataset("Q_imag", data=q.imag.astype(np.float32))
            for optional_name in ["angles", "theta_true"]:
                if optional_name in gin:
                    gout.create_dataset(optional_name, data=np.array(gin[optional_name]))
            for key, value in gin.attrs.items():
                gout.attrs[key] = value

    print(f"K={k_value}: saved learned Q to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export learned Q for B6 K=1/2/3 composite spectra.")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output_width", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_path = resolve_path(args.model, script_dir)

    print(f"Device: {DEVICE}")
    print(f"Loading Ours model from {model_path}...")
    model, _angle_grid, model_m, output_width, _metadata = load_model_from_checkpoint(
        str(model_path),
        args.output_width,
    )
    print(f"Model loaded. m={model_m}, output_width={output_width}")

    for k_value in args.k_values:
        export_for_k(args, script_dir, model, model_m, output_width, int(k_value))


if __name__ == "__main__":
    main()
