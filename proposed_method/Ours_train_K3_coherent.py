"""
Ours_train_K3_coherent.py

Convenience wrapper for training the K=3 coherent-source Ours model used by B6.
It delegates to Ours_train.py while fixing the experiment defaults:
    data:   Ours_train_K3_coherent.h5
    output: ours_model_K3_coherent_rank7.pth
    k:      3
    rank:   7
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the K=3 coherent rank-7 Ours model via Ours_train.py."
    )
    parser.add_argument("--data", default="Ours_train_K3_coherent.h5")
    parser.add_argument("--output", default="ours_model_K3_coherent_rank7.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda_margin", type=float, default=1.0)
    parser.add_argument("--lambda_trace", type=float, default=1e-4)
    parser.add_argument("--bg_step", type=float, default=0.5)
    parser.add_argument("--true_tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args, extra = parser.parse_known_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "Ours_train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--data",
        args.data,
        "--output",
        args.output,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--val_split",
        str(args.val_split),
        "--patience",
        str(args.patience),
        "--rank",
        str(args.rank),
        "--k",
        str(args.k),
        "--d",
        str(args.d),
        "--margin",
        str(args.margin),
        "--lambda_margin",
        str(args.lambda_margin),
        "--lambda_trace",
        str(args.lambda_trace),
        "--bg_step",
        str(args.bg_step),
        "--true_tol",
        str(args.true_tol),
        "--seed",
        str(args.seed),
    ]
    cmd.extend(extra)

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(script_dir), check=True)


if __name__ == "__main__":
    main()
