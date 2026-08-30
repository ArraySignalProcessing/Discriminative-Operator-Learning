"""
B4 construction-criterion ablation training.

This script follows Ours_train.py and only changes the supervision objective.
It keeps the same network, PSD parameterization, trace normalization, training
data, optimizer style, and logging style.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from Ours_train import (
    EPS,
    DiscriminativeOperatorNet,
    OperatorDataset,
    operator_loss,
    steering_vectors_torch,
)


DEFAULT_OUTPUTS = {
    "projector_fitting": "ablation_models/B4_projector_fitting.pth",
    "null_only": "ablation_models/B4_null_only.pth",
    "full": "ablation_models/B4_full.pth",
}


def ideal_noise_projector(true_angles: torch.Tensor, m: int, d: float) -> torch.Tensor:
    """Construct nominal ideal noise projector from true DOAs."""
    A = steering_vectors_torch(true_angles, m=m, d=d).to(torch.complex64)  # [B, K, M]
    A = A.transpose(-1, -2)                                                # [B, M, K]
    AH = A.conj().transpose(-1, -2)                                        # [B, K, M]
    gram = AH @ A
    gram_inv = torch.linalg.pinv(gram)
    P_signal = A @ gram_inv @ AH
    eye = torch.eye(m, device=true_angles.device, dtype=A.dtype).unsqueeze(0)
    Pn = eye - P_signal
    return 0.5 * (Pn + Pn.conj().transpose(-1, -2))


def trace_regularizer(Q: torch.Tensor) -> torch.Tensor:
    tr = torch.diagonal(Q, dim1=-2, dim2=-1).real.sum(dim=-1)
    return ((tr - Q.shape[-1]) ** 2).mean()


def projector_fitting_loss(
    Q: torch.Tensor,
    true_angles: torch.Tensor,
    k: int,
    d: float,
    lambda_trace: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = Q.shape[-1]
    Pn = ideal_noise_projector(true_angles, m=m, d=d).to(Q.dtype)
    target = Pn * (m / (m - k))
    diff = Q - target
    loss_proj = (diff.abs() ** 2).sum(dim=(-2, -1)).mean()
    loss_trace = trace_regularizer(Q)
    total = loss_proj + lambda_trace * loss_trace
    return total, loss_proj, loss_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train B4 operator-ablation models")
    parser.add_argument(
        "--loss_mode",
        choices=["projector_fitting", "null_only", "full"],
        required=True,
    )
    parser.add_argument("--data", default="Ours_train.h5", help="HDF5 training data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--d", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda_margin", type=float, default=1.0)
    parser.add_argument("--lambda_trace", type=float, default=1e-4)
    parser.add_argument("--bg_step", type=float, default=0.5)
    parser.add_argument("--true_tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path

    if args.output is None:
        args.output = DEFAULT_OUTPUTS[args.loss_mode]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OperatorDataset(str(data_path), k=args.k)
    m = dataset.X.shape[-1]

    full_bg_grid_np = np.arange(
        float(dataset.angle_grid[0]),
        float(dataset.angle_grid[-1]) + 0.5 * args.bg_step,
        args.bg_step,
        dtype=np.float32,
    )
    bg_grid = torch.from_numpy(full_bg_grid_np).float().to(device)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = DiscriminativeOperatorNet(m=m, rank=args.rank).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=8,
    )

    best_val = float("inf")
    stale = 0
    history = []

    print(f"Training B4 ablation model: loss_mode={args.loss_mode}")
    print(f"Output: {output_path}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        model.train()
        sums = np.zeros(4, dtype=np.float64)
        count = 0

        for X, ang in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False, unit="batch"):
            X = X.to(device, non_blocking=True)
            ang = ang.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            Q = model.make_Q(X)

            if args.loss_mode == "projector_fitting":
                total, primary, trace = projector_fitting_loss(
                    Q, ang, args.k, args.d, args.lambda_trace
                )
                aux = torch.zeros_like(primary)
            else:
                lambda_margin = 0.0 if args.loss_mode == "null_only" else args.lambda_margin
                parts = operator_loss(
                    Q,
                    ang,
                    bg_grid,
                    margin=args.margin,
                    lambda_margin=lambda_margin,
                    lambda_trace=args.lambda_trace,
                    true_tol=args.true_tol,
                )
                total = parts.total
                primary = parts.null
                aux = parts.margin
                trace = parts.trace_reg

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            b = X.size(0)
            sums += b * np.array([total.item(), primary.item(), aux.item(), trace.item()])
            count += b

        train_avg = sums / max(count, 1)

        model.eval()
        sums = np.zeros(4, dtype=np.float64)
        count = 0

        with torch.no_grad():
            for X, ang in val_loader:
                X = X.to(device, non_blocking=True)
                ang = ang.to(device, non_blocking=True)
                Q = model.make_Q(X)

                if args.loss_mode == "projector_fitting":
                    total, primary, trace = projector_fitting_loss(
                        Q, ang, args.k, args.d, args.lambda_trace
                    )
                    aux = torch.zeros_like(primary)
                else:
                    lambda_margin = 0.0 if args.loss_mode == "null_only" else args.lambda_margin
                    parts = operator_loss(
                        Q,
                        ang,
                        bg_grid,
                        margin=args.margin,
                        lambda_margin=lambda_margin,
                        lambda_trace=args.lambda_trace,
                        true_tol=args.true_tol,
                    )
                    total = parts.total
                    primary = parts.null
                    aux = parts.margin
                    trace = parts.trace_reg

                b = X.size(0)
                sums += b * np.array([total.item(), primary.item(), aux.item(), trace.item()])
                count += b

        val_avg = sums / max(count, 1)
        scheduler.step(val_avg[0])
        history.append([epoch, *train_avg.tolist(), *val_avg.tolist()])

        primary_name = "proj" if args.loss_mode == "projector_fitting" else "null"
        aux_name = "margin"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train total/{primary_name}/{aux_name}/trace="
            f"{train_avg[0]:.6g}/{train_avg[1]:.6g}/{train_avg[2]:.6g}/{train_avg[3]:.6g} | "
            f"Val total/{primary_name}/{aux_name}/trace="
            f"{val_avg[0]:.6g}/{val_avg[1]:.6g}/{val_avg[2]:.6g}/{val_avg[3]:.6g} | "
            f"Time={time.perf_counter() - t0:.1f}s"
        )

        if val_avg[0] < best_val:
            best_val = float(val_avg[0])
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "m": m,
                    "rank": args.rank,
                    "angle_grid": dataset.angle_grid,
                    "bg_grid": full_bg_grid_np,
                    "loss_mode": args.loss_mode,
                    "args": vars(args),
                },
                output_path,
            )
            print(f"  -> Best model saved to {output_path}")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val loss={best_val:.6g}")
                break

    history_path = output_path.with_name(output_path.stem + "_history.npz")
    np.savez(history_path, history=np.asarray(history, dtype=np.float64))
    print(f"Training finished. Best val loss={best_val:.6g}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
