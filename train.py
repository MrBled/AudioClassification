import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fcnn import FCNN  # or FCNN
from dataloader import AcousticScenesDatasetTA

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item()

@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    amp_enabled = torch.cuda.is_available()
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)  # (B, C, T, F)
        yb = yb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(xb)
            loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_count += xb.size(0)

    val_loss = total_loss / max(1, total_count)
    val_acc = total_correct / max(1, total_count)
    return val_loss, val_acc

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler) -> tuple[float, float]:
    model.train()
    running_loss, running_correct, running_count = 0.0, 0, 0
    amp_enabled = torch.cuda.is_available()

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * xb.size(0)
        running_correct += (logits.argmax(1) == yb).sum().item()
        running_count += xb.size(0)

    train_loss = running_loss / max(1, running_count)
    train_acc = running_correct / max(1, running_count)
    return train_loss, train_acc

def save_checkpoint(path: str,
                    epoch: int,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    val_metrics: dict,
                    class_names: list,
                    extra_cfg: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "class_names": class_names,
        "config": extra_cfg
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train FCNN on DCASE-style acoustic scenes")
    parser.add_argument("--root", type=str,
                        help="Dataset root (folder that contains meta.txt and audio/)",
                        default="/data/clement/data/acoustic_scenes/TUT-acoustic-scenes-2017-development")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--segment_seconds", type=float, default=10.0,
                        help="Random crop length in seconds; set None to use full clip")
    parser.add_argument("--val_split", type=float, default=0.10,
                        help="Validation fraction (0.10 = 10%)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_fcnn.pt")
    parser.add_argument("--model_capacity", type=str, default="fixed",
                        choices=["fixed", "simple"],
                        help="'fixed' keeps capacity of the old 6ch variant via block widths; "
                             "'simple' shrinks channels (uses in_channels directly).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & split 90:10
    ds = AcousticScenesDatasetTA(
        root_dir=args.root,
        meta_filename="meta.txt",
        audio_subdir="audio",
        sr=44100,
        n_mels=args.n_mels,
        segment_seconds=args.segment_seconds,
        compute_deltas=True,
        duplicate_channels_to=None,   # <-- 3 channels only
        seed=args.seed
    )

    n_total = len(ds)
    n_val = int(round(args.val_split * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        ds,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(2024)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    num_classes = ds.num_classes
    in_channels = 3  # (log-mel, Δ, ΔΔ)
    print(f"Classes ({num_classes}): {ds.class_names}")

    model = FCNN(
        num_classes=num_classes,
        in_channels=in_channels,
        block_channels=(144, 288, 576),
        attention_ratio=2,
        return_probs=False
    ).to(device)

    # --------------------------
    # Optim, loss, sched, AMP
    # --------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion
        )

        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc*100:5.1f}% | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc*100:5.1f}% | "
              f"lr: {lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                path=args.ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_metrics={"val_loss": val_loss, "val_acc": val_acc},
                class_names=ds.class_names,
                extra_cfg={
                    "in_channels": in_channels,
                    "n_mels": args.n_mels,
                    "segment_seconds": args.segment_seconds,
                    "model_capacity": args.model_capacity
                }
            )
            print(f"  ↳ Saved new best checkpoint: {args.ckpt} (val_acc={val_acc*100:.2f}% @ epoch {epoch})")

    print(f"Best val acc: {best_val_acc*100:.2f}% (epoch {best_epoch})")

if __name__ == "__main__":
    main()
