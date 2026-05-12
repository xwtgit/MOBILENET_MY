import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dreamer import CLASS_NAMES, DreamerECGDataset, load_dreamer_samples, make_subject_split, save_split
from models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss detected ({loss.item()}). Try lowering --lr or disabling --amp for this model."
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, split: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for inputs, targets in tqdm(loader, desc=split, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, average="macro", labels=list(range(len(CLASS_NAMES))), zero_division=0),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_loader(dataset: DreamerECGDataset, batch_size: int, workers: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
    )


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, float | int | str]:
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_json = args.split_json
    if split_json is None:
        samples = load_dreamer_samples(args.dreamer_mat, args.label_threshold)
        split = make_subject_split(samples, args.val_fraction, args.test_fraction, seed)
        split_json = output_dir / "subject_split.json"
        save_split(split, split_json)

    dataset_train = DreamerECGDataset(
        args.dreamer_mat,
        "train",
        split_json=split_json,
        max_length=args.max_length,
        label_threshold=args.label_threshold,
        seed=seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    dataset_val = DreamerECGDataset(
        args.dreamer_mat,
        "val",
        split_json=split_json,
        max_length=dataset_train.max_length,
        label_threshold=args.label_threshold,
        seed=seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    dataset_test = DreamerECGDataset(
        args.dreamer_mat,
        "test",
        split_json=split_json,
        max_length=dataset_train.max_length,
        label_threshold=args.label_threshold,
        seed=seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    loader_train = build_loader(dataset_train, args.batch_size, args.workers, True, seed, device)
    loader_val = build_loader(dataset_val, args.batch_size, args.workers, False, seed, device)
    loader_test = build_loader(dataset_test, args.batch_size, args.workers, False, seed, device)

    model = build_model(
        args.model,
        in_channels=args.in_channels,
        num_classes=len(CLASS_NAMES),
        width_mult=args.width_mult,
        dropout=args.dropout,
        reduced_tail=args.reduced_tail,
        stochastic_depth_prob=args.stochastic_depth_prob,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"

    print(
        f"seed={seed} model={args.model} device={device} "
        f"params={count_parameters(model):,} max_length={dataset_train.max_length}"
    )
    print(f"train subjects={dataset_train.subjects} counts={dataset_train.label_counts()}")
    print(f"val subjects={dataset_val.subjects} counts={dataset_val.label_counts()}")
    print(f"test subjects={dataset_test.subjects} counts={dataset_test.label_counts()}")

    best_val_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    start = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loader_train, criterion, optimizer, device, use_amp)
        val_metrics = evaluate(model, loader_val, criterion, device, "val")
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "model_name": args.model,
                    "args": vars(args),
                    "max_length": dataset_train.max_length,
                    "class_names": CLASS_NAMES,
                },
                output_dir / "best.pth",
            )

    checkpoint = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loader_test, criterion, device, "test")

    result = {
        "seed": seed,
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "elapsed_seconds": time.time() - start,
        "checkpoint": str(output_dir / "best.pth"),
    }

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def summarize(results: list[dict[str, float | int | str]], output_dir: Path) -> None:
    accuracy = np.array([float(result["test_accuracy"]) for result in results], dtype=np.float64)
    macro_f1 = np.array([float(result["test_macro_f1"]) for result in results], dtype=np.float64)
    summary = {
        "model": results[0]["model"] if results else None,
        "seeds": [int(result["seed"]) for result in results],
        "test_accuracy_mean": float(accuracy.mean()),
        "test_accuracy_std": float(accuracy.std(ddof=0)),
        "test_macro_f1_mean": float(macro_f1.mean()),
        "test_macro_f1_std": float(macro_f1.std(ddof=0)),
        "results": results,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        "summary "
        f"acc={summary['test_accuracy_mean']:.4f}+/-{summary['test_accuracy_std']:.4f} "
        f"macro_f1={summary['test_macro_f1_mean']:.4f}+/-{summary['test_macro_f1_std']:.4f}"
    )


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 1D MobileNetV3 on DREAMER ECG emotion recognition")
    parser.add_argument("--dreamer-mat", required=True, type=str, help="Path to DREAMER.mat")
    parser.add_argument("--output-dir", default="outputs/dreamer_mobilenetv3_1d", type=str)
    parser.add_argument(
        "--model",
        default="mobilenet_v3_small_1d",
        choices=["mobilenet_v3_small_1d", "efficientnet_v2_s_1d", "resnet50_1d", "densenet169_1d"],
        help="1D model architecture",
    )
    parser.add_argument("--split-json", default=None, type=str, help="Optional subject split JSON with train/val/test keys")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seeds", default=[0, 1, 2, 3, 4], nargs="+", type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--width-mult", default=1.0, type=float)
    parser.add_argument("--reduced-tail", action="store_true")
    parser.add_argument("--stochastic-depth-prob", default=0.2, type=float)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--in-channels", default=2, type=int)
    parser.add_argument("--max-length", default=None, type=int, help="Pad/crop length. Defaults to the longest DREAMER ECG")
    parser.add_argument("--label-threshold", default=3.0, type=float)
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--test-fraction", default=0.2, type=float)
    return parser


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    results = [run_seed(args, seed) for seed in args.seeds]
    summarize(results, output_dir)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
