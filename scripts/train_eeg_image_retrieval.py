import argparse
import hashlib
import json
import os
import random
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_epoch_path(row, project_root):
    project_root = Path(project_root)
    candidates = []
    for column in ("epoch_path_resolved", "epoch_path"):
        raw_path = row.get(column)
        if pd.isna(raw_path):
            continue
        normalized = str(raw_path).replace("\\", "/")
        path = Path(normalized)
        candidates.extend((path, project_root / path))
        parts = PurePosixPath(normalized).parts
        if "epochs" in parts:
            candidates.append(project_root.joinpath(*parts[parts.index("epochs") :]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve EEG epoch for image {row.get('image_id')!r} under {project_root}"
    )


def resolve_manifest_epoch_paths(frame, project_root):
    frame = frame.copy()
    frame["epoch_path_resolved"] = [
        str(resolve_epoch_path(row, project_root)) for _, row in frame.iterrows()
    ]
    return frame


class EEGImageDataset(Dataset):
    def __init__(
        self,
        frame,
        image_to_embedding,
        target_embedding_by_image,
        embedding_matrix,
        mean,
        std,
        preload=True,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_to_embedding = image_to_embedding
        self.target_embedding_by_image = target_embedding_by_image
        self.embedding_matrix = embedding_matrix
        self.mean = mean
        self.std = std
        self.preloaded = None
        if preload:
            self.preloaded = [self._load(index) for index in range(len(self.frame))]

    def __len__(self):
        return len(self.frame)

    def _load(self, index):
        row = self.frame.iloc[index]
        with np.load(row["epoch_path_resolved"], allow_pickle=True) as data:
            epoch = data["epoch"].astype(np.float32)
        epoch = (epoch - self.mean) / self.std
        image_index = self.target_embedding_by_image[str(row["image_id"])]
        target = self.embedding_matrix[image_index]
        return (
            torch.from_numpy(epoch[None, :, :]),
            torch.from_numpy(target),
            str(row["image_id"]),
        )

    def __getitem__(self, index):
        return self.preloaded[index] if self.preloaded is not None else self._load(index)


class EEGEmbeddingNet(nn.Module):
    def __init__(
        self,
        num_channels,
        num_samples,
        embedding_dim,
        f1=8,
        depth_multiplier=2,
        f2=16,
        kernel_length=128,
        dropout=0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                (num_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                (1, 32),
                padding=(0, 16),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_samples)
            feature_count = int(np.prod(self.features(dummy).shape[1:]))
        self.projection = nn.Sequential(
            nn.Linear(feature_count, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, inputs):
        features = torch.flatten(self.features(inputs), start_dim=1)
        return nn.functional.normalize(self.projection(features), dim=1)


def compute_train_stats(frame):
    channel_sum = None
    channel_sq_sum = None
    sample_count = 0
    for epoch_path in frame["epoch_path_resolved"]:
        with np.load(epoch_path, allow_pickle=True) as data:
            epoch = data["epoch"].astype(np.float64)
        current_sum = epoch.sum(axis=1)
        current_sq_sum = (epoch**2).sum(axis=1)
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sq_sum = current_sq_sum if channel_sq_sum is None else channel_sq_sum + current_sq_sum
        sample_count += epoch.shape[1]
    mean = channel_sum / sample_count
    variance = channel_sq_sum / sample_count - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32)[:, None], std.astype(np.float32)[:, None]


def contrastive_loss(predicted, targets, temperature):
    logits = predicted @ targets.T / temperature
    labels = torch.arange(len(predicted), device=predicted.device)
    return (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)) / 2


def cosine_loss(predicted, targets):
    return (1.0 - (predicted * targets).sum(dim=1)).mean()


def evaluate(model, loader, candidate_matrix, candidate_ids, candidate_categories, device):
    model.eval()
    predicted_rows = []
    target_ids = []
    with torch.inference_mode():
        for eeg, _, image_ids in loader:
            predicted_rows.append(model(eeg.to(device)).cpu())
            target_ids.extend(image_ids)
    predicted = torch.cat(predicted_rows)
    candidates = torch.from_numpy(candidate_matrix)
    scores = predicted @ candidates.T
    order = scores.argsort(dim=1, descending=True)
    candidate_lookup = {image_id: index for index, image_id in enumerate(candidate_ids)}
    target_indices = torch.tensor([candidate_lookup[image_id] for image_id in target_ids])
    ranks = (order == target_indices[:, None]).nonzero()[:, 1] + 1
    predicted_indices = order[:, 0].numpy()
    predicted_ids = [candidate_ids[index] for index in predicted_indices]
    category_lookup = dict(zip(candidate_ids, candidate_categories))
    target_categories = [category_lookup[image_id] for image_id in target_ids]
    predicted_categories = [category_lookup[image_id] for image_id in predicted_ids]
    metrics = {
        "top1": float((ranks <= 1).float().mean()),
        "top5": float((ranks <= min(5, len(candidate_ids))).float().mean()),
        "top10": float((ranks <= min(10, len(candidate_ids))).float().mean()),
        "median_rank": float(ranks.float().median()),
        "mean_rank": float(ranks.float().mean()),
        "samples": int(len(ranks)),
        "candidates": int(len(candidate_ids)),
        "category_top1": float(
            np.mean(
                [
                    predicted == target
                    for predicted, target in zip(predicted_categories, target_categories)
                ]
            )
        ),
    }
    metrics["chance_top1"] = 1.0 / len(candidate_ids)
    metrics["chance_top5"] = min(5, len(candidate_ids)) / len(candidate_ids)
    metrics["chance_top10"] = min(10, len(candidate_ids)) / len(candidate_ids)
    predictions = pd.DataFrame(
        {
            "target_image_id": target_ids,
            "predicted_image_id": predicted_ids,
            "target_category": target_categories,
            "predicted_category": predicted_categories,
            "rank": ranks.numpy(),
        }
    )
    return metrics, predictions


def load_embedding_artifacts(embedding_dir, manifest_dir):
    embedding_dir = Path(embedding_dir)
    summary = json.loads((embedding_dir / "embedding_summary.json").read_text(encoding="utf-8"))
    expected_hash = file_sha256(Path(manifest_dir) / "split_summary.json")
    if summary["manifest_summary_sha256"] != expected_hash:
        raise ValueError("Embedding manifest hash does not match the requested manifests")
    matrix = np.load(embedding_dir / "image_embeddings.npy").astype(np.float32)
    index = pd.read_csv(embedding_dir / "image_embedding_index.csv")
    if len(index) != len(matrix):
        raise ValueError("Embedding index and matrix length differ")
    return matrix, index, summary


def save_training_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    history,
    best_state,
    best_epoch,
    best_metric,
    args,
):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_top5": best_metric,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "args": vars(args),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def restore_training_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_random_state"]])
    return checkpoint


def train(args):
    seed_everything(args.seed)
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)

    frames = {
        split: pd.read_csv(manifest_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    project_root = Path(args.project_root).resolve()
    frames = {
        split: resolve_manifest_epoch_paths(frame, project_root)
        for split, frame in frames.items()
    }
    matrix, image_index, embedding_summary = load_embedding_artifacts(
        args.embedding_dir, manifest_dir
    )
    image_to_embedding = dict(
        zip(image_index["image_id"].astype(str), image_index["embedding_index"].astype(int))
    )
    for split, frame in frames.items():
        missing = set(frame["image_id"].astype(str)) - set(image_to_embedding)
        if missing:
            raise ValueError(f"{split} has missing image embeddings: {sorted(missing)[:5]}")

    target_embedding_by_image = dict(image_to_embedding)
    if args.target_control == "permute":
        rng = np.random.default_rng(args.seed)
        image_ids = sorted(target_embedding_by_image)
        permuted_indices = np.array([target_embedding_by_image[image_id] for image_id in image_ids])
        rng.shuffle(permuted_indices)
        target_embedding_by_image = dict(zip(image_ids, permuted_indices.tolist()))

    mean, std = compute_train_stats(frames["train"])
    datasets = {
        split: EEGImageDataset(
            frame,
            image_to_embedding,
            target_embedding_by_image,
            matrix,
            mean,
            std,
            preload=not args.no_preload,
        )
        for split, frame in frames.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }

    first_eeg, _, _ = datasets["train"][0]
    _, channels, samples = first_eeg.shape
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EEGEmbeddingNet(
        channels,
        samples,
        matrix.shape[1],
        f1=args.f1,
        depth_multiplier=args.depth_multiplier,
        f2=args.f2,
        kernel_length=args.kernel_length,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    candidate_sets = {}
    for split in ("validation", "test"):
        candidate_table = (
            frames[split][["image_id", "image_category"]]
            .drop_duplicates()
            .sort_values("image_id")
        )
        ids = candidate_table["image_id"].astype(str).tolist()
        categories = candidate_table["image_category"].astype(str).tolist()
        candidate_sets[split] = (
            matrix[[image_to_embedding[image_id] for image_id in ids]],
            ids,
            categories,
        )

    best_metric = -1.0
    best_epoch = None
    best_state = None
    history = []
    start_epoch = 1
    checkpoint_path = output_dir / "training_checkpoint.pt"
    if args.resume and checkpoint_path.is_file():
        checkpoint = restore_training_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        best_state = checkpoint.get("best_state_dict")
        best_epoch = checkpoint.get("best_epoch")
        best_metric = float(checkpoint.get("best_validation_top5", -1.0))
    print(f"Manifest: {manifest_dir}")
    print(f"Train/validation/test: {len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}")
    print(f"Input: {tuple(first_eeg.shape)} -> embedding {matrix.shape[1]}")
    print(f"Loss: {args.loss}; device: {device}")
    print(f"Target control: {args.target_control}")
    if start_epoch > 1:
        print(f"Resuming from checkpoint: epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for eeg, targets, _ in loaders["train"]:
            eeg = eeg.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(eeg)
            loss = (
                contrastive_loss(predicted, targets, args.temperature)
                if args.loss == "contrastive"
                else cosine_loss(predicted, targets)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(eeg)
            total_count += len(eeg)
        scheduler.step()
        validation, _ = evaluate(
            model,
            loaders["validation"],
            candidate_sets["validation"][0],
            candidate_sets["validation"][1],
            candidate_sets["validation"][2],
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        selection_metric = validation["top5"]
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "mean": mean,
                    "std": std,
                    "input_shape": tuple(first_eeg.shape),
                    "embedding_summary": embedding_summary,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_validation_top5": best_metric,
                },
                output_dir / "best_partial.pt",
            )
        pd.DataFrame(history).to_csv(output_dir / "retrieval_history.partial.csv", index=False)
        with (output_dir / "training_progress.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "last_completed_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_top5": best_metric,
                    "latest_validation": validation,
                },
                handle,
                indent=2,
            )
        if args.checkpoint_every > 0 and (
            epoch % args.checkpoint_every == 0 or epoch == args.epochs
        ):
            save_training_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                best_state,
                best_epoch,
                best_metric,
                args,
            )
        print(
            f"epoch {epoch:03d} | loss={row['train_loss']:.4f} | "
            f"val_top1={validation['top1']:.4f} | val_top5={validation['top5']:.4f} | "
            f"median_rank={validation['median_rank']:.1f}"
        )

    model.load_state_dict(best_state)
    test, test_predictions = evaluate(
        model,
        loaders["test"],
        candidate_sets["test"][0],
        candidate_sets["test"][1],
        candidate_sets["test"][2],
        device,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "input_shape": tuple(first_eeg.shape),
            "embedding_summary": embedding_summary,
            "args": vars(args),
            "best_epoch": best_epoch,
        },
        output_dir / "eeg_image_retrieval.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "retrieval_history.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    summary = {
        "manifest_dir": str(manifest_dir),
        "embedding_dir": str(Path(args.embedding_dir)),
        "project_root": str(project_root),
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["validation"]),
        "test_samples": len(datasets["test"]),
        "input_shape": list(first_eeg.shape),
        "embedding_dim": int(matrix.shape[1]),
        "loss": args.loss,
        "target_control": args.target_control,
        "best_epoch": best_epoch,
        "selection_metric": "validation_top5",
        "best_validation_top5": best_metric,
        "test": test,
    }
    with (output_dir / "retrieval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG-to-image embedding retrieval.")
    parser.add_argument("--manifest-dir", default="reconstruction_manifests/image")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve Windows and Linux epoch paths in manifests.",
    )
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--f2", type=int, default=16)
    parser.add_argument("--kernel-length", type=int, default=128)
    parser.add_argument("--loss", choices=["cosine", "contrastive"], default="cosine")
    parser.add_argument("--target-control", choices=["none", "permute"], default="none")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
