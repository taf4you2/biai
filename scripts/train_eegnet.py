import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def resolve_epoch_path(raw_path, dataset_dir):
    dataset_dir = Path(dataset_dir)
    epoch_path = Path(raw_path)
    candidates = [epoch_path]

    if not epoch_path.is_absolute():
        candidates.append(dataset_dir / epoch_path)

    parts = list(epoch_path.parts)
    if "epochs" in parts:
        epochs_idx = parts.index("epochs")
        candidates.append(dataset_dir.joinpath(*parts[epochs_idx:]))

    candidates.append(dataset_dir / "epochs" / epoch_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list((dataset_dir / "epochs").rglob(epoch_path.name))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Could not resolve epoch path {raw_path!r} under {dataset_dir}")


class EpochDataset(Dataset):
    def __init__(
        self,
        metadata,
        dataset_dir,
        label_encoder,
        mean=None,
        std=None,
        participant_stats=None,
        normalization="global",
        preload=True,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.label_encoder = label_encoder
        self.mean = mean
        self.std = std
        self.participant_stats = participant_stats or {}
        self.normalization = normalization
        self.preloaded_epochs = None
        self.preloaded_labels = None
        if preload:
            epochs = []
            labels = []
            for idx in range(len(self.metadata)):
                epoch, label = self._load_item(idx)
                epochs.append(epoch)
                labels.append(label)
            self.preloaded_epochs = torch.stack(epochs)
            self.preloaded_labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.metadata)

    def _resolve_epoch_path(self, raw_path):
        return resolve_epoch_path(raw_path, self.dataset_dir)

    def _load_item(self, idx):
        row = self.metadata.iloc[idx]
        epoch_path = self._resolve_epoch_path(row["epoch_path"])
        with np.load(epoch_path, allow_pickle=True) as data:
            epoch = data["epoch"].astype(np.float32)

        if self.normalization == "participant":
            participant = str(row["participant"])
            stats = self.participant_stats.get(participant)
            if stats is None:
                raise KeyError(f"No participant stats for {participant!r}")
            epoch = (epoch - stats["mean"]) / stats["std"]
        elif self.normalization == "epoch":
            mean = epoch.mean(axis=1, keepdims=True)
            std = epoch.std(axis=1, keepdims=True)
            epoch = (epoch - mean) / np.maximum(std, 1e-6)
        elif self.mean is not None and self.std is not None:
            epoch = (epoch - self.mean) / self.std

        # EEGNet expects N x 1 x channels x samples.
        epoch = epoch[None, :, :]
        label = self.label_encoder.transform([row["image_category"]])[0]
        return torch.from_numpy(epoch), int(label)

    def __getitem__(self, idx):
        if self.preloaded_epochs is not None and self.preloaded_labels is not None:
            return self.preloaded_epochs[idx], self.preloaded_labels[idx]
        epoch, label = self._load_item(idx)
        return epoch, torch.tensor(label, dtype=torch.long)


class EEGNet(nn.Module):
    def __init__(
        self,
        num_channels,
        num_samples,
        num_classes,
        f1=8,
        d=2,
        f2=16,
        kernel_length=128,
        dropout=0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(num_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 32), padding=(0, 16), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_samples)
            feature_count = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Linear(feature_count, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truthy_series(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_metadata(dataset_dir, only_qc_accepted=False):
    metadata = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    metadata = metadata[metadata["image_category"].notna()].copy()
    metadata = metadata[metadata["epoch_path"].notna()].copy()
    if only_qc_accepted:
        if "qc_accepted" not in metadata.columns:
            raise ValueError("--only-qc-accepted requires a metadata.csv column named 'qc_accepted'")
        metadata = metadata[truthy_series(metadata["qc_accepted"])].copy()
    return metadata


def apply_label_control(metadata, args):
    if args.label_control == "none":
        return metadata, "none"

    metadata = metadata.copy()
    metadata["original_image_category"] = metadata["image_category"]

    if args.label_control == "permute":
        rng = np.random.default_rng(args.seed)
        shuffled = metadata["image_category"].to_numpy(copy=True)
        rng.shuffle(shuffled)
        metadata["image_category"] = shuffled
        return metadata, f"permuted image_category with seed {args.seed}"

    raise ValueError(f"Unknown label control: {args.label_control}")


def split_metadata(metadata, args):
    if args.split == "participant":
        if "participant" not in metadata.columns:
            raise ValueError("Participant split requires a 'participant' column in metadata.csv")
        participants = sorted(str(value) for value in metadata["participant"].dropna().unique())
        test_participant = args.test_participant or participants[-1]
        if test_participant not in participants:
            raise ValueError(f"Unknown test participant {test_participant!r}; available: {participants}")
        test_mask = metadata["participant"].astype(str) == test_participant
        train_df = metadata.loc[~test_mask].copy()
        test_df = metadata.loc[test_mask].copy()
        return train_df, test_df, f"leave-one-participant; test participant: {test_participant}"

    if args.split == "participant_image":
        required = {"participant", "image_id"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"participant_image split requires metadata columns: {sorted(missing)}")
        participants = sorted(str(value) for value in metadata["participant"].dropna().unique())
        test_participant = args.test_participant or participants[-1]
        if test_participant not in participants:
            raise ValueError(f"Unknown test participant {test_participant!r}; available: {participants}")

        participant_mask = metadata["participant"].astype(str) == test_participant
        participant_df = metadata.loc[participant_mask].copy()
        groups = participant_df["image_id"].astype(str).to_numpy()
        if len(set(groups)) < 2:
            raise ValueError("participant_image split requires at least two image_id groups for the test participant")

        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        _, test_idx = next(splitter.split(participant_df, participant_df["image_category"], groups=groups))
        test_df = participant_df.iloc[test_idx].copy()
        test_images = set(test_df["image_id"].astype(str))

        image_ids = metadata["image_id"].astype(str)
        train_mask = (~participant_mask) & (~image_ids.isin(test_images))
        train_df = metadata.loc[train_mask].copy()
        excluded_count = int(len(metadata) - len(train_df) - len(test_df))
        if train_df.empty or test_df.empty:
            raise ValueError("participant_image split produced an empty train or test set")
        return (
            train_df,
            test_df,
            (
                "held-out participant + held-out image_id; "
                f"test participant: {test_participant}; "
                f"test images: {len(test_images)}; excluded samples: {excluded_count}"
            ),
        )

    if args.split == "series" and "series_id" in metadata.columns:
        groups = metadata["series_id"].fillna(-1).astype(int).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(metadata, metadata["image_category"], groups=groups))
        train_df = metadata.iloc[train_idx].copy()
        test_df = metadata.iloc[test_idx].copy()
        test_series = [int(series_id) for series_id in sorted(set(groups[test_idx]))]
        return train_df, test_df, f"grouped by series_id; test series: {test_series}"

    if args.split == "image" and "image_id" in metadata.columns:
        groups = metadata["image_id"].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(metadata, metadata["image_category"], groups=groups))
        train_df = metadata.iloc[train_idx].copy()
        test_df = metadata.iloc[test_idx].copy()
        return train_df, test_df, f"grouped by image_id; test images: {len(set(groups[test_idx]))}"

    train_df, test_df = train_test_split(
        metadata,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=metadata["image_category"],
    )
    return train_df.copy(), test_df.copy(), "random stratified"


def can_stratify(metadata, test_size):
    counts = metadata["image_category"].value_counts()
    if counts.empty or counts.min() < 2:
        return False

    if 0 < test_size < 1:
        test_count = int(np.ceil(len(metadata) * test_size))
    else:
        test_count = int(test_size)
    return test_count >= len(counts)


def split_validation_metadata(train_df, args):
    if args.val_size <= 0:
        return train_df.copy(), None, "disabled"

    val_split = args.val_split
    if val_split == "auto":
        participant_count = train_df["participant"].nunique() if "participant" in train_df.columns else 0
        if args.split == "participant" and participant_count > 1:
            val_split = "participant"
        elif "series_id" in train_df.columns and train_df["series_id"].nunique() > 1:
            val_split = "series"
        else:
            val_split = "random"

    if val_split == "participant" and "participant" in train_df.columns and train_df["participant"].nunique() > 1:
        groups = train_df["participant"].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        inner_train_idx, val_idx = next(splitter.split(train_df, train_df["image_category"], groups=groups))
        inner_train_df = train_df.iloc[inner_train_idx].copy()
        val_df = train_df.iloc[val_idx].copy()
        val_participants = sorted(str(value) for value in val_df["participant"].dropna().unique())
        return inner_train_df, val_df, f"grouped by participant; validation participants: {val_participants}"

    if val_split == "series" and "series_id" in train_df.columns and train_df["series_id"].nunique() > 1:
        groups = train_df["series_id"].fillna(-1).astype(int).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        inner_train_idx, val_idx = next(splitter.split(train_df, train_df["image_category"], groups=groups))
        inner_train_df = train_df.iloc[inner_train_idx].copy()
        val_df = train_df.iloc[val_idx].copy()
        val_series = [int(series_id) for series_id in sorted(set(groups[val_idx]))]
        return inner_train_df, val_df, f"grouped by series_id; validation series: {val_series}"

    stratify = train_df["image_category"] if can_stratify(train_df, args.val_size) else None
    inner_train_df, val_df = train_test_split(
        train_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify,
    )
    description = "random stratified" if stratify is not None else "random"
    return inner_train_df.copy(), val_df.copy(), description


def compute_train_stats(train_df, dataset_dir):
    channel_sum = None
    channel_sq_sum = None
    sample_count = 0
    dataset_dir = Path(dataset_dir)

    for _, row in train_df.iterrows():
        epoch_path = resolve_epoch_path(row["epoch_path"], dataset_dir)
        with np.load(epoch_path, allow_pickle=True) as data:
            epoch = data["epoch"].astype(np.float64)

        current_sum = epoch.sum(axis=1)
        current_sq_sum = (epoch**2).sum(axis=1)
        current_count = epoch.shape[1]

        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sq_sum = current_sq_sum if channel_sq_sum is None else channel_sq_sum + current_sq_sum
        sample_count += current_count

    mean = channel_sum / sample_count
    variance = channel_sq_sum / sample_count - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32)[:, None], std.astype(np.float32)[:, None]


def compute_participant_stats(metadata, dataset_dir):
    stats = {}
    for participant, participant_df in metadata.groupby(metadata["participant"].astype(str)):
        mean, std = compute_train_stats(participant_df, dataset_dir)
        stats[str(participant)] = {"mean": mean, "std": std}
    return stats


def make_balanced_sampler(train_df, label_encoder, mode):
    if mode == "none":
        return None

    weights = np.ones(len(train_df), dtype=np.float64)
    if mode in {"category", "category_participant"}:
        labels = label_encoder.transform(train_df["image_category"])
        _, counts = np.unique(labels, return_counts=True)
        label_weights = {label: 1.0 / counts[label] for label in range(len(counts))}
        weights *= np.array([label_weights[label] for label in labels])

    if mode in {"participant", "category_participant"}:
        participants = train_df["participant"].astype(str).to_numpy()
        participant_counts = pd.Series(participants).value_counts().to_dict()
        weights *= np.array([1.0 / participant_counts[participant] for participant in participants])

    weights = weights / weights.sum()
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item() * len(y))
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    mean_loss = sum(losses) / max(1, len(y_true))
    accuracy = accuracy_score(y_true, y_pred)
    return mean_loss, accuracy, np.array(y_true), np.array(y_pred)


def train(args):
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.dataset_dir, args.only_qc_accepted)
    metadata, label_control_description = apply_label_control(metadata, args)
    train_df, test_df, split_description = split_metadata(metadata, args)
    train_df, val_df, validation_description = split_validation_metadata(train_df, args)

    label_encoder = LabelEncoder()
    label_encoder.fit(metadata["image_category"])
    labels = label_encoder.classes_

    if args.normalization == "participant":
        if "participant" not in metadata.columns:
            raise ValueError("Participant normalization requires a 'participant' column in metadata.csv")
        train_participant_stats = compute_participant_stats(train_df, args.dataset_dir)
        # Validation/test participant stats use only unlabeled samples from their own split.
        val_participant_stats = compute_participant_stats(val_df, args.dataset_dir) if val_df is not None else None
        test_participant_stats = compute_participant_stats(test_df, args.dataset_dir)
        mean, std = None, None
    else:
        train_participant_stats = None
        val_participant_stats = None
        test_participant_stats = None
        mean, std = compute_train_stats(train_df, args.dataset_dir)

    train_dataset = EpochDataset(
        train_df,
        args.dataset_dir,
        label_encoder,
        mean=mean,
        std=std,
        participant_stats=train_participant_stats,
        normalization=args.normalization,
        preload=not args.no_preload,
    )
    val_dataset = None
    if val_df is not None:
        val_dataset = EpochDataset(
            val_df,
            args.dataset_dir,
            label_encoder,
            mean=mean,
            std=std,
            participant_stats=val_participant_stats,
            normalization=args.normalization,
            preload=not args.no_preload,
        )
    test_dataset = EpochDataset(
        test_df,
        args.dataset_dir,
        label_encoder,
        mean=mean,
        std=std,
        participant_stats=test_participant_stats,
        normalization=args.normalization,
        preload=not args.no_preload,
    )

    sampler = make_balanced_sampler(train_df, label_encoder, args.balanced_sampler)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    first_x, _ = train_dataset[0]
    _, num_channels, num_samples = first_x.shape
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EEGNet(
        num_channels=num_channels,
        num_samples=num_samples,
        num_classes=len(labels),
        f1=args.f1,
        d=args.depth_multiplier,
        f2=args.f2,
        kernel_length=args.kernel_length,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    history = []

    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {len(metadata)}")
    print(f"Classes: {len(labels)} -> {', '.join(labels)}")
    print(f"Train/validation/test: {len(train_dataset)}/{len(val_dataset) if val_dataset is not None else 0}/{len(test_dataset)}")
    print(f"Split: {split_description}")
    print(f"Validation split: {validation_description}")
    print(f"Label control: {label_control_description}")
    print(f"Only QC accepted: {args.only_qc_accepted}")
    print(f"Normalization: {args.normalization}")
    print(f"Balanced sampler: {args.balanced_sampler}")
    print(f"Input shape: {tuple(first_x.shape)}")
    print(f"Device: {device}")
    print()

    for epoch_idx in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_count += len(y)

        scheduler.step()
        train_loss = total_loss / max(1, total_count)
        history_row = {"epoch": epoch_idx, "train_loss": train_loss}

        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
            history_row.update({"val_loss": val_loss, "val_acc": val_acc})
            is_better = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
            if is_better:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch_idx
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"epoch {epoch_idx:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        else:
            best_epoch = epoch_idx
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"epoch {epoch_idx:03d} | train_loss={train_loss:.4f}")

        history.append(history_row)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    label_ids = np.arange(len(labels))
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels.tolist(),
            "mean": mean,
            "std": std,
            "args": vars(args),
            "input_shape": tuple(first_x.shape),
            "best_epoch": best_epoch,
        },
        output_dir / "eegnet.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "eegnet_history.csv", index=False)
    matrix_df.to_csv(output_dir / "eegnet_confusion_matrix.csv")

    summary = {
        "dataset": args.dataset_dir,
        "samples": int(len(metadata)),
        "train_samples": int(len(train_dataset)),
        "validation_samples": int(len(val_dataset)) if val_dataset is not None else 0,
        "test_samples": int(len(test_dataset)),
        "split": split_description,
        "validation_split": validation_description,
        "label_control": args.label_control,
        "label_control_description": label_control_description,
        "only_qc_accepted": bool(args.only_qc_accepted),
        "normalization": args.normalization,
        "balanced_sampler": args.balanced_sampler,
        "labels": labels.tolist(),
        "input_shape": tuple(int(dim) for dim in first_x.shape),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_validation_accuracy": float(best_val_acc) if val_loader is not None else None,
        "best_validation_loss": float(best_val_loss) if val_loader is not None else None,
        "best_test_accuracy": float(test_acc),
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
        "test_selected_by": "validation_accuracy" if val_loader is not None else "last_epoch",
    }
    with (output_dir / "eegnet_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "eegnet_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Samples: {len(metadata)}\n")
        f.write(f"Train/validation/test: {len(train_dataset)}/{len(val_dataset) if val_dataset is not None else 0}/{len(test_dataset)}\n")
        f.write(f"Split: {split_description}\n")
        f.write(f"Validation split: {validation_description}\n")
        f.write(f"Label control: {label_control_description}\n")
        f.write(f"Only QC accepted: {args.only_qc_accepted}\n")
        f.write(f"Normalization: {args.normalization}\n")
        f.write(f"Balanced sampler: {args.balanced_sampler}\n")
        f.write(f"Input shape: {tuple(first_x.shape)}\n")
        if val_loader is not None:
            f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
            f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Final test accuracy: {test_acc:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, labels=label_ids, target_names=labels, zero_division=0))
        f.write("\nConfusion matrix:\n")
        f.write(matrix_df.to_string())
        f.write("\n")

    print()
    if val_loader is not None:
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Saved model and reports to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet on trigger-aligned raw EEG epochs.")
    parser.add_argument("--dataset-dir", default="event_epoch_dataset")
    parser.add_argument("--output-dir", default="eegnet_results")
    parser.add_argument("--split", choices=["series", "random", "participant", "image", "participant_image"], default="series")
    parser.add_argument("--test-participant", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--val-split", choices=["auto", "random", "series", "participant"], default="auto")
    parser.add_argument("--label-control", choices=["none", "permute"], default="none")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--f2", type=int, default=16)
    parser.add_argument("--kernel-length", type=int, default=128)
    parser.add_argument("--normalization", choices=["global", "participant", "epoch"], default="global")
    parser.add_argument("--balanced-sampler", choices=["none", "category", "participant", "category_participant"], default="none")
    parser.add_argument("--only-qc-accepted", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
