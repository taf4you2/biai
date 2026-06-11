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
        epoch_path = Path(raw_path)
        if epoch_path.exists():
            return epoch_path
        return self.dataset_dir / "epochs" / epoch_path.name

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


def load_metadata(dataset_dir):
    metadata = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    metadata = metadata[metadata["image_category"].notna()].copy()
    metadata = metadata[metadata["epoch_path"].notna()].copy()
    return metadata


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

    if args.split == "series" and "series_id" in metadata.columns:
        groups = metadata["series_id"].fillna(-1).astype(int).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(metadata, metadata["image_category"], groups=groups))
        train_df = metadata.iloc[train_idx].copy()
        test_df = metadata.iloc[test_idx].copy()
        test_series = [int(series_id) for series_id in sorted(set(groups[test_idx]))]
        return train_df, test_df, f"grouped by series_id; test series: {test_series}"

    train_df, test_df = train_test_split(
        metadata,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=metadata["image_category"],
    )
    return train_df.copy(), test_df.copy(), "random stratified"


def compute_train_stats(train_df, dataset_dir):
    channel_sum = None
    channel_sq_sum = None
    sample_count = 0
    dataset_dir = Path(dataset_dir)

    for _, row in train_df.iterrows():
        epoch_path = Path(row["epoch_path"])
        if not epoch_path.exists():
            epoch_path = dataset_dir / "epochs" / epoch_path.name
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

    metadata = load_metadata(args.dataset_dir)
    train_df, test_df, split_description = split_metadata(metadata, args)

    label_encoder = LabelEncoder()
    label_encoder.fit(metadata["image_category"])
    labels = label_encoder.classes_

    if args.normalization == "participant":
        if "participant" not in metadata.columns:
            raise ValueError("Participant normalization requires a 'participant' column in metadata.csv")
        train_participant_stats = compute_participant_stats(train_df, args.dataset_dir)
        # Test participant stats are computed without labels from the test split itself.
        test_participant_stats = compute_participant_stats(test_df, args.dataset_dir)
        mean, std = None, None
    else:
        train_participant_stats = None
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

    best_acc = -1.0
    best_state = None
    history = []

    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {len(metadata)}")
    print(f"Classes: {len(labels)} -> {', '.join(labels)}")
    print(f"Train/test: {len(train_dataset)}/{len(test_dataset)}")
    print(f"Split: {split_description}")
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
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
        history.append({"epoch": epoch_idx, "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"epoch {epoch_idx:03d} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    matrix = confusion_matrix(y_true, y_pred)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels.tolist(),
            "mean": mean,
            "std": std,
            "args": vars(args),
            "input_shape": tuple(first_x.shape),
        },
        output_dir / "eegnet.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "eegnet_history.csv", index=False)
    matrix_df.to_csv(output_dir / "eegnet_confusion_matrix.csv")

    summary = {
        "dataset": args.dataset_dir,
        "samples": int(len(metadata)),
        "train_samples": int(len(train_dataset)),
        "test_samples": int(len(test_dataset)),
        "split": split_description,
        "normalization": args.normalization,
        "balanced_sampler": args.balanced_sampler,
        "labels": labels.tolist(),
        "input_shape": tuple(int(dim) for dim in first_x.shape),
        "best_test_accuracy": float(best_acc),
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
    }
    with (output_dir / "eegnet_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "eegnet_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Samples: {len(metadata)}\n")
        f.write(f"Train/test: {len(train_dataset)}/{len(test_dataset)}\n")
        f.write(f"Split: {split_description}\n")
        f.write(f"Normalization: {args.normalization}\n")
        f.write(f"Balanced sampler: {args.balanced_sampler}\n")
        f.write(f"Input shape: {tuple(first_x.shape)}\n")
        f.write(f"Best test accuracy: {best_acc:.4f}\n")
        f.write(f"Final test accuracy: {test_acc:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
        f.write("\nConfusion matrix:\n")
        f.write(matrix_df.to_string())
        f.write("\n")

    print()
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Saved model and reports to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet on trigger-aligned raw EEG epochs.")
    parser.add_argument("--dataset-dir", default="event_epoch_dataset")
    parser.add_argument("--output-dir", default="eegnet_results")
    parser.add_argument("--split", choices=["series", "random", "participant"], default="series")
    parser.add_argument("--test-participant", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
