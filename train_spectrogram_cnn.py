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
from torch.utils.data import DataLoader, Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, metadata, dataset_dir, label_encoder, mean=None, std=None):
        self.metadata = metadata.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.label_encoder = label_encoder
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.metadata)

    def _resolve_tensor_path(self, raw_path):
        tensor_path = Path(raw_path)
        if tensor_path.exists():
            return tensor_path
        return self.dataset_dir / "tensors" / tensor_path.name

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        tensor_path = self._resolve_tensor_path(row["tensor_path"])
        with np.load(tensor_path, allow_pickle=True) as data:
            spectrogram = data["spectrogram"].astype(np.float32)

        if self.mean is not None and self.std is not None:
            spectrogram = (spectrogram - self.mean) / self.std

        label = self.label_encoder.transform([row["image_category"]])[0]
        return torch.from_numpy(spectrogram), torch.tensor(label, dtype=torch.long)


class SmallSpectrogramCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(dataset_dir):
    metadata = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    metadata = metadata[metadata["image_category"].notna()].copy()
    metadata = metadata[metadata["tensor_path"].notna()].copy()
    return metadata


def split_metadata(metadata, args):
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
        tensor_path = Path(row["tensor_path"])
        if not tensor_path.exists():
            tensor_path = dataset_dir / "tensors" / tensor_path.name
        with np.load(tensor_path, allow_pickle=True) as data:
            spectrogram = data["spectrogram"].astype(np.float64)

        axes = (1, 2)
        current_sum = spectrogram.sum(axis=axes)
        current_sq_sum = (spectrogram**2).sum(axis=axes)
        current_count = spectrogram.shape[1] * spectrogram.shape[2]

        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sq_sum = current_sq_sum if channel_sq_sum is None else channel_sq_sum + current_sq_sum
        sample_count += current_count

    mean = channel_sum / sample_count
    variance = channel_sq_sum / sample_count - mean**2
    std = np.sqrt(np.maximum(variance, 1e-8))
    return mean.astype(np.float32)[:, None, None], std.astype(np.float32)[:, None, None]


def evaluate(model, loader, device):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    criterion = nn.CrossEntropyLoss()
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

    mean, std = compute_train_stats(train_df, args.dataset_dir)
    train_dataset = SpectrogramDataset(train_df, args.dataset_dir, label_encoder, mean=mean, std=std)
    test_dataset = SpectrogramDataset(test_df, args.dataset_dir, label_encoder, mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    first_x, _ = train_dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SmallSpectrogramCNN(input_channels=first_x.shape[0], num_classes=len(labels)).to(device)

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
    print(f"Input shape: {tuple(first_x.shape)}")
    print(f"Device: {device}")
    print()

    for epoch in range(1, args.epochs + 1):
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
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

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
        output_dir / "spectrogram_cnn.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "spectrogram_cnn_history.csv", index=False)
    matrix_df.to_csv(output_dir / "spectrogram_cnn_confusion_matrix.csv")

    summary = {
        "dataset": args.dataset_dir,
        "samples": int(len(metadata)),
        "train_samples": int(len(train_dataset)),
        "test_samples": int(len(test_dataset)),
        "split": split_description,
        "labels": labels.tolist(),
        "input_shape": tuple(int(dim) for dim in first_x.shape),
        "best_test_accuracy": float(best_acc),
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
    }
    with (output_dir / "spectrogram_cnn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "spectrogram_cnn_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Samples: {len(metadata)}\n")
        f.write(f"Train/test: {len(train_dataset)}/{len(test_dataset)}\n")
        f.write(f"Split: {split_description}\n")
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
    parser = argparse.ArgumentParser(description="Train a small PyTorch CNN on trigger-aligned EEG spectrograms.")
    parser.add_argument("--dataset-dir", default="event_spectrogram_dataset")
    parser.add_argument("--output-dir", default="cnn_results")
    parser.add_argument("--split", choices=["series", "random"], default="series")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
