import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eeg_image_retrieval import (  # noqa: E402
    EEGEmbeddingNet,
    EEGImageDataset,
    resolve_manifest_epoch_paths,
)
from train_eegnet import EEGNet  # noqa: E402


EPS = 1e-8


def resolve_path(path, project_root, required_file=None):
    project_root = Path(project_root)
    normalized = str(path).replace("\\", "/")
    raw = Path(normalized)
    candidates = [raw, project_root / raw, project_root / raw.name]
    parts = PurePosixPath(normalized).parts
    for marker in ("biai", "biai_unclip", "reconstruction_manifests"):
        if marker in parts:
            suffix = parts[parts.index(marker) + 1 :]
            if suffix:
                candidates.append(project_root.joinpath(*suffix))
    for index, part in enumerate(parts):
        if part.startswith("image_embeddings_") or part in {"event_epoch_multisession_image_on_0_0p8_qc", "images"}:
            candidates.append(project_root.joinpath(*parts[index:]))
            break

    for candidate in candidates:
        if required_file is None and candidate.exists():
            return candidate
        if required_file is not None and (candidate / required_file).is_file():
            return candidate
    tried = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {path!r}. Tried:\n{tried}")


def resolve_image_path(path, project_root):
    project_root = Path(project_root)
    normalized = str(path).replace("\\", "/")
    raw = Path(normalized)
    candidates = [raw, project_root / raw, project_root / "images" / raw.name]
    parts = PurePosixPath(normalized).parts
    if "images" in parts:
        candidates.append(project_root.joinpath(*parts[parts.index("images") :]))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(path)


def load_font(size):
    for candidate in [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_image(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def batch_ssim(predicted, target):
    c1, c2 = 0.01**2, 0.03**2
    mu_x = torch.nn.functional.avg_pool2d(predicted, 11, stride=1, padding=5)
    mu_y = torch.nn.functional.avg_pool2d(target, 11, stride=1, padding=5)
    sigma_x = torch.nn.functional.avg_pool2d(predicted.square(), 11, 1, 5) - mu_x.square()
    sigma_y = torch.nn.functional.avg_pool2d(target.square(), 11, 1, 5) - mu_y.square()
    sigma_xy = torch.nn.functional.avg_pool2d(predicted * target, 11, 1, 5) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean(dim=(1, 2, 3))


def image_metrics(predicted_image, target_image):
    predicted = TF.to_tensor(predicted_image).unsqueeze(0)
    reference = TF.to_tensor(target_image).unsqueeze(0)
    mse = (predicted - reference).square().mean()
    return {
        "l1": float((predicted - reference).abs().mean()),
        "mse": float(mse),
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-10))),
        "ssim": float(batch_ssim(predicted, reference)[0]),
    }


def compare_images(target_path, predicted_path, metric_size):
    target = fit_image(target_path, metric_size)
    predicted = fit_image(predicted_path, metric_size)
    return image_metrics(predicted, target)


def l2_normalize(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), EPS)


def zscore_columns(scores):
    return (scores - scores.mean(axis=0, keepdims=True)) / scores.std(axis=0, keepdims=True).clip(min=1e-6)


def zscore_rows(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / scores.std(axis=1, keepdims=True).clip(min=1e-6)


def logsumexp(values, axis):
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values, axis=axis) + np.log(
        np.exp(values - max_values).sum(axis=axis).clip(min=EPS)
    )


def load_embedding_artifacts(embedding_dir, project_root):
    embedding_dir = Path(embedding_dir)
    matrix = np.load(embedding_dir / "image_embeddings.npy").astype(np.float32)
    index = pd.read_csv(embedding_dir / "image_embedding_index.csv")
    image_to_index = dict(zip(index["image_id"].astype(str), index["embedding_index"].astype(int)))
    category_lookup = dict(zip(index["image_id"].astype(str), index["image_category"].astype(str)))
    image_lookup = {
        str(row.image_id): resolve_image_path(row.image_path, project_root)
        for row in index.itertuples(index=False)
    }
    return matrix, index, image_to_index, category_lookup, image_lookup


def load_retrieval_model(checkpoint_path, embedding_dim):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    _, channels, samples = checkpoint["input_shape"]
    model = EEGEmbeddingNet(
        channels,
        samples,
        embedding_dim,
        f1=args["f1"],
        depth_multiplier=args["depth_multiplier"],
        f2=args["f2"],
        kernel_length=args["kernel_length"],
        dropout=args["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_retrieval_scores(frame, model, checkpoint, embedding_matrix, image_to_index, candidate_ids, batch_size):
    candidate_matrix = embedding_matrix[[image_to_index[image_id] for image_id in candidate_ids]]
    dataset = EEGImageDataset(
        frame,
        image_to_index,
        image_to_index,
        embedding_matrix,
        checkpoint["mean"],
        checkpoint["std"],
        preload=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predicted_batches = []
    target_ids = []
    with torch.inference_mode():
        for eeg, _, image_ids in loader:
            predicted_batches.append(model(eeg).cpu().numpy())
            target_ids.extend([str(image_id) for image_id in image_ids])
    predicted = np.concatenate(predicted_batches, axis=0)
    sample_scores = predicted @ candidate_matrix.T

    rows = []
    for image_id, group_indices in pd.Series(range(len(target_ids))).groupby(target_ids).groups.items():
        indices = list(group_indices)
        averaged_embedding = l2_normalize(predicted[indices].mean(axis=0, keepdims=True))[0]
        averaged_scores = sample_scores[indices].mean(axis=0)
        rows.append(
            {
                "target_image_id": image_id,
                "repetitions": len(indices),
                "embedding": averaged_embedding,
                "scores": averaged_scores,
            }
        )

    rows = sorted(rows, key=lambda row: row["target_image_id"])
    target_ids = [row["target_image_id"] for row in rows]
    averaged_scores = np.vstack([row["scores"] for row in rows]).astype(np.float32)
    averaged_embeddings = np.vstack([row["embedding"] for row in rows]).astype(np.float32)
    repetitions = np.array([row["repetitions"] for row in rows], dtype=np.int64)
    embedding_scores = averaged_embeddings @ candidate_matrix.T
    return target_ids, averaged_scores, embedding_scores, repetitions


def compute_participant_stats(frame):
    stats = {}
    for participant, group in frame.groupby(frame["participant"].astype(str)):
        channel_sum = None
        channel_sq_sum = None
        sample_count = 0
        for epoch_path in group["epoch_path_resolved"]:
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
        stats[str(participant)] = {
            "mean": mean.astype(np.float32)[:, None],
            "std": std.astype(np.float32)[:, None],
        }
    return stats


class EEGNetProbDataset(Dataset):
    def __init__(self, frame, participant_stats):
        self.frame = frame.reset_index(drop=True)
        self.participant_stats = participant_stats

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(row["epoch_path_resolved"], allow_pickle=True) as data:
            epoch = data["epoch"].astype(np.float32)
        stats = self.participant_stats[str(row["participant"])]
        epoch = (epoch - stats["mean"]) / stats["std"]
        return torch.from_numpy(epoch[None, :, :]), str(row["image_id"])


def load_eegnet_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    _, channels, samples = checkpoint["input_shape"]
    model = EEGNet(
        num_channels=channels,
        num_samples=samples,
        num_classes=len(checkpoint["labels"]),
        f1=args["f1"],
        d=args["depth_multiplier"],
        f2=args["f2"],
        kernel_length=args["kernel_length"],
        dropout=args["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_eegnet_category_probs(frame, model, labels, batch_size):
    participant_stats = compute_participant_stats(frame)
    dataset = EEGNetProbDataset(frame, participant_stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    image_ids = []
    with torch.inference_mode():
        for eeg, batch_image_ids in loader:
            logits = model(eeg)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            image_ids.extend([str(image_id) for image_id in batch_image_ids])
    probs = np.concatenate(probs, axis=0)

    rows = []
    for image_id, group_indices in pd.Series(range(len(image_ids))).groupby(image_ids).groups.items():
        averaged = probs[list(group_indices)].mean(axis=0)
        averaged = averaged / averaged.sum().clip(min=EPS)
        rows.append({"target_image_id": image_id, "probs": averaged})
    rows = sorted(rows, key=lambda row: row["target_image_id"])
    return [row["target_image_id"] for row in rows], np.vstack([row["probs"] for row in rows]).astype(np.float32)


def candidate_table_for_split(frame):
    return frame[["image_id", "image_category"]].drop_duplicates().sort_values("image_id").reset_index(drop=True)


def category_matrix(labels, candidate_categories):
    label_to_index = {label: index for index, label in enumerate(labels)}
    candidate_label_indices = np.array([label_to_index[category] for category in candidate_categories])
    one_hot = np.zeros((len(candidate_categories), len(labels)), dtype=np.float32)
    one_hot[np.arange(len(candidate_categories)), candidate_label_indices] = 1.0
    return candidate_label_indices, one_hot


def category_scores_from_candidates(scores, candidate_categories, labels, reducer):
    result = np.full((scores.shape[0], len(labels)), -1e9, dtype=np.float32)
    for label_index, label in enumerate(labels):
        indices = [idx for idx, category in enumerate(candidate_categories) if category == label]
        if not indices:
            continue
        values = scores[:, indices]
        if reducer == "mean":
            result[:, label_index] = values.mean(axis=1)
        elif reducer == "max":
            result[:, label_index] = values.max(axis=1)
        elif reducer == "logsumexp":
            result[:, label_index] = logsumexp(values, axis=1)
        else:
            raise ValueError(f"Unknown reducer {reducer!r}")
    return zscore_rows(result)


def score_with_category_prior(base_scores, category_scores, candidate_label_indices, alpha):
    candidate_prior = category_scores[:, candidate_label_indices]
    return base_scores + alpha * candidate_prior


def restrict_to_topk_categories(base_scores, category_scores, candidate_label_indices, k):
    topk = np.argsort(-category_scores, axis=1)[:, :k]
    allowed = np.zeros_like(base_scores, dtype=bool)
    for row_index in range(base_scores.shape[0]):
        allowed[row_index] = np.isin(candidate_label_indices, topk[row_index])
    restricted = base_scores.copy()
    restricted[~allowed] = -1e9
    return restricted


def build_score_variants(base_scores, candidate_categories, labels, eegnet_probs=None):
    candidate_label_indices, _ = category_matrix(labels, candidate_categories)
    variants = {
        "raw_cosine": base_scores["raw_cosine"],
        "embedding_mean_cosine": base_scores["embedding_mean_cosine"],
        "candidate_zscore": base_scores["candidate_zscore"],
    }
    alphas = [0.5, 1.0, 1.5, 2.0]

    for reducer in ["mean", "max", "logsumexp"]:
        cat_scores = category_scores_from_candidates(
            base_scores["candidate_zscore"], candidate_categories, labels, reducer
        )
        for alpha in alphas:
            variants[f"self_{reducer}_category_prior_alpha_{alpha:g}"] = score_with_category_prior(
                base_scores["candidate_zscore"], cat_scores, candidate_label_indices, alpha
            )
        for k in [1, 2, 3, 5]:
            variants[f"self_{reducer}_top{k}_category_gate"] = restrict_to_topk_categories(
                base_scores["candidate_zscore"], cat_scores, candidate_label_indices, k
            )

    if eegnet_probs is not None:
        log_probs = np.log(eegnet_probs.clip(min=1e-6))
        zlog_probs = zscore_rows(log_probs)
        for alpha in alphas:
            variants[f"eegnet_zlogprob_alpha_{alpha:g}"] = score_with_category_prior(
                base_scores["candidate_zscore"], zlog_probs, candidate_label_indices, alpha
            )
        for k in [1, 2, 3]:
            variants[f"eegnet_top{k}_category_gate"] = restrict_to_topk_categories(
                base_scores["candidate_zscore"], eegnet_probs, candidate_label_indices, k
            )

        self_max = category_scores_from_candidates(
            base_scores["candidate_zscore"], candidate_categories, labels, "max"
        )
        for alpha in [0.5, 1.0]:
            for beta in [0.5, 1.0]:
                variants[f"hybrid_selfmax_{alpha:g}_eegnetzlog_{beta:g}"] = (
                    base_scores["candidate_zscore"]
                    + alpha * self_max[:, candidate_label_indices]
                    + beta * zlog_probs[:, candidate_label_indices]
                )

    true_category_scores = np.zeros((base_scores["candidate_zscore"].shape[0], len(labels)), dtype=np.float32)
    return variants, candidate_label_indices, true_category_scores


def evaluate_scores(method, scores, target_ids, candidate_ids, category_lookup, image_lookup, repetitions, metric_size, metric_cache):
    candidate_position = {image_id: index for index, image_id in enumerate(candidate_ids)}
    order = np.argsort(-scores, axis=1)
    rows = []
    for row_index, target_id in enumerate(target_ids):
        target_position = candidate_position[target_id]
        predicted_position = int(order[row_index, 0])
        predicted_id = candidate_ids[predicted_position]
        rank = int(np.where(order[row_index] == target_position)[0][0] + 1)
        metric_key = (target_id, predicted_id)
        if metric_key not in metric_cache:
            metric_cache[metric_key] = compare_images(
                image_lookup[target_id], image_lookup[predicted_id], metric_size
            )
        metrics = metric_cache[metric_key]
        rows.append(
            {
                "method": method,
                "target_image_id": target_id,
                "predicted_image_id": predicted_id,
                "target_category": category_lookup[target_id],
                "predicted_category": category_lookup[predicted_id],
                "rank": rank,
                "repetitions": int(repetitions[row_index]),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    counts = frame["predicted_image_id"].value_counts()
    candidate_count = len(candidate_ids)
    summary = {
        "method": method,
        "images": int(len(frame)),
        "top1": float((frame["rank"] <= 1).mean()),
        "top5": float((frame["rank"] <= min(5, candidate_count)).mean()),
        "top10": float((frame["rank"] <= min(10, candidate_count)).mean()),
        "category_top1": float((frame["target_category"] == frame["predicted_category"]).mean()),
        "median_rank": float(frame["rank"].median()),
        "mean_rank": float(frame["rank"].mean()),
        "l1": float(frame["l1"].mean()),
        "mse": float(frame["mse"].mean()),
        "psnr": float(frame["psnr"].mean()),
        "ssim": float(frame["ssim"].mean()),
        "distinct_predictions": int(counts.size),
        "top_prediction_share": float(counts.iloc[0] / len(frame)),
        "top3_prediction_share": float(counts.head(3).sum() / len(frame)),
    }
    return frame, summary


def add_true_category_oracle(variants, target_ids, candidate_categories, category_lookup, labels, candidate_label_indices):
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_category_scores = np.full((len(target_ids), len(labels)), -1e9, dtype=np.float32)
    for row_index, target_id in enumerate(target_ids):
        true_category_scores[row_index, label_to_index[category_lookup[target_id]]] = 1.0
    variants["true_category_oracle_candidate_zscore"] = restrict_to_topk_categories(
        variants["candidate_zscore"], true_category_scores, candidate_label_indices, 1
    )


def category_diagnostics(target_ids, candidate_categories, category_lookup, labels, split_name, base_scores, eegnet_probs):
    candidate_label_indices, _ = category_matrix(labels, candidate_categories)
    rows = []
    true_indices = np.array([labels.index(category_lookup[target_id]) for target_id in target_ids])

    self_max = category_scores_from_candidates(base_scores["candidate_zscore"], candidate_categories, labels, "max")
    for source, scores in [("self_max_from_zscore", self_max)]:
        order = np.argsort(-scores, axis=1)
        rows.append(
            {
                "split": split_name,
                "source": source,
                "category_top1": float((order[:, 0] == true_indices).mean()),
                "category_top3": float(np.mean([true_indices[i] in order[i, :3] for i in range(len(true_indices))])),
                "mean_true_category_score": float(scores[np.arange(len(scores)), true_indices].mean()),
            }
        )

    if eegnet_probs is not None:
        order = np.argsort(-eegnet_probs, axis=1)
        rows.append(
            {
                "split": split_name,
                "source": "eegnet_probs",
                "category_top1": float((order[:, 0] == true_indices).mean()),
                "category_top3": float(np.mean([true_indices[i] in order[i, :3] for i in range(len(true_indices))])),
                "mean_true_category_score": float(eegnet_probs[np.arange(len(eegnet_probs)), true_indices].mean()),
            }
        )
    return rows


def make_grid(frame, output_path, title, image_lookup, rows=8, image_size=190):
    selected = frame.sort_values(["ssim", "rank"], ascending=[False, True]).head(rows).reset_index(drop=True)
    title_lines = textwrap.wrap(title, width=46)
    gap = 14
    label_h = 66
    header_h = 18 + 25 * len(title_lines)
    canvas = Image.new(
        "RGB",
        (2 * image_size + gap + 30, header_h + len(selected) * (image_size + label_h + gap) + 10),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    small_font = load_font(12)
    for line_number, line in enumerate(title_lines):
        draw.text((15, 10 + line_number * 25), line, fill="black", font=title_font)
    y = header_h
    for _, row in selected.iterrows():
        target = fit_image(image_lookup[row["target_image_id"]], image_size)
        predicted = fit_image(image_lookup[row["predicted_image_id"]], image_size)
        canvas.paste(target, (15, y))
        canvas.paste(predicted, (15 + image_size + gap, y))
        color = "#2e7d32" if row["target_category"] == row["predicted_category"] else "#c62828"
        draw.rectangle((15, y, 15 + image_size, y + image_size), outline="#2e7d32", width=3)
        draw.rectangle(
            (15 + image_size + gap, y, 15 + 2 * image_size + gap, y + image_size),
            outline=color,
            width=3,
        )
        draw.text((15, y + image_size + 5), f"CEL {row['target_category']}: {row['target_image_id'][:28]}", fill="#333", font=small_font)
        draw.text((15 + image_size + gap, y + image_size + 5), f"PRED {row['predicted_category']}: {row['predicted_image_id'][:28]}", fill=color, font=small_font)
        draw.text((15 + image_size + gap, y + image_size + 25), f"rank={int(row['rank'])} SSIM={row['ssim']:.3f}", fill="#0d47a1", font=small_font)
        y += image_size + label_h + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def run_split(
    split_name,
    frame,
    retrieval_model,
    retrieval_checkpoint,
    embedding_matrix,
    image_to_index,
    category_lookup,
    image_lookup,
    labels,
    eegnet_model,
    eegnet_labels,
    args,
):
    candidate_table = candidate_table_for_split(frame)
    candidate_ids = candidate_table["image_id"].astype(str).tolist()
    candidate_categories = candidate_table["image_category"].astype(str).tolist()
    target_ids, sample_mean_scores, embedding_scores, repetitions = predict_retrieval_scores(
        frame,
        retrieval_model,
        retrieval_checkpoint,
        embedding_matrix,
        image_to_index,
        candidate_ids,
        args.batch_size,
    )
    if target_ids != candidate_ids:
        # The split is image-averaged and candidate-sorted by image_id. Keeping this invariant
        # makes ranks and category priors easy to audit.
        raise ValueError(f"{split_name}: target image order does not match candidate order")

    eegnet_probs = None
    if eegnet_model is not None:
        prob_ids, probs = predict_eegnet_category_probs(frame, eegnet_model, eegnet_labels, args.batch_size)
        if prob_ids != target_ids:
            raise ValueError(f"{split_name}: EEGNet probability order does not match target order")
        if list(eegnet_labels) != list(labels):
            raise ValueError("EEGNet labels differ from retrieval/category labels")
        eegnet_probs = probs

    base_scores = {
        "raw_cosine": sample_mean_scores,
        "embedding_mean_cosine": embedding_scores,
        "candidate_zscore": zscore_columns(sample_mean_scores),
    }
    variants, candidate_label_indices, _ = build_score_variants(
        base_scores, candidate_categories, labels, eegnet_probs=eegnet_probs
    )
    add_true_category_oracle(
        variants, target_ids, candidate_categories, category_lookup, labels, candidate_label_indices
    )

    frames = {}
    summaries = []
    split_dir = Path(args.output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    metric_cache = {}
    for method, scores in variants.items():
        frame_out, summary = evaluate_scores(
            method,
            scores,
            target_ids,
            candidate_ids,
            category_lookup,
            image_lookup,
            repetitions,
            args.metric_size,
            metric_cache,
        )
        frames[method] = frame_out
        summaries.append(summary)
        frame_out.to_csv(split_dir / f"{method}_reconstructions.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df["is_oracle"] = summary_df["method"].str.startswith("true_category_oracle")
    summary_df = summary_df.sort_values(["is_oracle", "ssim", "top5"], ascending=[True, False, False])
    summary_df.to_csv(split_dir / "method_comparison.csv", index=False)

    diagnostics = category_diagnostics(
        target_ids,
        candidate_categories,
        category_lookup,
        labels,
        split_name,
        base_scores,
        eegnet_probs,
    )
    pd.DataFrame(diagnostics).to_csv(split_dir / "category_diagnostics.csv", index=False)
    (split_dir / "metric_cache_summary.json").write_text(
        json.dumps(
            {
                "unique_image_pairs_scored": len(metric_cache),
                "methods": len(variants),
                "images": len(target_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "split": split_name,
        "target_ids": target_ids,
        "candidate_ids": candidate_ids,
        "summaries": summaries,
        "summary_df": summary_df,
        "frames": frames,
        "diagnostics": diagnostics,
    }


def main(args):
    project_root = Path(args.project_root).resolve()
    manifest_dir = resolve_path(args.manifest_dir, project_root, "test.csv")
    embedding_dir = resolve_path(args.embedding_dir, project_root, "image_embeddings.npy")
    output_dir = Path(args.output_dir)
    if args.force and output_dir.exists():
        resolved_output = output_dir.resolve()
        resolved_root = project_root.resolve()
        if not resolved_output.is_relative_to(resolved_root):
            raise ValueError(f"Refusing to remove output outside project root: {resolved_output}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_matrix, embedding_index, image_to_index, category_lookup, image_lookup = load_embedding_artifacts(
        embedding_dir, project_root
    )
    labels = sorted(embedding_index["image_category"].astype(str).unique().tolist())
    retrieval_model, retrieval_checkpoint = load_retrieval_model(args.retrieval_checkpoint, embedding_matrix.shape[1])

    eegnet_model = None
    eegnet_labels = None
    if args.eegnet_checkpoint:
        eegnet_model, eegnet_checkpoint = load_eegnet_model(args.eegnet_checkpoint)
        eegnet_labels = eegnet_checkpoint["labels"]

    frames = {
        split_name: resolve_manifest_epoch_paths(pd.read_csv(manifest_dir / f"{split_name}.csv"), project_root)
        for split_name in ["validation", "test"]
    }

    validation = run_split(
        "validation",
        frames["validation"],
        retrieval_model,
        retrieval_checkpoint,
        embedding_matrix,
        image_to_index,
        category_lookup,
        image_lookup,
        labels,
        eegnet_model,
        eegnet_labels,
        args,
    )
    test = run_split(
        "test",
        frames["test"],
        retrieval_model,
        retrieval_checkpoint,
        embedding_matrix,
        image_to_index,
        category_lookup,
        image_lookup,
        labels,
        eegnet_model,
        eegnet_labels,
        args,
    )

    validation_non_oracle = validation["summary_df"][~validation["summary_df"]["is_oracle"]]
    selected_validation = validation_non_oracle.sort_values(
        ["ssim", "top5", "category_top1"], ascending=[False, False, False]
    ).iloc[0]
    selected_method = str(selected_validation["method"])
    test_selected = test["summary_df"][test["summary_df"]["method"] == selected_method].iloc[0]
    test_best_non_oracle = test["summary_df"][~test["summary_df"]["is_oracle"]].sort_values(
        ["ssim", "top5", "category_top1"], ascending=[False, False, False]
    ).iloc[0]

    comparison_rows = []
    for label, row in [
        ("validation_selected", selected_validation),
        ("test_selected_by_validation", test_selected),
        ("test_best_non_oracle_posthoc", test_best_non_oracle),
    ]:
        comparison_rows.append({"selection": label, **row.drop(labels=["is_oracle"]).to_dict()})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "selected_method_comparison.csv", index=False)

    grid_dir = output_dir / "grids"
    grid_methods = ["raw_cosine", "candidate_zscore", selected_method, str(test_best_non_oracle["method"]), "true_category_oracle_candidate_zscore"]
    for method in dict.fromkeys(grid_methods):
        if method in test["frames"]:
            make_grid(
                test["frames"][method],
                grid_dir / f"{method}_best.jpg",
                f"Test: {method} — najlepsze według SSIM",
                image_lookup,
                rows=args.grid_rows,
                image_size=args.image_size,
            )

    summary = {
        "experiment": "category-aware EEG image retrieval/reranking",
        "manifest_dir": str(manifest_dir),
        "embedding_dir": str(embedding_dir),
        "retrieval_checkpoint": str(args.retrieval_checkpoint),
        "eegnet_checkpoint": str(args.eegnet_checkpoint) if args.eegnet_checkpoint else None,
        "selection_rule": "best non-oracle validation SSIM, tie-broken by top5 and category_top1",
        "selected_method": selected_method,
        "validation_selected": selected_validation.drop(labels=["is_oracle"]).to_dict(),
        "test_selected_by_validation": test_selected.drop(labels=["is_oracle"]).to_dict(),
        "test_best_non_oracle_posthoc": test_best_non_oracle.drop(labels=["is_oracle"]).to_dict(),
        "validation_category_diagnostics": validation["diagnostics"],
        "test_category_diagnostics": test["diagnostics"],
    }
    with (output_dir / "category_aware_reranking_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved category-aware reranking outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local category-aware reranking after EEG->CLIP retrieval."
    )
    parser.add_argument("--manifest-dir", default="reconstruction_manifests/participant_image_mole_no_abc")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--embedding-dir", default="image_embeddings_unclip_participant_image_mole_no_abc_local_20260627")
    parser.add_argument("--retrieval-checkpoint", default="wyniki colab/unclip_mole_retrieval/eeg_image_retrieval.pt")
    parser.add_argument("--eegnet-checkpoint", default="wyniki colab/eegnet_mole_colab/eegnet.pt")
    parser.add_argument("--output-dir", default="wyniki colab/category_aware_reranking_mole_local")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--metric-size", type=int, default=256)
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=190)
    parser.add_argument("--force", action="store_true", help="Remove the output directory before running.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
