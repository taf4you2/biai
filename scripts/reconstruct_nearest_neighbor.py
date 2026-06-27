import argparse
import json
import math
import sys
import textwrap
from pathlib import PurePosixPath
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eeg_image_retrieval import (
    EEGEmbeddingNet,
    EEGImageDataset,
    resolve_manifest_epoch_paths,
)


def load_font(size):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
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


def resolve_saved_dir(saved_path, project_root, required_file):
    project_root = Path(project_root)
    normalized = str(saved_path).replace("\\", "/")
    raw = Path(normalized)
    candidates = [raw, project_root / raw, project_root / raw.name]
    parts = PurePosixPath(normalized).parts
    for marker in ("biai_unclip", "reconstruction_manifests"):
        if marker in parts:
            start = parts.index(marker)
            suffix = parts[start + 1 :] if marker == "biai_unclip" else parts[start:]
            if suffix:
                candidates.append(project_root.joinpath(*suffix))
    for index, part in enumerate(parts):
        if part.startswith("image_embeddings_"):
            candidates.append(project_root.joinpath(*parts[index:]))
            break

    for candidate in candidates:
        if (candidate / required_file).is_file():
            return candidate
    tried = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve saved directory {saved_path!r} under {project_root}. "
        f"Expected {required_file}. Tried:\n{tried}"
    )


def resolve_image_lookup_path(path, project_root):
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


def rank_predictions(predicted, candidate_matrix, candidate_ids):
    scores = predicted @ candidate_matrix.T
    order = np.argsort(-scores, axis=1)
    predicted_ids = [candidate_ids[index] for index in order[:, 0]]
    return scores, order, predicted_ids


def add_metrics(frame, candidate_count):
    return {
        "samples": int(len(frame)),
        "candidates": int(candidate_count),
        "top1": float((frame["rank"] <= 1).mean()),
        "top5": float((frame["rank"] <= min(5, candidate_count)).mean()),
        "top10": float((frame["rank"] <= min(10, candidate_count)).mean()),
        "category_top1": float(
            (frame["target_category"] == frame["predicted_category"]).mean()
        ),
        "median_rank": float(frame["rank"].median()),
        "mean_rank": float(frame["rank"].mean()),
        "chance_top1": 1.0 / candidate_count,
        "chance_top5": min(5, candidate_count) / candidate_count,
        "chance_top10": min(10, candidate_count) / candidate_count,
    }


def make_grid(frame, output_path, title, image_lookup, rows=5, image_size=220):
    frame = frame.head(rows).reset_index(drop=True)
    title_lines = textwrap.wrap(title, width=43)
    header_height = 22 + 29 * len(title_lines)
    label_height = 76
    gap = 18
    pair_width = image_size * 2 + gap
    canvas_width = pair_width + 40
    canvas_height = header_height + len(frame) * (image_size + label_height + gap) + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(21)
    label_font = load_font(17)
    small_font = load_font(14)
    for line_number, line in enumerate(title_lines):
        draw.text((20, 12 + line_number * 29), line, fill="black", font=title_font)

    y = header_height
    for _, row in frame.iterrows():
        target = fit_image(image_lookup[row["target_image_id"]], image_size)
        predicted = fit_image(image_lookup[row["predicted_image_id"]], image_size)
        canvas.paste(target, (20, y))
        canvas.paste(predicted, (20 + image_size + gap, y))
        draw.rectangle((20, y, 20 + image_size, y + image_size), outline="#2e7d32", width=4)
        color = "#2e7d32" if row["target_category"] == row["predicted_category"] else "#c62828"
        draw.rectangle(
            (
                20 + image_size + gap,
                y,
                20 + image_size * 2 + gap,
                y + image_size,
            ),
            outline=color,
            width=4,
        )
        label_y = y + image_size + 7
        draw.text(
            (20, label_y),
            f"CEL: {row['target_category']}",
            fill="#1b5e20",
            font=label_font,
        )
        draw.text(
            (20 + image_size + gap, label_y),
            f"EEG: {row['predicted_category']}",
            fill=color,
            font=label_font,
        )
        draw.text(
            (20, label_y + 25),
            f"rank={int(row['rank'])}  powtorzenia={int(row.get('repetitions', 1))}",
            fill="#333333",
            font=small_font,
        )
        draw.text(
            (20, label_y + 46),
            str(row["target_image_id"])[:38],
            fill="#555555",
            font=small_font,
        )
        y += image_size + label_height + gap
    canvas.save(output_path, quality=95)


def select_diverse_examples(frame, count, best=True):
    ordered = frame.sort_values(["rank", "target_category"], ascending=[best, True])
    selected = []
    used_categories = set()
    for _, row in ordered.iterrows():
        if row["target_category"] not in used_categories:
            selected.append(row)
            used_categories.add(row["target_category"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        for _, row in ordered.iterrows():
            if row.name not in {item.name for item in selected}:
                selected.append(row)
            if len(selected) >= count:
                break
    return pd.DataFrame(selected)


def reconstruct(args):
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = torch.load(
        result_dir / "eeg_image_retrieval.pt",
        map_location="cpu",
        weights_only=False,
    )
    train_args = checkpoint["args"]
    project_root = Path(args.project_root or train_args.get("project_root", ".")).resolve()
    manifest_dir = resolve_saved_dir(
        train_args["manifest_dir"],
        project_root,
        "test.csv",
    )
    embedding_dir = resolve_saved_dir(
        train_args["embedding_dir"],
        project_root,
        "image_embeddings.npy",
    )
    test_frame = resolve_manifest_epoch_paths(
        pd.read_csv(manifest_dir / "test.csv"),
        project_root,
    ).reset_index(drop=True)
    embedding_matrix = np.load(embedding_dir / "image_embeddings.npy").astype(np.float32)
    embedding_index = pd.read_csv(embedding_dir / "image_embedding_index.csv")
    embedding_summary = json.loads(
        (embedding_dir / "embedding_summary.json").read_text(encoding="utf-8")
    )
    image_to_index = dict(
        zip(
            embedding_index["image_id"].astype(str),
            embedding_index["embedding_index"].astype(int),
        )
    )
    image_lookup = dict(
        zip(
            embedding_index["image_id"].astype(str),
            [
                resolve_image_lookup_path(image_path, project_root)
                for image_path in embedding_index["image_path"]
            ],
        )
    )
    category_lookup = dict(
        zip(
            embedding_index["image_id"].astype(str),
            embedding_index["image_category"].astype(str),
        )
    )

    dataset = EEGImageDataset(
        test_frame,
        image_to_index,
        image_to_index,
        embedding_matrix,
        checkpoint["mean"],
        checkpoint["std"],
        preload=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    _, channels, samples = checkpoint["input_shape"]
    model = EEGEmbeddingNet(
        channels,
        samples,
        embedding_matrix.shape[1],
        f1=train_args["f1"],
        depth_multiplier=train_args["depth_multiplier"],
        f2=train_args["f2"],
        kernel_length=train_args["kernel_length"],
        dropout=train_args["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predicted_batches = []
    target_ids = []
    with torch.inference_mode():
        for eeg, _, image_ids in loader:
            predicted_batches.append(model(eeg).cpu().numpy())
            target_ids.extend(image_ids)
    predicted_embeddings = np.concatenate(predicted_batches, axis=0)

    candidate_table = (
        test_frame[["image_id", "image_category"]]
        .drop_duplicates()
        .sort_values("image_id")
    )
    candidate_ids = candidate_table["image_id"].astype(str).tolist()
    candidate_matrix = embedding_matrix[[image_to_index[image_id] for image_id in candidate_ids]]
    _, sample_order, sample_predicted_ids = rank_predictions(
        predicted_embeddings, candidate_matrix, candidate_ids
    )
    candidate_position = {image_id: index for index, image_id in enumerate(candidate_ids)}
    sample_ranks = [
        int(np.where(sample_order[index] == candidate_position[target_id])[0][0] + 1)
        for index, target_id in enumerate(target_ids)
    ]
    sample_frame = test_frame[
        ["image_id", "image_category", "participant", "epoch_path_resolved"]
    ].copy()
    sample_frame = sample_frame.rename(
        columns={"image_id": "target_image_id", "image_category": "target_category"}
    )
    sample_frame["predicted_image_id"] = sample_predicted_ids
    sample_frame["predicted_category"] = [
        category_lookup[image_id] for image_id in sample_predicted_ids
    ]
    sample_frame["rank"] = sample_ranks
    sample_frame["repetitions"] = 1
    sample_frame.to_csv(output_dir / "sample_reconstructions.csv", index=False)

    aggregated_rows = []
    for target_id, indices in sample_frame.groupby("target_image_id").groups.items():
        averaged = predicted_embeddings[list(indices)].mean(axis=0)
        averaged /= max(np.linalg.norm(averaged), 1e-12)
        _, order, predicted_ids = rank_predictions(
            averaged[None, :], candidate_matrix, candidate_ids
        )
        rank = int(np.where(order[0] == candidate_position[target_id])[0][0] + 1)
        predicted_id = predicted_ids[0]
        aggregated_rows.append(
            {
                "target_image_id": target_id,
                "predicted_image_id": predicted_id,
                "target_category": category_lookup[target_id],
                "predicted_category": category_lookup[predicted_id],
                "rank": rank,
                "repetitions": len(indices),
            }
        )
    aggregate_frame = pd.DataFrame(aggregated_rows)
    metric_rows = []
    for _, row in aggregate_frame.iterrows():
        metrics = compare_images(
            image_lookup[row["target_image_id"]],
            image_lookup[row["predicted_image_id"]],
            args.metric_size,
        )
        metric_rows.append({f"nn_{key}": value for key, value in metrics.items()})
    aggregate_frame = pd.concat(
        [aggregate_frame.reset_index(drop=True), pd.DataFrame(metric_rows)],
        axis=1,
    )
    aggregate_frame.to_csv(output_dir / "image_averaged_reconstructions.csv", index=False)

    summary = {
        "result_dir": str(result_dir),
        "manifest_dir": str(manifest_dir),
        "method": f"nearest neighbor in frozen {embedding_summary['model']} embedding space",
        "sample_level": add_metrics(sample_frame, len(candidate_ids)),
        "image_averaged": add_metrics(aggregate_frame, len(candidate_ids)),
        "image_averaged_image_metrics": {
            key: float(aggregate_frame[f"nn_{key}"].mean())
            for key in ("l1", "mse", "psnr", "ssim")
        },
        "metric_size": int(args.metric_size),
    }
    with (output_dir / "reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    examples_dir = output_dir / "grids"
    examples_dir.mkdir()
    best = select_diverse_examples(aggregate_frame, args.grid_rows, best=True)
    worst = select_diverse_examples(aggregate_frame, args.grid_rows, best=False)
    category_correct = aggregate_frame[
        aggregate_frame["target_category"] == aggregate_frame["predicted_category"]
    ].sort_values("rank")
    make_grid(
        best,
        examples_dir / "best_reconstructions.jpg",
        "Najlepsze rekonstrukcje EEG -> najblizszy obraz",
        image_lookup,
        rows=args.grid_rows,
        image_size=args.image_size,
    )
    make_grid(
        worst,
        examples_dir / "worst_reconstructions.jpg",
        "Najtrudniejsze rekonstrukcje",
        image_lookup,
        rows=args.grid_rows,
        image_size=args.image_size,
    )
    if not category_correct.empty:
        make_grid(
            category_correct,
            examples_dir / "category_correct_reconstructions.jpg",
            "Poprawna kategoria odzyskana z EEG",
            image_lookup,
            rows=args.grid_rows,
            image_size=args.image_size,
        )

    page_size = args.grid_rows
    pages = math.ceil(len(aggregate_frame) / page_size)
    for page in range(pages):
        page_frame = aggregate_frame.iloc[page * page_size : (page + 1) * page_size]
        make_grid(
            page_frame,
            examples_dir / f"all_reconstructions_{page + 1:02d}.jpg",
            f"Wszystkie obrazy testowe — strona {page + 1}/{pages}",
            image_lookup,
            rows=page_size,
            image_size=args.image_size,
        )
    print(json.dumps(summary, indent=2))
    print(f"Saved nearest-neighbor reconstructions to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create nearest-neighbor image reconstructions from EEG embeddings."
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root used to resolve Colab/local manifest, image, and epoch paths.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=220)
    parser.add_argument("--metric-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    reconstruct(parse_args())
