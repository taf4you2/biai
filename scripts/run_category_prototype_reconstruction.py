import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_category_aware_reranking import (  # noqa: E402
    add_true_category_oracle,
    build_score_variants,
    category_matrix,
    load_eegnet_model,
    load_embedding_artifacts,
    load_retrieval_model,
    predict_eegnet_category_probs,
    predict_retrieval_scores,
    resolve_manifest_epoch_paths,
    resolve_path,
    zscore_columns,
)


EPS = 1e-8


def load_font(size):
    for candidate in [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_image(path, size):
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


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


def image_metrics_arrays(predicted, target):
    predicted_tensor = torch.from_numpy(predicted.transpose(2, 0, 1)).float().unsqueeze(0)
    target_tensor = torch.from_numpy(target.transpose(2, 0, 1)).float().unsqueeze(0)
    mse = (predicted_tensor - target_tensor).square().mean()
    return {
        "l1": float((predicted_tensor - target_tensor).abs().mean()),
        "mse": float(mse),
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-10))),
        "ssim": float(batch_ssim(predicted_tensor, target_tensor)[0]),
    }


def softmax(values, temperature):
    scaled = values / max(float(temperature), EPS)
    scaled = scaled - scaled.max()
    weights = np.exp(scaled)
    return weights / weights.sum().clip(min=EPS)


def load_image_arrays(candidate_ids, image_lookup, image_size):
    arrays = {}
    for image_id in candidate_ids:
        image = fit_image(image_lookup[image_id], image_size)
        arrays[image_id] = np.asarray(image).astype(np.float32) / 255.0
    return arrays


def prototype_for_row(scores, candidate_ids, image_arrays, top_k, temperature, anchor_top1_weight):
    order = np.argsort(-scores)
    selected_positions = order[:top_k]
    selected_scores = scores[selected_positions]
    weights = softmax(selected_scores, temperature)
    selected_images = np.stack([image_arrays[candidate_ids[index]] for index in selected_positions])
    prototype = np.tensordot(weights, selected_images, axes=(0, 0))
    if anchor_top1_weight > 0:
        top1 = image_arrays[candidate_ids[selected_positions[0]]]
        prototype = anchor_top1_weight * top1 + (1.0 - anchor_top1_weight) * prototype
    return np.clip(prototype, 0.0, 1.0), order


def evaluate_prototype_variant(
    variant_name,
    scores,
    target_ids,
    candidate_ids,
    image_arrays,
    category_lookup,
    top_k,
    temperature,
    anchor_top1_weight,
):
    candidate_position = {image_id: index for index, image_id in enumerate(candidate_ids)}
    rows = []
    generated = {}
    for row_index, target_id in enumerate(target_ids):
        prototype, order = prototype_for_row(
            scores[row_index],
            candidate_ids,
            image_arrays,
            top_k=top_k,
            temperature=temperature,
            anchor_top1_weight=anchor_top1_weight,
        )
        top1_id = candidate_ids[int(order[0])]
        rank = int(np.where(order == candidate_position[target_id])[0][0] + 1)
        metrics = image_metrics_arrays(prototype, image_arrays[target_id])
        generated[target_id] = prototype
        rows.append(
            {
                "variant": variant_name,
                "score_method": variant_name,
                "target_image_id": target_id,
                "top1_image_id": top1_id,
                "target_category": category_lookup[target_id],
                "top1_category": category_lookup[top1_id],
                "rank": rank,
                "top_k": int(top_k),
                "temperature": float(temperature),
                "anchor_top1_weight": float(anchor_top1_weight),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    candidate_count = len(candidate_ids)
    summary = {
        "variant": variant_name,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "anchor_top1_weight": float(anchor_top1_weight),
        "images": int(len(frame)),
        "top1": float((frame["rank"] <= 1).mean()),
        "top5": float((frame["rank"] <= min(5, candidate_count)).mean()),
        "category_top1": float((frame["target_category"] == frame["top1_category"]).mean()),
        "median_rank": float(frame["rank"].median()),
        "mean_rank": float(frame["rank"].mean()),
        "l1": float(frame["l1"].mean()),
        "mse": float(frame["mse"].mean()),
        "psnr": float(frame["psnr"].mean()),
        "ssim": float(frame["ssim"].mean()),
        "is_oracle": bool(variant_name.startswith("true_category_oracle")),
        "is_pixel_prototype": bool(top_k > 1 and anchor_top1_weight < 1.0),
    }
    return frame, summary, generated


def candidate_table_for_split(frame):
    return frame[["image_id", "image_category"]].drop_duplicates().sort_values("image_id").reset_index(drop=True)


def build_selected_score_variants(
    frame,
    retrieval_model,
    retrieval_checkpoint,
    embedding_matrix,
    image_to_index,
    category_lookup,
    labels,
    eegnet_model,
    eegnet_labels,
    batch_size,
    keep_methods,
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
        batch_size,
    )
    if target_ids != candidate_ids:
        raise ValueError("Target image order does not match candidate order")

    eegnet_probs = None
    if eegnet_model is not None:
        prob_ids, probs = predict_eegnet_category_probs(frame, eegnet_model, eegnet_labels, batch_size)
        if prob_ids != target_ids:
            raise ValueError("EEGNet probability order does not match target order")
        if list(eegnet_labels) != list(labels):
            raise ValueError("EEGNet labels differ from embedding labels")
        eegnet_probs = probs

    base_scores = {
        "raw_cosine": sample_mean_scores,
        "embedding_mean_cosine": embedding_scores,
        "candidate_zscore": zscore_columns(sample_mean_scores),
    }
    all_variants, candidate_label_indices, _ = build_score_variants(
        base_scores,
        candidate_categories,
        labels,
        eegnet_probs=eegnet_probs,
    )
    add_true_category_oracle(
        all_variants,
        target_ids,
        candidate_categories,
        category_lookup,
        labels,
        candidate_label_indices,
    )
    variants = {name: all_variants[name] for name in keep_methods if name in all_variants}
    missing = sorted(set(keep_methods) - set(variants))
    if missing:
        raise KeyError(f"Requested score methods are missing: {missing}")
    return target_ids, candidate_ids, variants, repetitions


def save_generated_images(generated, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, array in generated.items():
        image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
        image.save(output_dir / f"{image_id}_prototype.png")


def make_grid(frame, generated, image_arrays, output_path, title, rows=8, image_size=190):
    selected = frame.sort_values(["ssim", "rank"], ascending=[False, True]).head(rows).reset_index(drop=True)
    gap = 14
    label_h = 68
    header_h = 44
    canvas = Image.new(
        "RGB",
        (2 * image_size + gap + 30, header_h + len(selected) * (image_size + label_h + gap) + 10),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    small_font = load_font(12)
    draw.text((15, 12), title, fill="black", font=title_font)
    y = header_h
    for _, row in selected.iterrows():
        target = Image.fromarray((image_arrays[row["target_image_id"]] * 255).astype(np.uint8)).resize((image_size, image_size))
        predicted = Image.fromarray((generated[row["target_image_id"]] * 255).astype(np.uint8)).resize((image_size, image_size))
        canvas.paste(target, (15, y))
        canvas.paste(predicted, (15 + image_size + gap, y))
        color = "#2e7d32" if row["target_category"] == row["top1_category"] else "#c62828"
        draw.rectangle((15, y, 15 + image_size, y + image_size), outline="#2e7d32", width=3)
        draw.rectangle((15 + image_size + gap, y, 15 + 2 * image_size + gap, y + image_size), outline=color, width=3)
        draw.text((15, y + image_size + 5), f"CEL {row['target_category']}: {row['target_image_id'][:28]}", fill="#333", font=small_font)
        draw.text((15 + image_size + gap, y + image_size + 5), f"PROTOTYP top1 {row['top1_category']}", fill=color, font=small_font)
        draw.text((15 + image_size + gap, y + image_size + 25), f"k={int(row['top_k'])} T={row['temperature']:.2g} SSIM={row['ssim']:.3f}", fill="#0d47a1", font=small_font)
        y += image_size + label_h + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def run_split(split_name, frame, common, args):
    target_ids, candidate_ids, score_variants, _ = build_selected_score_variants(
        frame=frame,
        retrieval_model=common["retrieval_model"],
        retrieval_checkpoint=common["retrieval_checkpoint"],
        embedding_matrix=common["embedding_matrix"],
        image_to_index=common["image_to_index"],
        category_lookup=common["category_lookup"],
        labels=common["labels"],
        eegnet_model=common["eegnet_model"],
        eegnet_labels=common["eegnet_labels"],
        batch_size=args.batch_size,
        keep_methods=args.score_methods,
    )
    image_arrays = load_image_arrays(candidate_ids, common["image_lookup"], args.metric_size)

    summaries = []
    frames = {}
    generated_by_key = {}
    for variant_name, scores in score_variants.items():
        for top_k in args.top_k:
            for temperature in args.temperatures:
                for anchor in args.anchor_top1_weights:
                    if top_k == 1 and (temperature != args.temperatures[0] or anchor != 0.0):
                        continue
                    key = f"{variant_name}_k{top_k}_t{temperature:g}_anchor{anchor:g}"
                    frame_out, summary, generated = evaluate_prototype_variant(
                        variant_name,
                        scores,
                        target_ids,
                        candidate_ids,
                        image_arrays,
                        common["category_lookup"],
                        top_k,
                        temperature,
                        anchor,
                    )
                    frame_out["prototype_key"] = key
                    summary["prototype_key"] = key
                    summaries.append(summary)
                    frames[key] = frame_out
                    generated_by_key[key] = generated

    split_dir = Path(args.output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["is_oracle", "is_pixel_prototype", "ssim"],
        ascending=[True, False, False],
    )
    summary_df.to_csv(split_dir / "prototype_method_comparison.csv", index=False)
    for key, frame_out in frames.items():
        frame_out.to_csv(split_dir / f"{key}_metrics.csv", index=False)
    return {
        "split": split_name,
        "summary_df": summary_df,
        "frames": frames,
        "generated": generated_by_key,
        "image_arrays": image_arrays,
    }


def main(args):
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    if args.force and output_dir.exists():
        resolved_output = output_dir.resolve()
        if not resolved_output.is_relative_to(project_root):
            raise ValueError(f"Refusing to remove output outside project root: {resolved_output}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = resolve_path(args.manifest_dir, project_root, "test.csv")
    embedding_dir = resolve_path(args.embedding_dir, project_root, "image_embeddings.npy")
    embedding_matrix, embedding_index, image_to_index, category_lookup, image_lookup = load_embedding_artifacts(
        embedding_dir, project_root
    )
    labels = sorted(embedding_index["image_category"].astype(str).unique().tolist())
    retrieval_model, retrieval_checkpoint = load_retrieval_model(args.retrieval_checkpoint, embedding_matrix.shape[1])
    eegnet_model, eegnet_checkpoint = load_eegnet_model(args.eegnet_checkpoint)

    common = {
        "embedding_matrix": embedding_matrix,
        "image_to_index": image_to_index,
        "category_lookup": category_lookup,
        "image_lookup": image_lookup,
        "labels": labels,
        "retrieval_model": retrieval_model,
        "retrieval_checkpoint": retrieval_checkpoint,
        "eegnet_model": eegnet_model,
        "eegnet_labels": eegnet_checkpoint["labels"],
    }
    frames = {
        split: resolve_manifest_epoch_paths(pd.read_csv(manifest_dir / f"{split}.csv"), project_root)
        for split in ["validation", "test"]
    }

    validation = run_split("validation", frames["validation"], common, args)
    test = run_split("test", frames["test"], common, args)

    validation_candidates = validation["summary_df"][
        (~validation["summary_df"]["is_oracle"]) & (validation["summary_df"]["is_pixel_prototype"])
    ].copy()
    if validation_candidates.empty:
        raise RuntimeError("No non-oracle pixel prototype variants were evaluated")
    selected_validation = validation_candidates.sort_values(
        ["ssim", "l1"], ascending=[False, True]
    ).iloc[0]
    selected_key = str(selected_validation["prototype_key"])
    selected_test = test["summary_df"][test["summary_df"]["prototype_key"] == selected_key].iloc[0]

    baseline_temperature = f"{args.temperatures[0]:g}"
    baseline_keys = [
        key
        for key in [
            f"eegnet_top1_category_gate_k1_t{baseline_temperature}_anchor0",
            f"candidate_zscore_k1_t{baseline_temperature}_anchor0",
            f"true_category_oracle_candidate_zscore_k1_t{baseline_temperature}_anchor0",
        ]
        if key in test["frames"]
    ]
    test_best_pixel = test["summary_df"][
        (~test["summary_df"]["is_oracle"]) & (test["summary_df"]["is_pixel_prototype"])
    ].sort_values(["ssim", "l1"], ascending=[False, True]).iloc[0]

    comparison_rows = [
        {"selection": "validation_selected_pixel_prototype", **selected_validation.to_dict()},
        {"selection": "test_selected_by_validation", **selected_test.to_dict()},
        {"selection": "test_best_pixel_posthoc", **test_best_pixel.to_dict()},
    ]
    for key in baseline_keys:
        row = test["summary_df"][test["summary_df"]["prototype_key"] == key].iloc[0]
        comparison_rows.append({"selection": f"test_baseline_{key}", **row.to_dict()})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "prototype_selected_comparison.csv", index=False)

    generated_dir = output_dir / "generated" / selected_key
    save_generated_images(test["generated"][selected_key], generated_dir)
    grid_dir = output_dir / "grids"
    make_grid(
        test["frames"][selected_key],
        test["generated"][selected_key],
        test["image_arrays"],
        grid_dir / f"{selected_key}_best.jpg",
        f"Wybrany prototyp: {selected_key}",
        rows=args.grid_rows,
        image_size=args.image_size,
    )
    for key in baseline_keys[:2]:
        make_grid(
            test["frames"][key],
            test["generated"][key],
            test["image_arrays"],
            grid_dir / f"{key}_best.jpg",
            f"Baseline: {key}",
            rows=args.grid_rows,
            image_size=args.image_size,
        )

    summary = {
        "experiment": "category-aware top-k pixel prototype reconstruction",
        "interpretation": (
            "Tests whether moving from a single selected candidate to a weighted top-k "
            "pixel prototype improves candidate-constrained EEG image reconstruction."
        ),
        "manifest_dir": str(manifest_dir),
        "embedding_dir": str(embedding_dir),
        "selection_rule": "best non-oracle pixel prototype on validation SSIM, tie-broken by L1",
        "selected_key": selected_key,
        "validation_selected": selected_validation.to_dict(),
        "test_selected_by_validation": selected_test.to_dict(),
        "test_best_pixel_posthoc": test_best_pixel.to_dict(),
        "baseline_rows": comparison_rows[3:],
    }
    with (output_dir / "prototype_reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved prototype reconstruction outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local top-k pixel prototype reconstruction after category-aware reranking."
    )
    parser.add_argument("--manifest-dir", default="reconstruction_manifests/participant_image_mole_no_abc")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--embedding-dir", default="image_embeddings_unclip_participant_image_mole_no_abc_local_20260627")
    parser.add_argument("--retrieval-checkpoint", default="wyniki colab/unclip_mole_retrieval/eeg_image_retrieval.pt")
    parser.add_argument("--eegnet-checkpoint", default="wyniki colab/eegnet_mole_colab/eegnet.pt")
    parser.add_argument("--output-dir", default="wyniki colab/category_prototype_reconstruction_mole_local")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--metric-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=190)
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument(
        "--score-methods",
        nargs="+",
        default=[
            "candidate_zscore",
            "eegnet_top1_category_gate",
            "true_category_oracle_candidate_zscore",
        ],
        help="Score variants to prototype. Keep this short; image metrics are the costly part.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.5])
    parser.add_argument("--anchor-top1-weights", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
