import argparse
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_category_aware_reranking import (  # noqa: E402
    build_score_variants,
    load_eegnet_model,
    load_embedding_artifacts,
    load_retrieval_model,
    predict_eegnet_category_probs,
    predict_retrieval_scores,
    resolve_manifest_epoch_paths,
    resolve_path,
    zscore_columns,
)


METHOD_TO_SCORE = {
    "nn_candidate_zscore": "candidate_zscore",
    "nn_eegnet_top1_gate": "eegnet_top1_category_gate",
    "nn_eegnet_zlogprob_1p5": "eegnet_zlogprob_alpha_1.5",
    "nn_eegnet_zlogprob_2": "eegnet_zlogprob_alpha_2",
}


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
        return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def load_image_lookup(embedding_dir, project_root):
    index = pd.read_csv(Path(embedding_dir) / "image_embedding_index.csv")
    return {
        str(row.image_id): resolve_image_path(row.image_path, project_root)
        for row in index.itertuples(index=False)
    }


def normalize_nn_frame(frame, method_key):
    out = frame.copy()
    out["source_type"] = "nearest_neighbor"
    out["method_key"] = method_key
    out["display_prediction_id"] = out["predicted_image_id"].astype(str)
    return out[
        [
            "method_key",
            "source_type",
            "target_image_id",
            "display_prediction_id",
            "target_category",
            "predicted_category",
            "rank",
            "l1",
            "mse",
            "psnr",
            "ssim",
        ]
    ]


def normalize_prototype_frame(frame, method_key):
    out = frame.copy()
    out["source_type"] = "pixel_prototype"
    out["method_key"] = method_key
    out["display_prediction_id"] = out["prototype_key"].astype(str)
    out["predicted_category"] = out["top1_category"].astype(str)
    return out[
        [
            "method_key",
            "source_type",
            "target_image_id",
            "display_prediction_id",
            "target_category",
            "predicted_category",
            "rank",
            "l1",
            "mse",
            "psnr",
            "ssim",
        ]
    ]


def load_outcome_table(category_dir, prototype_dir, split, prototype_key):
    category_dir = Path(category_dir)
    prototype_dir = Path(prototype_dir)
    frames = []
    nn_files = {
        "nn_candidate_zscore": "candidate_zscore",
        "nn_eegnet_top1_gate": "eegnet_top1_category_gate",
        "nn_eegnet_zlogprob_1p5": "eegnet_zlogprob_alpha_1.5",
        "nn_eegnet_zlogprob_2": "eegnet_zlogprob_alpha_2",
    }
    for method_key, stem in nn_files.items():
        path = category_dir / split / f"{stem}_reconstructions.csv"
        if path.is_file():
            frames.append(normalize_nn_frame(pd.read_csv(path), method_key))

    prototype_path = prototype_dir / split / f"{prototype_key}_metrics.csv"
    if prototype_path.is_file():
        frames.append(normalize_prototype_frame(pd.read_csv(prototype_path), f"prototype_{prototype_key}"))

    if not frames:
        raise FileNotFoundError(f"No outcome files found for split {split!r}")
    out = pd.concat(frames, ignore_index=True)
    out["split"] = split
    out["target_image_id"] = out["target_image_id"].astype(str)
    return out


def candidate_table_for_split(frame):
    return frame[["image_id", "image_category"]].drop_duplicates().sort_values("image_id").reset_index(drop=True)


def stable_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    weights = np.exp(values)
    return weights / np.maximum(weights.sum(), 1e-12)


def entropy(weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[weights > 0]
    if len(weights) == 0:
        return 0.0
    return float(-(weights * np.log(weights)).sum())


def build_split_score_context(
    split_name,
    frame,
    common,
    batch_size,
    prototype_key,
):
    candidate_table = candidate_table_for_split(frame)
    candidate_ids = candidate_table["image_id"].astype(str).tolist()
    candidate_categories = candidate_table["image_category"].astype(str).tolist()
    labels = common["labels"]
    category_to_index = {label: index for index, label in enumerate(labels)}

    target_ids, sample_mean_scores, embedding_scores, repetitions = predict_retrieval_scores(
        frame,
        common["retrieval_model"],
        common["retrieval_checkpoint"],
        common["embedding_matrix"],
        common["image_to_index"],
        candidate_ids,
        batch_size,
    )
    if target_ids != candidate_ids:
        raise ValueError(f"{split_name}: target order does not match candidate order")

    prob_ids, eegnet_probs = predict_eegnet_category_probs(
        frame,
        common["eegnet_model"],
        common["eegnet_labels"],
        batch_size,
    )
    if prob_ids != target_ids:
        raise ValueError(f"{split_name}: EEGNet probability order does not match targets")

    base_scores = {
        "raw_cosine": sample_mean_scores,
        "embedding_mean_cosine": embedding_scores,
        "candidate_zscore": zscore_columns(sample_mean_scores),
    }
    variants, _, _ = build_score_variants(
        base_scores,
        candidate_categories,
        labels,
        eegnet_probs=eegnet_probs,
    )

    prototype_method_key = f"prototype_{prototype_key}"
    method_to_score = dict(METHOD_TO_SCORE)
    method_to_score[prototype_method_key] = "eegnet_top1_category_gate"
    method_to_source = {
        **{key: "nearest_neighbor" for key in METHOD_TO_SCORE},
        prototype_method_key: "pixel_prototype",
    }

    top1_by_method = {}
    for method_key, score_key in method_to_score.items():
        scores = variants[score_key]
        order = np.argsort(-scores, axis=1)
        top1_by_method[method_key] = [candidate_ids[index] for index in order[:, 0]]

    hubness_by_method = {
        method: pd.Series(ids).value_counts().to_dict()
        for method, ids in top1_by_method.items()
    }

    rows = []
    for row_index, target_id in enumerate(target_ids):
        eeg_probs = eegnet_probs[row_index]
        eeg_order = np.argsort(-eeg_probs)
        eeg_top1_index = int(eeg_order[0])
        eeg_top2_index = int(eeg_order[1])
        eeg_top1_category = labels[eeg_top1_index]
        eeg_top1_prob = float(eeg_probs[eeg_top1_index])
        eeg_top2_prob = float(eeg_probs[eeg_top2_index])
        eeg_entropy = entropy(eeg_probs) / np.log(len(eeg_probs))

        for method_key, score_key in method_to_score.items():
            scores = variants[score_key][row_index]
            finite = scores > -1e8
            finite_scores = scores[finite]
            finite_candidate_indices = np.where(finite)[0]
            order_finite = finite_candidate_indices[np.argsort(-finite_scores)]
            top1_pos = int(order_finite[0])
            top2_pos = int(order_finite[1]) if len(order_finite) > 1 else top1_pos
            top5_positions = order_finite[: min(5, len(order_finite))]
            top1_id = candidate_ids[top1_pos]
            top1_category = candidate_categories[top1_pos]
            top1_category_index = category_to_index[top1_category]
            sorted_scores = scores[order_finite]
            probs = stable_softmax(sorted_scores[: min(10, len(sorted_scores))])
            row = {
                "split": split_name,
                "target_image_id": target_id,
                "method_key": method_key,
                "source_type": method_to_source[method_key],
                "score_key": score_key,
                "repetitions": int(repetitions[row_index]),
                "candidate_count_allowed": int(finite.sum()),
                "score_top1": float(sorted_scores[0]),
                "score_top2": float(sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0]),
                "score_top5_mean": float(sorted_scores[: min(5, len(sorted_scores))].mean()),
                "score_std_allowed": float(finite_scores.std()) if len(finite_scores) > 1 else 0.0,
                "score_margin_12": float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0,
                "score_margin_15": float(sorted_scores[0] - sorted_scores[min(4, len(sorted_scores) - 1)]),
                "score_entropy_top10": entropy(probs) / np.log(len(probs)) if len(probs) > 1 else 0.0,
                "top1_candidate_hubness_count": int(hubness_by_method[method_key].get(top1_id, 0)),
                "top1_candidate_hubness_share": float(hubness_by_method[method_key].get(top1_id, 0) / len(target_ids)),
                "top1_candidate_category": top1_category,
                "top1_matches_eegnet_category": float(top1_category == eeg_top1_category),
                "top5_contains_eegnet_category": float(
                    any(candidate_categories[pos] == eeg_top1_category for pos in top5_positions)
                ),
                "eegnet_top1_category": eeg_top1_category,
                "eegnet_top1_prob": eeg_top1_prob,
                "eegnet_top2_prob": eeg_top2_prob,
                "eegnet_margin_12": float(eeg_top1_prob - eeg_top2_prob),
                "eegnet_entropy": eeg_entropy,
                "eegnet_prob_for_top1_candidate_category": float(eeg_probs[top1_category_index]),
                "is_prototype": float(method_to_source[method_key] == "pixel_prototype"),
                "is_nn": float(method_to_source[method_key] == "nearest_neighbor"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def one_hot_features(train_frame, test_frame, categorical_columns):
    combined = pd.concat([train_frame, test_frame], ignore_index=True)
    encoded = pd.get_dummies(combined, columns=categorical_columns, dtype=float)
    for column in categorical_columns:
        encoded[column] = combined[column].to_numpy()
    train_encoded = encoded.iloc[: len(train_frame)].reset_index(drop=True)
    test_encoded = encoded.iloc[len(train_frame) :].reset_index(drop=True)
    return train_encoded, test_encoded


def feature_columns(frame):
    blocked = {
        "split",
        "target_image_id",
        "target_category",
        "predicted_category",
        "display_prediction_id",
        "source_type",
        "method_key",
        "score_key",
        "ssim",
        "l1",
        "mse",
        "psnr",
        "rank",
    }
    return [
        column
        for column in frame.columns
        if column not in blocked and pd.api.types.is_numeric_dtype(frame[column])
    ]


def selected_rows_from_predictions(candidate_frame, predictions, selector_name):
    frame = candidate_frame.copy()
    frame["predicted_ssim"] = predictions
    rows = []
    for _, group in frame.groupby("target_image_id", sort=True):
        rows.append(group.sort_values("predicted_ssim", ascending=False).iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected["selector"] = selector_name
    return selected


def summarize_selection(selector, frame, candidate_count=44):
    return {
        "selector": selector,
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
        "predicted_ssim_mean": float(frame["predicted_ssim"].mean()) if "predicted_ssim" in frame else None,
        "methods_used": {
            str(key): int(value)
            for key, value in frame["method_key"].value_counts().sort_index().items()
        },
    }


def global_best(validation_candidates, test_candidates):
    method = validation_candidates.groupby("method_key")["ssim"].mean().sort_values(ascending=False).index[0]
    selected = test_candidates[test_candidates["method_key"] == method].copy().reset_index(drop=True)
    selected["selector"] = f"global_validation_best:{method}"
    selected["predicted_ssim"] = float(validation_candidates[validation_candidates["method_key"] == method]["ssim"].mean())
    return str(method), selected


def oracle_best(test_candidates):
    rows = []
    for _, group in test_candidates.groupby("target_image_id", sort=True):
        rows.append(group.sort_values(["ssim", "l1"], ascending=[False, True]).iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected["selector"] = "oracle_per_image_best"
    selected["predicted_ssim"] = selected["ssim"]
    return selected


def cross_validate_alphas(train_encoded, feature_cols, alphas):
    groups = train_encoded["target_image_id"].astype(str).to_numpy()
    y = train_encoded["ssim"].to_numpy(dtype=float)
    logo = LeaveOneGroupOut()
    rows = []
    for alpha in alphas:
        predictions = np.zeros(len(train_encoded), dtype=float)
        for train_idx, val_idx in logo.split(train_encoded[feature_cols], y, groups):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train_encoded.iloc[train_idx][feature_cols], y[train_idx])
            predictions[val_idx] = model.predict(train_encoded.iloc[val_idx][feature_cols])
        selected = selected_rows_from_predictions(train_encoded, predictions, f"ridge_cv_alpha_{alpha:g}")
        rows.append(
            {
                "alpha": float(alpha),
                "cv_selected_ssim": float(selected["ssim"].mean()),
                "cv_selected_l1": float(selected["l1"].mean()),
                "cv_prediction_rmse": float(mean_squared_error(y, predictions) ** 0.5),
                "methods_used": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in selected["method_key"].value_counts().sort_index().items()
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["cv_selected_ssim", "cv_selected_l1"], ascending=[False, True])


def make_grid(frame, output_path, image_lookup, generated_root, title, rows=8, image_size=190):
    selected = frame.sort_values(["ssim", "rank"], ascending=[False, True]).head(rows).reset_index(drop=True)
    gap = 14
    label_h = 78
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
        target_id = str(row["target_image_id"])
        target = fit_image(image_lookup[target_id], image_size)
        if row["source_type"] == "nearest_neighbor":
            predicted = fit_image(image_lookup[str(row["display_prediction_id"])], image_size)
        else:
            generated_path = generated_root / str(row["display_prediction_id"]) / f"{target_id}_prototype.png"
            predicted = fit_image(generated_path, image_size)
        canvas.paste(target, (15, y))
        canvas.paste(predicted, (15 + image_size + gap, y))
        color = "#2e7d32" if row["target_category"] == row["predicted_category"] else "#c62828"
        draw.rectangle((15, y, 15 + image_size, y + image_size), outline="#2e7d32", width=3)
        draw.rectangle((15 + image_size + gap, y, 15 + 2 * image_size + gap, y + image_size), outline=color, width=3)
        draw.text((15, y + image_size + 5), f"CEL {row['target_category']}: {target_id[:28]}", fill="#333", font=small_font)
        draw.text((15 + image_size + gap, y + image_size + 5), f"{row['method_key'][:34]}", fill=color, font=small_font)
        draw.text(
            (15 + image_size + gap, y + image_size + 25),
            f"SSIM={row['ssim']:.3f} pred={row.get('predicted_ssim', 0):.3f}",
            fill="#0d47a1",
            font=small_font,
        )
        y += image_size + label_h + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


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
    feature_frames = {
        split: build_split_score_context(
            split,
            frames[split],
            common,
            args.batch_size,
            args.prototype_key,
        )
        for split in ["validation", "test"]
    }
    outcome_frames = {
        split: load_outcome_table(args.category_dir, args.prototype_dir, split, args.prototype_key)
        for split in ["validation", "test"]
    }
    merged = {}
    for split in ["validation", "test"]:
        merged[split] = outcome_frames[split].merge(
            feature_frames[split],
            on=["split", "target_image_id", "method_key", "source_type"],
            how="inner",
            validate="one_to_one",
        )
        merged[split].to_csv(output_dir / f"{split}_confidence_features.csv", index=False)

    categorical = ["method_key", "source_type", "score_key", "top1_candidate_category", "eegnet_top1_category"]
    validation_encoded, test_encoded = one_hot_features(merged["validation"], merged["test"], categorical)
    feature_cols = feature_columns(validation_encoded)

    alphas = [float(value) for value in args.alphas]
    cv = cross_validate_alphas(validation_encoded, feature_cols, alphas)
    cv.to_csv(output_dir / "validation_leave_one_image_cv.csv", index=False)
    best_alpha = float(cv.iloc[0]["alpha"])

    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    model.fit(validation_encoded[feature_cols], validation_encoded["ssim"].to_numpy(dtype=float))
    test_predictions = model.predict(test_encoded[feature_cols])
    selected = selected_rows_from_predictions(test_encoded, test_predictions, f"ridge_feature_selector_alpha_{best_alpha:g}")
    selected.to_csv(output_dir / "ridge_feature_selector_selected_rows.csv", index=False)

    validation_nn = validation_encoded[validation_encoded["source_type"] == "nearest_neighbor"].reset_index(drop=True)
    test_nn = test_encoded[test_encoded["source_type"] == "nearest_neighbor"].reset_index(drop=True)
    cv_nn = cross_validate_alphas(validation_nn, feature_cols, alphas)
    cv_nn.to_csv(output_dir / "validation_leave_one_image_cv_nn_only.csv", index=False)
    best_alpha_nn = float(cv_nn.iloc[0]["alpha"])
    model_nn = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_nn))
    model_nn.fit(validation_nn[feature_cols], validation_nn["ssim"].to_numpy(dtype=float))
    test_predictions_nn = model_nn.predict(test_nn[feature_cols])
    selected_nn = selected_rows_from_predictions(
        test_nn,
        test_predictions_nn,
        f"ridge_feature_selector_nn_only_alpha_{best_alpha_nn:g}",
    )
    selected_nn.to_csv(output_dir / "ridge_feature_selector_nn_only_selected_rows.csv", index=False)

    global_method, global_selected = global_best(merged["validation"], test_encoded)
    oracle_selected = oracle_best(test_encoded)
    oracle_nn_selected = oracle_best(test_nn)

    selector_frames = {
        f"ridge_feature_selector_alpha_{best_alpha:g}": selected,
        f"ridge_feature_selector_nn_only_alpha_{best_alpha_nn:g}": selected_nn,
        "global_validation_best": global_selected,
        "oracle_nn_only_per_image_best": oracle_nn_selected,
        "oracle_per_image_best": oracle_selected,
    }
    summaries = [summarize_selection(name, frame) for name, frame in selector_frames.items()]
    summary_df = pd.DataFrame(summaries).sort_values("ssim", ascending=False)
    summary_df.to_csv(output_dir / "selector_comparison.csv", index=False)

    image_lookup = load_image_lookup(args.embedding_dir, project_root)
    generated_root = Path(args.prototype_dir) / "generated"
    grid_dir = output_dir / "grids"
    for name, frame in selector_frames.items():
        make_grid(
            frame,
            grid_dir / f"{name}_best.jpg",
            image_lookup,
            generated_root,
            f"{name} — najlepsze według SSIM",
            rows=args.grid_rows,
            image_size=args.image_size,
        )

    summary = {
        "experiment": "confidence feature selector",
        "interpretation": (
            "Learns from validation features such as score margins, EEGNet entropy, hubness "
            "and candidate agreement to choose the reconstruction variant per image."
        ),
        "manifest_dir": str(manifest_dir),
        "embedding_dir": str(embedding_dir),
        "prototype_key": str(args.prototype_key),
        "feature_columns": feature_cols,
        "best_alpha_by_validation_leave_one_image_cv": best_alpha,
        "validation_cv": cv.to_dict(orient="records"),
        "best_alpha_nn_only_by_validation_leave_one_image_cv": best_alpha_nn,
        "validation_cv_nn_only": cv_nn.to_dict(orient="records"),
        "global_validation_best_method": global_method,
        "selectors": summaries,
    }
    with (output_dir / "confidence_feature_selector_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved confidence feature selector outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a validation-calibrated confidence feature selector.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest-dir", default="reconstruction_manifests/participant_image_mole_no_abc")
    parser.add_argument("--embedding-dir", default="image_embeddings_unclip_participant_image_mole_no_abc_local_20260627")
    parser.add_argument("--retrieval-checkpoint", default="wyniki colab/unclip_mole_retrieval/eeg_image_retrieval.pt")
    parser.add_argument("--eegnet-checkpoint", default="wyniki colab/eegnet_mole_colab/eegnet.pt")
    parser.add_argument("--category-dir", default="wyniki colab/category_aware_reranking_mole_local")
    parser.add_argument("--prototype-dir", default="wyniki colab/category_prototype_reconstruction_mole_local")
    parser.add_argument("--prototype-key", default="eegnet_top1_category_gate_k3_t0.5_anchor0.5")
    parser.add_argument("--output-dir", default="wyniki colab/confidence_feature_selector_mole_local")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=190)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
