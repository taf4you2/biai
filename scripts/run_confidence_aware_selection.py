import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


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


def load_candidate_frames(category_dir, prototype_dir, split, prototype_key):
    category_dir = Path(category_dir)
    prototype_dir = Path(prototype_dir)
    frames = []
    nn_methods = {
        "nn_candidate_zscore": "candidate_zscore",
        "nn_eegnet_top1_gate": "eegnet_top1_category_gate",
        "nn_eegnet_zlogprob_1p5": "eegnet_zlogprob_alpha_1.5",
        "nn_eegnet_zlogprob_2": "eegnet_zlogprob_alpha_2",
    }
    for method_key, filename_stem in nn_methods.items():
        path = category_dir / split / f"{filename_stem}_reconstructions.csv"
        if path.is_file():
            frames.append(normalize_nn_frame(pd.read_csv(path), method_key))

    prototype_path = prototype_dir / split / f"{prototype_key}_metrics.csv"
    if prototype_path.is_file():
        frames.append(normalize_prototype_frame(pd.read_csv(prototype_path), f"prototype_{prototype_key}"))

    if not frames:
        raise FileNotFoundError(f"No candidate frames found for split {split!r}")
    combined = pd.concat(frames, ignore_index=True)
    combined["target_image_id"] = combined["target_image_id"].astype(str)
    combined["target_category"] = combined["target_category"].astype(str)
    combined["predicted_category"] = combined["predicted_category"].astype(str)
    return combined


def summarize_selection(name, frame, candidate_count=44):
    return {
        "selector": name,
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
        "methods_used": {
            str(key): int(value)
            for key, value in frame["method_key"].value_counts().sort_index().items()
        },
    }


def gate_category_map(frame):
    gate = frame[frame["method_key"] == "nn_eegnet_top1_gate"][
        ["target_image_id", "predicted_category"]
    ].drop_duplicates("target_image_id")
    return dict(zip(gate["target_image_id"].astype(str), gate["predicted_category"].astype(str)))


def select_global_best(validation, test):
    means = validation.groupby("method_key")["ssim"].mean().sort_values(ascending=False)
    method = str(means.index[0])
    selected = test[test["method_key"] == method].copy()
    selected["selector_reason"] = f"global validation best: {method}"
    return method, selected


def select_by_gate_category(validation, test, min_count):
    gate_val = gate_category_map(validation)
    gate_test = gate_category_map(test)
    validation = validation.copy()
    test = test.copy()
    validation["gate_category"] = validation["target_image_id"].map(gate_val)
    test["gate_category"] = test["target_image_id"].map(gate_test)

    global_method = validation.groupby("method_key")["ssim"].mean().sort_values(ascending=False).index[0]
    category_methods = {}
    for category, group in validation.groupby("gate_category"):
        by_method = (
            group.groupby("method_key")
            .agg(mean_ssim=("ssim", "mean"), count=("ssim", "size"))
            .sort_values("mean_ssim", ascending=False)
        )
        eligible = by_method[by_method["count"] >= min_count]
        if eligible.empty:
            category_methods[str(category)] = str(global_method)
        else:
            category_methods[str(category)] = str(eligible.index[0])

    rows = []
    for target_id, group in test.groupby("target_image_id", sort=True):
        gate_category = str(group["gate_category"].iloc[0])
        method = category_methods.get(gate_category, str(global_method))
        chosen = group[group["method_key"] == method]
        if chosen.empty:
            chosen = group[group["method_key"] == global_method]
        row = chosen.iloc[0].copy()
        row["selector_reason"] = f"gate_category={gate_category}; method={method}"
        rows.append(row)
    selected = pd.DataFrame(rows).reset_index(drop=True)
    return category_methods, selected


def select_oracle_upper_bound(test):
    rows = []
    for _, group in test.groupby("target_image_id", sort=True):
        rows.append(group.sort_values(["ssim", "l1"], ascending=[False, True]).iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected["selector_reason"] = "oracle per-image best non-oracle candidate"
    return selected


def make_grid(frame, output_path, image_lookup, generated_root, title, rows=8, image_size=190):
    selected = frame.sort_values(["ssim", "rank"], ascending=[False, True]).head(rows).reset_index(drop=True)
    gap = 14
    label_h = 76
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
        draw.text((15 + image_size + gap, y + image_size + 25), f"rank={int(row['rank'])} SSIM={row['ssim']:.3f}", fill="#0d47a1", font=small_font)
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

    validation = load_candidate_frames(args.category_dir, args.prototype_dir, "validation", args.prototype_key)
    test = load_candidate_frames(args.category_dir, args.prototype_dir, "test", args.prototype_key)

    global_method, global_selected = select_global_best(validation, test)
    category_methods, category_selected = select_by_gate_category(validation, test, args.min_category_count)
    oracle_selected = select_oracle_upper_bound(test)

    selectors = {
        "global_validation_best": global_selected,
        "gate_category_validation_selector": category_selected,
        "oracle_per_image_upper_bound": oracle_selected,
    }
    summaries = []
    for name, frame in selectors.items():
        frame.to_csv(output_dir / f"{name}_selected_rows.csv", index=False)
        summaries.append(summarize_selection(name, frame))
    summary_df = pd.DataFrame(summaries).sort_values("ssim", ascending=False)
    summary_df.to_csv(output_dir / "selector_comparison.csv", index=False)

    image_lookup = load_image_lookup(args.embedding_dir, project_root)
    generated_root = Path(args.prototype_dir) / "generated"
    grid_dir = output_dir / "grids"
    for name, frame in selectors.items():
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
        "experiment": "validation-calibrated confidence/category selector",
        "interpretation": (
            "Tests whether validation can choose between nearest-neighbor and top-k prototype "
            "reconstructions using only observable gate categories."
        ),
        "category_dir": str(args.category_dir),
        "prototype_dir": str(args.prototype_dir),
        "prototype_key": str(args.prototype_key),
        "global_validation_best_method": str(global_method),
        "gate_category_methods": category_methods,
        "selectors": summaries,
    }
    with (output_dir / "confidence_selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved confidence-aware selection outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-calibrated selector over category-aware NN and prototype reconstructions."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--category-dir", default="wyniki colab/category_aware_reranking_mole_local")
    parser.add_argument("--prototype-dir", default="wyniki colab/category_prototype_reconstruction_mole_local")
    parser.add_argument("--prototype-key", default="eegnet_top1_category_gate_k3_t0.5_anchor0.5")
    parser.add_argument("--embedding-dir", default="image_embeddings_unclip_participant_image_mole_no_abc_local_20260627")
    parser.add_argument("--output-dir", default="wyniki colab/confidence_aware_selection_mole_local")
    parser.add_argument("--min-category-count", type=int, default=2)
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=190)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
