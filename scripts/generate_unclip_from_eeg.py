import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eeg_image_retrieval import (
    EEGEmbeddingNet,
    EEGImageDataset,
    resolve_manifest_epoch_paths,
)


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


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


def image_metrics(generated, target):
    predicted = TF.to_tensor(generated).unsqueeze(0)
    reference = TF.to_tensor(target).unsqueeze(0)
    mse = (predicted - reference).square().mean()
    return {
        "l1": float((predicted - reference).abs().mean()),
        "mse": float(mse),
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-10))),
        "ssim": float(batch_ssim(predicted, reference)[0]),
    }


def load_model(checkpoint, embedding_dim):
    model_args = checkpoint["args"]
    _, channels, samples = checkpoint["input_shape"]
    model = EEGEmbeddingNet(
        channels,
        samples,
        embedding_dim,
        f1=model_args["f1"],
        depth_multiplier=model_args["depth_multiplier"],
        f2=model_args["f2"],
        kernel_length=model_args["kernel_length"],
        dropout=model_args["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_test_predictions(checkpoint, embeddings, index, project_root):
    retrieval_args = checkpoint["args"]
    manifest_dir = Path(retrieval_args["manifest_dir"])
    test_frame = resolve_manifest_epoch_paths(
        pd.read_csv(manifest_dir / "test.csv"),
        project_root,
    ).reset_index(drop=True)
    image_to_index = dict(zip(index.image_id.astype(str), index.embedding_index.astype(int)))
    dataset = EEGImageDataset(
        test_frame,
        image_to_index,
        image_to_index,
        embeddings,
        checkpoint["mean"],
        checkpoint["std"],
        preload=True,
    )
    model = load_model(checkpoint, embeddings.shape[1])
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    predicted = []
    ids = []
    with torch.inference_mode():
        for eeg, _, batch_ids in loader:
            predicted.append(model(eeg).cpu().numpy())
            ids.extend(batch_ids)
    predicted = np.concatenate(predicted)
    frame = pd.DataFrame({"image_id": ids})
    unique = test_frame[["image_id", "image_category"]].drop_duplicates("image_id").sort_values("image_id")
    averaged = []
    rows = []
    for _, row in unique.iterrows():
        indices = frame.index[frame.image_id == row.image_id].tolist()
        vector = predicted[indices].mean(axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        averaged.append(vector)
        rows.append(
            {
                "image_id": str(row.image_id),
                "image_category": str(row.image_category),
                "repetitions": len(indices),
            }
        )
    return pd.DataFrame(rows), np.stack(averaged), image_to_index


def make_grid(rows, image_lookup, generated_dir, output_path, image_key, rows_count=6, size=192):
    selected = rows.sort_values(f"{image_key}_ssim", ascending=False).head(rows_count)
    width = 30 + 2 * size + 16
    height = 45 + len(selected) * (size + 48)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), f"CEL | {image_key.upper()} -> Stable UnCLIP", fill="black", font=font(18))
    y = 45
    for _, row in selected.iterrows():
        with Image.open(image_lookup[row.image_id]) as image:
            target = ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
        with Image.open(generated_dir / f"{row.image_id}_{image_key}.png") as image:
            generated = ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
        canvas.paste(target, (15, y))
        canvas.paste(generated, (31 + size, y))
        draw.text((15, y + size + 4), f"{row.image_category} | {row.image_id}", fill="#444", font=font(12))
        draw.text((31 + size, y + size + 4), f"SSIM={row[f'{image_key}_ssim']:.3f}", fill="#0d47a1", font=font(12))
        y += size + 48
    canvas.save(output_path, quality=95)


def generate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir()

    checkpoint = torch.load(
        Path(args.retrieval_result_dir) / "eeg_image_retrieval.pt",
        map_location="cpu",
        weights_only=False,
    )
    retrieval_args = checkpoint["args"]
    embedding_dir = Path(retrieval_args["embedding_dir"])
    summary = json.loads((embedding_dir / "embedding_summary.json").read_text(encoding="utf-8"))
    if summary.get("pipeline") != "StableUnCLIPImg2ImgPipeline":
        raise ValueError("Embedding directory must come from extract_unclip_image_embeddings.py")
    raw_file = summary["raw_embeddings_file"]
    raw_embeddings = np.load(embedding_dir / raw_file).astype(np.float32)
    embeddings = np.load(embedding_dir / "image_embeddings.npy").astype(np.float32)
    index = pd.read_csv(embedding_dir / "image_embedding_index.csv")
    project_root = Path(retrieval_args.get("project_root", args.project_root)).resolve()
    rows, eeg_embeddings, image_to_index = load_test_predictions(
        checkpoint,
        embeddings,
        index,
        project_root,
    )
    if args.max_images is not None:
        rows = rows.head(args.max_images).reset_index(drop=True)
        eeg_embeddings = eeg_embeddings[: len(rows)]

    raw_scale = float(summary["raw_embedding_norm_mean"])
    eeg_raw = eeg_embeddings * raw_scale
    true_raw = raw_embeddings[[image_to_index[image_id] for image_id in rows.image_id]]
    image_lookup = dict(zip(index.image_id.astype(str), index.image_path.astype(str)))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("Stable UnCLIP requires a GPU in practice. Use --allow-cpu only for a smoke test.")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    result_rows = []
    for row_index, row in rows.iterrows():
        with Image.open(image_lookup[row.image_id]) as image:
            target_original = image.convert("RGB").copy()
        conditions = [("eeg", eeg_raw[row_index])]
        if args.oracle:
            conditions.append(("oracle", true_raw[row_index]))
        metrics = {"image_id": row.image_id, "image_category": row.image_category, "repetitions": int(row.repetitions)}
        for condition_name, condition in conditions:
            generator = torch.Generator(device=device).manual_seed(args.seed + row_index)
            output = pipe(
                image=None,
                image_embeds=torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=dtype),
                prompt=[""],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                noise_level=args.noise_level,
                generator=generator,
            )
            generated = output.images[0].convert("RGB")
            target = ImageOps.fit(target_original, generated.size, Image.Resampling.LANCZOS)
            generated.save(generated_dir / f"{row.image_id}_{condition_name}.png")
            metrics.update({f"{condition_name}_{key}": value for key, value in image_metrics(generated, target).items()})
        result_rows.append(metrics)
        print(f"generated {row_index + 1}/{len(rows)}: {row.image_id}")

    frame = pd.DataFrame(result_rows)
    frame.to_csv(output_dir / "unclip_metrics_per_image.csv", index=False)
    summary_out = {
        "retrieval_result_dir": args.retrieval_result_dir,
        "embedding_dir": str(embedding_dir),
        "model_id": args.model_id,
        "images": int(len(frame)),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "noise_level": args.noise_level,
        "raw_embedding_scale": raw_scale,
        "eeg": {key: float(frame[f"eeg_{key}"].mean()) for key in ("l1", "mse", "psnr", "ssim")},
    }
    if args.oracle:
        summary_out["oracle"] = {
            key: float(frame[f"oracle_{key}"].mean()) for key in ("l1", "mse", "psnr", "ssim")
        }
    (output_dir / "unclip_generation_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    grids_dir = output_dir / "grids"
    grids_dir.mkdir()
    make_grid(frame, image_lookup, generated_dir, grids_dir / "best_eeg_unclip.jpg", "eeg", args.grid_rows)
    if args.oracle:
        make_grid(frame, image_lookup, generated_dir, grids_dir / "best_oracle_unclip.jpg", "oracle", args.grid_rows)
    print(json.dumps(summary_out, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Stable UnCLIP images from EEG-predicted CLIP embeddings."
    )
    parser.add_argument("--retrieval-result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-id", default="diffusers/stable-diffusion-2-1-unclip-i2i-l")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--noise-level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
