import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(row, project_root):
    project_root = Path(project_root)
    raw_path = row.get("image_path")
    if isinstance(raw_path, str) and raw_path:
        normalized = raw_path.replace("\\", "/")
        candidates = [Path(normalized), project_root / Path(normalized)]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    image_file = row.get("image_file")
    if isinstance(image_file, str) and image_file:
        candidate = project_root / "images" / Path(image_file.replace("\\", "/"))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve image for {row.get('image_id')!r}")


def unique_images(manifest_dir, project_root):
    rows = []
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(Path(manifest_dir) / f"{split}.csv")
        needed = ["image_id", "image_category", "image_path", "image_file"]
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise ValueError(f"{split}.csv is missing columns: {missing}")
        frame = frame[needed].copy()
        frame["source_split"] = split
        rows.append(frame)

    images = pd.concat(rows, ignore_index=True)
    images = images.drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    images["image_path"] = [str(resolve_image_path(row, project_root)) for _, row in images.iterrows()]
    return images


def extract(args):
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    project_root = Path(args.project_root).resolve()
    images = unique_images(manifest_dir, project_root)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    processor = CLIPImageProcessor.from_pretrained(args.model_id, subfolder="feature_extractor")
    model = CLIPVisionModelWithProjection.from_pretrained(args.model_id, subfolder="image_encoder")
    model.eval().requires_grad_(False).to(device)

    raw_batches = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            batch = images.iloc[start : start + args.batch_size]
            pil_images = []
            for image_path in batch["image_path"]:
                with Image.open(image_path) as image:
                    pil_images.append(image.convert("RGB").copy())
            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            raw_batches.append(model(**inputs).image_embeds.cpu().numpy().astype(np.float32))
            print(f"embedded {min(start + len(batch), len(images))}/{len(images)}")

    raw = np.concatenate(raw_batches)
    norms = np.linalg.norm(raw, axis=1)
    if not np.isfinite(raw).all() or np.any(norms <= 0):
        raise ValueError("Invalid CLIP embeddings")
    normalized = raw / norms[:, None]

    np.save(output_dir / "image_embeddings.npy", normalized.astype(np.float32))
    np.save(output_dir / "raw_image_embeddings.npy", raw.astype(np.float32))
    index = images.copy()
    index.insert(0, "embedding_index", np.arange(len(index)))
    index.to_csv(output_dir / "image_embedding_index.csv", index=False)
    summary = {
        "manifest_dir": str(manifest_dir.resolve()),
        "manifest_summary_sha256": sha256(manifest_dir / "split_summary.json"),
        "project_root": str(project_root),
        "model": args.model_id,
        "pipeline": "StableUnCLIPImg2ImgPipeline",
        "raw_embeddings_file": "raw_image_embeddings.npy",
        "raw_embedding_norm_mean": float(norms.mean()),
        "raw_embedding_norm_std": float(norms.std(ddof=1)),
        "images": int(len(images)),
        "embedding_dim": int(raw.shape[1]),
        "normalized": True,
        "matrix_shape": list(normalized.shape),
        "matrix_dtype": str(normalized.dtype),
    }
    (output_dir / "embedding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings compatible with StableUnCLIPImg2ImgPipeline."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="diffusers/stable-diffusion-2-1-unclip-i2i-l")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
