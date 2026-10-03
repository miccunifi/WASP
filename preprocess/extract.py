import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backbones import load


def prepare(image, size=224):
    x = torch.as_tensor(image, dtype=torch.float32)
    rows, cols = torch.nonzero(x.abs() > 1e-8, as_tuple=True)
    x = x[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    h, w = x.shape
    scale = size / max(h, w)
    h, w = max(1, round(h * scale)), max(1, round(w * scale))
    x = F.interpolate(x[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    top, left = (size - h) // 2, (size - w) // 2
    x = F.pad(x, (left, size - w - left, top, size - h - top))
    mask = x.abs() > 1e-8
    values = x[mask]
    x[mask] = (values - values.min()) / (values.max() - values.min())
    return x


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["usfm", "echocare", "sam_med2d", "biomedparse"])
    parser.add_argument("--checkpoint", required=True, help="official checkpoint of the backbone")
    parser.add_argument("--images", nargs="+", required=True, help="directories of .npy images written by us.py and mri.py")
    parser.add_argument("--out", default="data/embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.backbone, args.checkpoint).to(device)
    paths = sorted(p for d in args.images for p in Path(d).glob("*.npy"))
    ids, features = [], []
    for start in range(0, len(paths), args.batch_size):
        batch = paths[start : start + args.batch_size]
        x = torch.stack([prepare(np.load(p)) for p in batch])[:, None].to(device)
        if model.size != x.shape[-1]:
            x = F.interpolate(x, size=(model.size, model.size), mode="bilinear", align_corners=False)
        features.append(model(x.repeat(1, 3, 1, 1)).float().cpu().numpy())
        ids += [p.stem for p in batch]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(args.out) / f"{args.backbone}.npz", id=np.array(ids), X=np.concatenate(features))
    print(args.backbone, len(ids))


if __name__ == "__main__":
    main()
