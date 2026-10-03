import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PLANES = np.array(["TC", "TT", "TV"])


def cohort(rng, modality, n_subjects, ga_range, dim, mixing, planes_per_subject, separation, noise):
    rows, vectors = [], []
    anchors = np.random.default_rng(0).normal(size=(len(PLANES), 2, 16))
    for s in range(n_subjects):
        ga = rng.uniform(*ga_range)
        subject = rng.normal(scale=noise, size=16)
        split = rng.choice(["pair", "train", "test"], p=[0.5, 0.35, 0.15])
        for p in rng.choice(len(PLANES), planes_per_subject, replace=False):
            t = (ga - 140) / 140
            latent = separation * anchors[p, 0] + t * anchors[p, 1] + subject
            vectors.append(np.tanh(latent @ mixing) + rng.normal(scale=noise, size=dim))
            rows.append({
                "id": f"{modality}-{s:04d}-{PLANES[p]}",
                "modality": modality,
                "split": split,
                "subject_id": f"{modality}-{s:04d}",
                "dataset": "toy",
                "ga_days": round(ga),
                "plane": PLANES[p],
            })
    return rows, vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    mri, mri_x = cohort(rng, "MRI", 120, (150, 265), args.dim, rng.normal(size=(16, args.dim)), 3, 0.15, 0.8)
    us, us_x = cohort(rng, "US", 600, (90, 280), args.dim, rng.normal(size=(16, args.dim)), 1, 0.5, 0.3)
    metadata = pd.DataFrame(mri + us)
    (args.out / "embeddings").mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.out / "metadata.csv", index=False)
    np.savez_compressed(args.out / "embeddings" / "toy.npz", id=metadata["id"].to_numpy(str), X=np.stack(mri_x + us_x).astype(np.float32))


if __name__ == "__main__":
    main()
