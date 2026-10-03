from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("pair", "train", "test")


def load(root, backbone):
    root = Path(root)
    metadata = pd.read_csv(root / "metadata.csv").set_index("id")
    archive = np.load(root / "embeddings" / f"{backbone}.npz", allow_pickle=False)
    X = pd.DataFrame(archive["X"], index=archive["id"]).loc[metadata.index].to_numpy(np.float32)
    data = {}
    for modality in metadata["modality"].unique():
        data[modality] = {}
        for split in SPLITS:
            mask = ((metadata["modality"] == modality) & (metadata["split"] == split)).to_numpy()
            rows = metadata[mask]
            data[modality][split] = {
                "id": rows.index.to_numpy(str),
                "X": X[mask],
                "ga": rows["ga_days"].to_numpy(float),
                "plane": rows["plane"].to_numpy(str),
                "subject": rows["subject_id"].to_numpy(str),
            }
    return data
