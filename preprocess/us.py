import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

LANDMARKS = ["ofd_1_x", "ofd_1_y", "ofd_2_x", "ofd_2_y", "bpd_1_x", "bpd_1_y", "bpd_2_x", "bpd_2_y"]


def ellipse_from_landmarks(ofd_1, ofd_2, bpd_1, bpd_2, scale=1.1):
    ofd_1, ofd_2, bpd_1, bpd_2 = (np.asarray(p, float) for p in (ofd_1, ofd_2, bpd_1, bpd_2))
    major = ofd_2 - ofd_1
    u = major / np.linalg.norm(major)
    v = np.array([-u[1], u[0]])
    center = (ofd_1 + ofd_2) / 2
    center = center + np.dot((bpd_1 + bpd_2) / 2 - center, v) * v
    a = max(8.0, 0.5 * np.linalg.norm(major) * scale)
    b = max(8.0, 0.5 * abs(np.dot(bpd_2 - bpd_1, v)) * scale)
    return center[0], center[1], a, b, np.degrees(np.arctan2(u[1], u[0]))


def strip_and_align(image, cx, cy, a, b, angle_deg):
    width, height = int(np.ceil(2 * a)), int(np.ceil(2 * b))
    t = np.radians(angle_deg)
    x0, y0 = (width - 1) / 2, (height - 1) / 2
    matrix = np.array([
        [np.cos(t), -np.sin(t), cx - np.cos(t) * x0 + np.sin(t) * y0],
        [np.sin(t), np.cos(t), cy - np.sin(t) * x0 - np.cos(t) * y0],
    ])
    out = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0.0)
    y, x = np.mgrid[:height, :width]
    out[((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 > 1] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="CSV with columns id, either path or dataset, file, and either cx, cy, semi_major, semi_minor, angle_deg or the OFD/BPD landmarks")
    parser.add_argument("--out", required=True, help="output directory of .npy images")
    parser.add_argument("--root", default=".", help="directory containing <dataset>/<file>")
    args = parser.parse_args()
    images = pd.read_csv(args.images)
    if "path" not in images:
        images["path"] = [str(Path(args.root) / d / f) for d, f in zip(images["dataset"], images["file"])]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for row in images.itertuples():
        if set(LANDMARKS) <= set(images.columns):
            p = [getattr(row, k) for k in LANDMARKS]
            ellipse = ellipse_from_landmarks(p[0:2], p[2:4], p[4:6], p[6:8])
        else:
            ellipse = (row.cx, row.cy, row.semi_major, row.semi_minor, row.angle_deg)
        image = np.asarray(Image.open(row.path).convert("L"), dtype=np.float32)
        np.save(Path(args.out) / f"{row.id}.npy", strip_and_align(image, *ellipse))


if __name__ == "__main__":
    main()
