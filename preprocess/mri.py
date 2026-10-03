import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

PLANES = pd.read_csv(Path(__file__).with_name("planes.csv"))
WEEKS = (21, 38)


def atlas_path(atlas_dir, week):
    return Path(atlas_dir) / f"STA{week}{'exp' if week >= 36 else ''}.nii.gz"


def week_of(ga_days):
    return int(np.clip(round(ga_days / 7), *WEEKS))


def register(fixed, moving):
    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=1)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.7)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SetInitialTransform(initial, inPlace=True)
    method.Execute(fixed, moving)
    return initial


def sample(volume, transform, plane):
    i, j = np.meshgrid(np.arange(plane.rows), np.arange(plane.cols), indexing="ij")
    origin = np.array([plane.ox, plane.oy, plane.oz])
    points = origin + i[..., None] * np.array([plane.rx, plane.ry, plane.rz]) + j[..., None] * np.array([plane.cx, plane.cy, plane.cz])
    points = points.reshape(-1, 3)
    if transform is not None:
        matrix = np.array(transform.GetMatrix()).reshape(3, 3)
        center = np.array(transform.GetCenter())
        points = (points - center) @ matrix.T + center + np.array(transform.GetTranslation())
    direction = np.array(volume.GetDirection()).reshape(3, 3)
    index = (points - np.array(volume.GetOrigin())) @ direction / np.array(volume.GetSpacing())
    array = sitk.GetArrayFromImage(volume)
    values = map_coordinates(array, index[:, ::-1].T, order=1, cval=0.0)
    return values.reshape(plane.rows, plane.cols).astype(np.float32)


def process(sid, path, ga_days, atlas_dir, out):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    week = week_of(ga_days)
    fixed = sitk.ReadImage(str(atlas_path(atlas_dir, week)), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(path), sitk.sitkFloat32)
    transform = None if str(path) == str(atlas_path(atlas_dir, week)) else register(fixed, moving)
    for plane in PLANES[PLANES.week == week].itertuples():
        np.save(Path(out) / f"{sid}.{plane.plane}.npy", sample(moving, transform, plane))
    return sid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", required=True, help="CSV with columns id, ga_days and either path or dataset, file")
    parser.add_argument("--atlas", dest="atlas_dir", required=True, help="CRL fetal brain atlas (STA21-STA38)")
    parser.add_argument("--out", required=True, help="output directory of .npy images")
    parser.add_argument("--root", default=".", help="directory containing <dataset>/<file>")
    parser.add_argument("--add-atlas", action="store_true", help="also slice the atlas templates themselves")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    subjects = pd.read_csv(args.volumes)
    if "path" not in subjects:
        subjects["path"] = [str(Path(args.root) / d / f) for d, f in zip(subjects["dataset"], subjects["file"])]
    if args.add_atlas:
        atlas = pd.DataFrame([{"id": f"atlas_mri_w{w}_tt.{7.0 * w}", "path": str(atlas_path(args.atlas_dir, w)), "ga_days": 7.0 * w} for w in range(WEEKS[0], WEEKS[1] + 1)])
        subjects = pd.concat([subjects, atlas], ignore_index=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        n = len(subjects)
        for done in pool.map(process, subjects["id"], subjects["path"], subjects["ga_days"], [args.atlas_dir] * n, [args.out] * n):
            print(done)


if __name__ == "__main__":
    main()
