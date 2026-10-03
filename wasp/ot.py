import numpy as np
import torch
from scipy.spatial.distance import cdist


def zscore(D):
    return (D - D.mean()) / (D.std() + 1e-12)


def _bins(ga, origin, width):
    return ((ga - origin) // width).astype(int)


def prototype_distance(src, tgt, bin_days):
    origin = min(src["ga"].min(), tgt["ga"].min())
    bins = _bins(tgt["ga"], origin, bin_days)
    keys = sorted(set(zip(tgt["plane"], bins)))
    centroids = np.stack([tgt["X"][(tgt["plane"] == p) & (bins == b)].astype(np.float64).mean(0) for p, b in keys])
    by_plane = {}
    for k, (p, b) in enumerate(keys):
        by_plane.setdefault(p, []).append((b, k))

    def lookup(cohort):
        out = []
        for p, b in zip(cohort["plane"], _bins(cohort["ga"], origin, bin_days)):
            candidates = by_plane[p]
            out.append(candidates[np.argmin([abs(c - b) for c, _ in candidates])][1])
        return np.array(out)

    return cdist(centroids, centroids)[np.ix_(lookup(src), lookup(tgt))]


def cost_matrix(src, tgt, bin_days, terms=("time", "embed", "space")):
    C = np.zeros((len(src["ga"]), len(tgt["ga"])))
    if "time" in terms:
        C += zscore(np.abs(src["ga"][:, None] - tgt["ga"][None, :]))
    if "embed" in terms:
        C += zscore(prototype_distance(src, tgt, bin_days))
    if "space" in terms:
        C += (src["plane"][:, None] != tgt["plane"][None, :]).astype(float)
    return C - C.min()


def sinkhorn(C, eps, n_iter=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = -torch.as_tensor(C, dtype=torch.float64, device=device) / eps
    n, m = K.shape
    log_a = torch.full((n,), -np.log(n), dtype=torch.float64, device=device)
    log_b = torch.full((m,), -np.log(m), dtype=torch.float64, device=device)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        v = log_b - torch.logsumexp(K + u[:, None], dim=0)
        u = log_a - torch.logsumexp(K + v[None, :], dim=1)
    return torch.exp(u[:, None] + K + v[None, :]).cpu().numpy()


def wasp_plan(src, tgt, eps, bin_days, terms=("time", "embed", "space")):
    return sinkhorn(cost_matrix(src, tgt, bin_days, terms), eps)


def barycentric(P, X):
    return (P / (P.sum(1, keepdims=True) + 1e-12)) @ X
