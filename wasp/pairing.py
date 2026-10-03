import numpy as np
from scipy.optimize import linear_sum_assignment

from .ot import barycentric, wasp_plan


def greedy(src, tgt):
    j = []
    for ga, plane in zip(src["ga"], src["plane"]):
        same = np.flatnonzero(tgt["plane"] == plane)
        j.append(same[np.argmin(np.abs(tgt["ga"][same] - ga))])
    return np.arange(len(src["ga"])), np.array(j)


def hungarian(src, tgt):
    i, j = [], []
    for plane in np.unique(src["plane"]):
        s, t = np.flatnonzero(src["plane"] == plane), np.flatnonzero(tgt["plane"] == plane)
        rows, cols = linear_sum_assignment(np.abs(src["ga"][s][:, None] - tgt["ga"][t][None, :]))
        i.extend(s[rows])
        j.extend(t[cols])
    return np.array(i), np.array(j)


def random(src, tgt, seed=0):
    n = len(src["ga"])
    return np.arange(n), np.random.default_rng(seed).integers(0, len(tgt["ga"]), n)


def cosine(src, tgt):
    a = src["X"] / (np.linalg.norm(src["X"], axis=1, keepdims=True) + 1e-12)
    b = tgt["X"] / (np.linalg.norm(tgt["X"], axis=1, keepdims=True) + 1e-12)
    return np.arange(len(a)), np.argmax(a @ b.T, axis=1)


PAIRINGS = {"greedy": greedy, "hungarian": hungarian, "random": random, "cosine": cosine}


def pairs(name, src, tgt, eps=None, bin_days=None, terms=("time", "embed", "space"), seed=0, anchor="source"):
    if name == "wasp":
        return src["X"], barycentric(wasp_plan(src, tgt, eps, bin_days, terms), tgt["X"])
    pairing = (lambda a, b: random(a, b, seed)) if name == "random" else PAIRINGS[name]
    if (anchor == "target" and name in ("greedy", "random")) or (anchor == "source" and name == "hungarian"):
        j, i = pairing(tgt, src)
    else:
        i, j = pairing(src, tgt)
    return src["X"][i], tgt["X"][j]
