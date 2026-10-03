import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.svm import SVC

from . import align
from .pairing import pairs

EPS = (1e-4, 5e-4, 1e-3, 5e-3)
BIN_DAYS = (7, 14, 21)
LAMBDA = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1)
HIDDEN = (256, 512, 1024)
PARAMS = {"ridge": {"lam": LAMBDA}, "mlp": {"lam": LAMBDA, "hidden": HIDDEN}, "cca": {}, "coral": {}}


def identity(x):
    return x


def subset(cohort, index):
    return {k: v[index] for k, v in cohort.items()}


def folds(cohort, k):
    subjects = np.array(sorted(set(cohort["subject"])))
    for f in range(k):
        held = np.isin(cohort["subject"], subjects[f::k])
        yield subset(cohort, ~held), subset(cohort, held)


def probe(train, test, transform=identity):
    x_train, x_test = transform(train["X"]), transform(test["X"])
    ga = Ridge(alpha=1.0).fit(x_train, train["ga"]).predict(x_test)
    plane = SVC(kernel="linear", class_weight="balanced").fit(x_train, train["plane"]).predict(x_test)
    return {"mae": float(np.abs(ga - test["ga"]).mean()), "acc": float((plane == test["plane"]).mean() * 100)}


def evaluate(cohort, transform, k):
    rows = [{"split": f"cv{f}", **probe(tr, va, transform)} for f, (tr, va) in enumerate(folds(cohort["train"], k))]
    return rows + [{"split": "test", **probe(cohort["train"], cohort["test"], transform)}]


def fit(alignment, xs, xt, lam=None, hidden=None, seed=0):
    if alignment == "mlp":
        torch.manual_seed(seed)
        return align.mlp(xs, xt, lam, hidden)
    if alignment == "ridge":
        return align.ridge(xs, xt, lam)
    return getattr(align, alignment)(xs, xt)


def grid(alignment, pairing):
    space = dict(PARAMS[alignment])
    if pairing == "wasp":
        space = {"eps": EPS, "bin_days": BIN_DAYS, **space}
    for values in itertools.product(*space.values()):
        yield dict(zip(space, values))


def run(src, tgt, alignment, pairing, k=3, seed=0, configs=None, terms=("time", "embed", "space"), anchor="source"):
    rows, cache = [], {}
    for config in configs or grid(alignment, pairing):
        if alignment == "coral":
            xs, xt = src["pair"]["X"], tgt["pair"]["X"]
        else:
            key = (config.get("eps"), config.get("bin_days"))
            if key not in cache:
                cache[key] = pairs(pairing, src["pair"], tgt["pair"], *key, terms=terms, seed=seed, anchor=anchor)
            xs, xt = cache[key]
        transform = fit(alignment, xs, xt, config.get("lam"), config.get("hidden"), seed)
        for row in evaluate(src, transform, k):
            rows.append({"alignment": alignment, "pairing": pairing, **config, **row})
    return pd.DataFrame(rows)


def select(results, baseline):
    config = [c for c in ("eps", "bin_days", "lam", "hidden") if c in results]
    results = results.copy()
    results[config] = results[config].fillna(-1)
    cv = results[results["split"] != "test"].merge(baseline, on="split", suffixes=("", "0"))
    cv["score"] = 0.5 * (cv["mae0"] - cv["mae"]) / cv["mae0"] + 0.5 * (cv["acc"] - cv["acc0"]) / cv["acc0"]
    keys = ["alignment", "pairing"] + config
    median = cv.groupby(keys, dropna=False)["score"].median().reset_index()
    best = median.loc[median.groupby(["alignment", "pairing"])["score"].idxmax()]
    test = results[results["split"] == "test"].merge(best, on=keys)
    test[config] = test[config].replace(-1, np.nan)
    return test.drop(columns="split")
