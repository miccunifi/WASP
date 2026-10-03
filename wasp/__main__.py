import argparse
import itertools
from pathlib import Path

import pandas as pd

from .data import load
from .pairing import pairs
from .protocol import LAMBDA, evaluate, fit, identity, run, select

OTHER = {"MRI": "US", "US": "MRI"}
TERMS = ("time", "embed", "space")


def grid_path(args, backbone):
    return args.out / f"{backbone}_{args.source}_grid.csv"


def anchor(args):
    return "source" if args.source == "US" else "target"


def baseline(cohort, k):
    return pd.DataFrame(evaluate(cohort, identity, k))


def show(frame):
    frame = frame.round({"mae": 1, "acc": 1, "score": 3})
    front = [c for c in ("setup", "terms", "pairing", "alignment", "eps", "bin_days", "lam", "hidden") if c in frame]
    print(frame[front + [c for c in frame if c not in front]].to_string(index=False, na_rep="-"))


def table(args):
    data = load(args.data, args.backbone)
    src, tgt = data[args.source], data[OTHER[args.source]]
    zero = baseline(src, args.folds)
    runs = [run(src, tgt, "coral", "none", args.folds, args.seed)]
    for pairing, alignment in itertools.product(args.pairings, args.alignments):
        print(f"{args.backbone} {args.source}->{OTHER[args.source]} {pairing} {alignment}", flush=True)
        runs.append(run(src, tgt, alignment, pairing, args.folds, args.seed, anchor=anchor(args)))
    grid = pd.concat(runs, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(grid_path(args, args.backbone), index=False)
    selected = select(grid, zero)
    selected.to_csv(args.out / f"{args.backbone}_{args.source}_table.csv", index=False)
    show(pd.concat([zero[zero["split"] == "test"].assign(pairing="zero-shot").drop(columns="split"), selected]))


def best(args, backbone, pairing, alignment):
    data = load(args.data, backbone)
    grid = pd.read_csv(grid_path(args, backbone))
    grid = grid[(grid["pairing"] == pairing) & (grid["alignment"] == alignment)]
    row = select(grid, baseline(data[args.source], args.folds)).iloc[0]
    config = {c: row[c] for c in ("eps", "bin_days", "lam", "hidden") if c in row and pd.notna(row[c])}
    for c in ("bin_days", "hidden"):
        if c in config:
            config[c] = int(config[c])
    return data, config


def ablate(args):
    data, config = best(args, args.backbone, "wasp", args.alignment)
    src, tgt = data[args.source], data[OTHER[args.source]]
    rows = [baseline(src, args.folds).query("split == 'test'").assign(terms="none")]
    for n in range(1, 4):
        for terms in itertools.combinations(TERMS, n):
            result = run(src, tgt, args.alignment, "wasp", args.folds, args.seed, [config], terms)
            rows.append(result.query("split == 'test'").assign(terms="+".join(terms)))
    show(pd.concat(rows)[["terms", "mae", "acc"]])


def sensitivity(args):
    grid = pd.read_csv(grid_path(args, args.backbone))
    grid = grid[(grid["pairing"] == "wasp") & (grid["alignment"] == "mlp") & (grid["split"] == "test")]
    print(grid.pivot_table(index="bin_days", columns="eps", values=["mae", "acc"], aggfunc="mean").round(1).to_string())


def distill(args):
    teacher, config = best(args, args.teacher, "wasp", "mlp")
    student = load(args.data, args.student)[args.source]
    assert all((student[s]["id"] == teacher[args.source][s]["id"]).all() for s in student)
    xs, xt = pairs("wasp", teacher[args.source]["pair"], teacher[OTHER[args.source]]["pair"], config["eps"], config["bin_days"])
    F = fit("mlp", xs, xt, config["lam"], config["hidden"], args.seed)
    zero = baseline(student, args.folds)
    rows = [zero.query("split == 'test'").assign(setup="zero-shot")]
    for setup, target in (("raw", identity), ("wasp", F)):
        grid = []
        for lam in LAMBDA:
            G = fit("ridge", student["pair"]["X"], target(teacher[args.source]["pair"]["X"]), lam)
            grid.append(pd.DataFrame(evaluate(student, G, args.folds)).assign(alignment="ridge", pairing=setup, lam=lam))
        rows.append(select(pd.concat(grid), zero).assign(setup=setup))
    show(pd.concat(rows)[["setup", "lam", "mae", "acc"]])


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=Path("data"))
    common.add_argument("--out", type=Path, default=Path("results"))
    common.add_argument("--source", choices=OTHER, default="MRI")
    common.add_argument("--folds", type=int, default=3)
    common.add_argument("--seed", type=int, default=0)
    parser = argparse.ArgumentParser(prog="wasp")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("table", parents=[common])
    p.add_argument("--backbone", required=True)
    p.add_argument("--pairings", nargs="+", default=["greedy", "hungarian", "wasp", "random", "cosine"])
    p.add_argument("--alignments", nargs="+", default=["cca", "ridge", "mlp"])
    p.set_defaults(func=table)
    p = commands.add_parser("ablate", parents=[common])
    p.add_argument("--backbone", required=True)
    p.add_argument("--alignment", default="ridge")
    p.set_defaults(func=ablate)
    p = commands.add_parser("sensitivity", parents=[common])
    p.add_argument("--backbone", required=True)
    p.set_defaults(func=sensitivity)
    p = commands.add_parser("distill", parents=[common])
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.set_defaults(func=distill)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
