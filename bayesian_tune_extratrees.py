"""Trip-grouped Gaussian-process search for the Extra Trees candidate ranker."""
from __future__ import annotations

import csv
import pathlib
import time
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold

from train_models import OUT, SEED, downsample, load

N_TRIALS = 10
INITIAL = [(80, 16, 5), (120, 16, 5), (100, 22, 3), (60, 12, 8)]
N_ESTIMATORS = [50, 60, 70, 80, 90, 100, 120, 140, 160]
MAX_DEPTHS = [10, 12, 14, 16, 18, 20, 22, 24]
MIN_LEAVES = [2, 3, 4, 5, 6, 8, 10]


def encode(config):
    n, depth, leaf = config
    return [n / 160.0, (depth - 10) / 14.0, np.log10(leaf) / np.log10(10)]


def top1(model, x, y, obs):
    probabilities = model.predict_proba(x)[:, 1]
    best = {}
    for i, key in enumerate(obs):
        key = tuple(key)
        if key not in best or probabilities[i] > best[key][0]:
            best[key] = (probabilities[i], i)
    return float(np.mean([int(y[i] == 1) for _, i in best.values()]))


def evaluate(config, folds):
    n_estimators, max_depth, min_samples_leaf = config
    scores, train_times = [], []
    for x_train, y_train, x_valid, y_valid, obs_valid, seed in folds:
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        train_times.append(time.perf_counter() - started)
        scores.append(top1(model, x_valid, y_valid, obs_valid))
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "mean_top1": float(np.mean(scores)),
        "std_top1": float(np.std(scores)),
        "fold_scores": scores,
        "mean_train_seconds": float(np.mean(train_times)),
    }


def propose(configs, scores):
    x = np.asarray([encode(config) for config in configs])
    y = np.asarray(scores)
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=1e-5),
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(x, y)
    grid = [(n, d, leaf) for n in N_ESTIMATORS for d in MAX_DEPTHS for leaf in MIN_LEAVES]
    encoded = np.asarray([encode(config) for config in grid])
    mean, std = gp.predict(encoded, return_std=True)
    improvement = mean - max(scores) - 0.001
    z = improvement / np.maximum(std, 1e-9)
    expected = improvement * norm.cdf(z) + std * norm.pdf(z)
    for i, config in enumerate(grid):
        if config in configs:
            expected[i] = -1
    return grid[int(np.argmax(expected))]


def main():
    x, y, groups, obs, _ = load()
    splitter = GroupKFold(n_splits=3)
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), 1):
        x_train, y_train, _ = downsample(x[train_idx], y[train_idx], groups[train_idx], obs[train_idx], SEED + fold)
        folds.append((x_train, y_train, x[valid_idx], y[valid_idx], obs[valid_idx], SEED + fold))
        print(f"Prepared fold {fold}: {len(set(groups[train_idx]))} train trips, {len(set(groups[valid_idx]))} validation trips", flush=True)

    configs, scores, results = [], [], []
    for trial in range(N_TRIALS):
        config = INITIAL[trial] if trial < len(INITIAL) else propose(configs, scores)
        result = evaluate(config, folds)
        result["trial"] = trial + 1
        results.append(result)
        configs.append(config)
        scores.append(result["mean_top1"])
        print(f"Trial {trial + 1}/{N_TRIALS}: {config}, Top-1={result['mean_top1']:.5f} +/- {result['std_top1']:.5f}", flush=True)

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "extra_trees_bayesian_trials_aligned.csv"
    fields = ["trial", "n_estimators", "max_depth", "min_samples_leaf", "mean_top1", "std_top1", "mean_train_seconds", "fold_scores"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "fold_scores": ";".join(f"{score:.6f}" for score in result["fold_scores"])})
    best = max(results, key=lambda row: row["mean_top1"])
    best_path = OUT / "extra_trees_bayesian_best_aligned.txt"
    best_path.write_text(
        "Bayesian search: Gaussian-process expected improvement; 10 trials; 3-fold GroupKFold by trip; training trips only.\n"
        "Objective: mean observation-level Top-1 accuracy.\n"
        f"Best configuration: n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']}, max_features=sqrt, class_weight=balanced.\n"
        f"Best 3-fold grouped Top-1: {best['mean_top1']:.6f} +/- {best['std_top1']:.6f}.\n"
        "Evaluator-test data was not used. Labels are heuristic route alignments.\n",
        encoding="utf-8",
    )
    print("Best configuration:", best, flush=True)
    print(f"Saved {csv_path} and {best_path}", flush=True)


if __name__ == "__main__":
    main()
