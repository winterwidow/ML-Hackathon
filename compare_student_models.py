"""Compare every trained model on the permitted student-test development split.

Fits on the 80 training trips, evaluates the 10 student-test trips, and reports
candidate, observation Top-1, HMM route, confusion-matrix, timing, and continuity
metrics. The evaluator_test partition is deliberately never read.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import time
from collections import defaultdict

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

import train_models
from cross_validate_models import make_models
from hmm_decode import continuity, decode
from pathsnap_person1 import process_trip

ROOT = pathlib.Path(__file__).resolve().parent
FEATURES = train_models.FEATURES


def read_candidates(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "trip_id": row["trip_id"],
                "obs_idx": int(row["obs_idx"]),
                "arc_id": int(row["candidate_arc_id"]),
                "x": [float(row[key] or 0) for key in FEATURES],
                "y": int(row["is_route_label"]),
                "known": int(row["label_known"]),
                "dt": float(row["dt_s"] or 0),
            })
    return rows


def arrays(rows):
    x = np.asarray([r["x"] for r in rows], dtype=np.float32)
    y = np.asarray([r["y"] for r in rows], dtype=np.int8)
    groups = np.asarray([r["trip_id"] for r in rows])
    obs = np.asarray([(r["trip_id"], r["obs_idx"]) for r in rows], dtype=object)
    return x, y, groups, obs


def all_route_label_counts(data_root):
    counts = {}
    for folder in sorted(p for p in (data_root / "student_test").iterdir() if p.is_dir()):
        _, summary = process_trip(folder, "student_test")
        counts[folder.name] = summary["gps_observations"] - summary["unlabeled_observations"]
    return counts


def evaluate_model(name, model, train_rows, test_rows, label_counts, data_root, save_model_dir=None):
    train_known = [r for r in train_rows if r["known"]]
    x_train, y_train, groups, obs = arrays(train_known)
    sampled = train_models.downsample(x_train, y_train, groups, obs, train_models.SEED)
    x_fit, y_fit, _ = sampled

    started = time.perf_counter()
    model.fit(x_fit, y_fit)
    fit_seconds = time.perf_counter() - started

    x_test, y_test, _, _ = arrays(test_rows)
    started = time.perf_counter()
    probability_all = model.predict_proba(x_test)[:, 1]
    inference_seconds = time.perf_counter() - started

    known_mask = np.asarray([bool(r["known"]) for r in test_rows])
    probability = probability_all[known_mask]
    y_known = np.asarray([r["y"] for r in test_rows if r["known"]], dtype=np.int8)
    predicted_binary = probability >= 0.5
    cm = confusion_matrix(y_known, predicted_binary, labels=[0, 1])

    by_trip_obs = defaultdict(lambda: defaultdict(list))
    for row, prob in zip(test_rows, probability_all):
        by_trip_obs[row["trip_id"]][row["obs_idx"]].append((row, float(prob)))

    classifier_correct = 0
    hmm_correct = 0
    route_labeled_total = sum(label_counts.values())
    positive_present = 0
    classifier_top1_by_trip = {}
    hmm_top1_by_trip = {}
    classifier_continuity = []
    hmm_continuity = []

    for trip_id, observations in sorted(by_trip_obs.items()):
        trip_folder = data_root / "student_test" / trip_id
        arc_path = trip_folder / f"{trip_id}.arcs"
        arcs = [[int(value) for value in line.split()] for line in arc_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        sequence = []
        classifier_path = {}
        truth = {}
        for obs_idx, candidates in sorted(observations.items()):
            candidates.sort(key=lambda pair: pair[0]["arc_id"])
            best_row, best_probability = max(candidates, key=lambda pair: pair[1])
            classifier_path[obs_idx] = best_row["arc_id"]
            states = [row["arc_id"] for row, _ in candidates]
            probabilities = np.asarray([prob for _, prob in candidates], dtype=float)
            sequence.append({"idx": obs_idx, "states": states, "probs": probabilities, "dt": candidates[0][0]["dt"]})
            positives = [row["arc_id"] for row, _ in candidates if row["known"] and row["y"] == 1]
            if positives:
                truth[obs_idx] = positives[0]

        decoded = decode(sequence, arcs)
        correct_cls = sum(classifier_path.get(i) == aid for i, aid in truth.items())
        correct_hmm = sum(decoded.get(i) == aid for i, aid in truth.items())
        labeled = label_counts.get(trip_id, 0)
        classifier_correct += correct_cls
        hmm_correct += correct_hmm
        positive_present += len(truth)
        classifier_top1_by_trip[trip_id] = correct_cls / max(1, labeled)
        hmm_top1_by_trip[trip_id] = correct_hmm / max(1, labeled)
        obs_by_idx = {item["idx"]: item for item in sequence}
        classifier_continuity.append(continuity(classifier_path, arcs, obs_by_idx))
        hmm_continuity.append(continuity(decoded, arcs, obs_by_idx))

    if save_model_dir is not None:
        save_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": FEATURES, "model_name": name}, save_model_dir / f"{name.lower().replace(' ', '_')}.joblib")

    return {
        "model": name,
        "train_trips": len(set(r["trip_id"] for r in train_known)),
        "student_test_trips": len(label_counts),
        "train_rows_after_negative_sampling": len(y_fit),
        "student_candidate_rows": int(known_mask.sum()),
        "route_labeled_observations": route_labeled_total,
        "positive_route_candidate_present": positive_present,
        "candidate_misses": max(0, route_labeled_total - positive_present),
        "candidate_miss_rate": max(0, route_labeled_total - positive_present) / max(1, route_labeled_total),
        "candidate_accuracy": accuracy_score(y_known, predicted_binary),
        "candidate_macro_f1": f1_score(y_known, predicted_binary, average="macro", zero_division=0),
        "candidate_log_loss": log_loss(y_known, probability, labels=[0, 1]),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "classifier_top1_weighted_all_labels": classifier_correct / max(1, route_labeled_total),
        "hmm_top1_weighted_all_labels": hmm_correct / max(1, route_labeled_total),
        "classifier_top1_macro_by_trip_all_labels": float(np.mean(list(classifier_top1_by_trip.values()))),
        "hmm_top1_macro_by_trip_all_labels": float(np.mean(list(hmm_top1_by_trip.values()))),
        "classifier_mean_continuity": float(np.nanmean(classifier_continuity)),
        "hmm_mean_continuity": float(np.nanmean(hmm_continuity)),
        "train_seconds": fit_seconds,
        "inference_ms_per_observation": 1000 * inference_seconds / max(1, len({(r['trip_id'], r['obs_idx']) for r in test_rows})),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", default=os.getenv("PATHSNAP_TRAIN_CANDIDATES", str(ROOT / "features" / "train_candidates_aligned.csv")))
    parser.add_argument("--student-candidates", default=os.getenv("PATHSNAP_STUDENT_CANDIDATES", str(ROOT / "features" / "student_test_candidates_aligned.csv")))
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--output", default=str(ROOT / "features" / "student_test_model_comparison_aligned.csv"))
    args = parser.parse_args()

    train_rows = read_candidates(pathlib.Path(args.train_candidates))
    test_rows = read_candidates(pathlib.Path(args.student_candidates))
    label_counts = all_route_label_counts(pathlib.Path(args.data_root))

    models = make_models()
    models["Extra Trees Bayesian tuned"] = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=16,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=train_models.SEED,
    )

    results = []
    for name, model in models.items():
        result = evaluate_model(name, model, train_rows, test_rows, label_counts, pathlib.Path(args.data_root))
        results.append(result)
        print(name, {key: round(value, 5) if isinstance(value, float) else value for key, value in result.items() if key != "model"}, flush=True)

    destination = pathlib.Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    selected = next(row for row in results if row["model"] == "Extra Trees Bayesian tuned")
    selected_path = destination.parent / "student_test_metrics_aligned.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected))
        writer.writeheader()
        writer.writerow(selected)

    aggregate_rows = [
        ("classifier_weighted_top1_all_route_labeled", selected["classifier_top1_weighted_all_labels"], "All heuristic route labels; candidate misses count incorrect."),
        ("hmm_weighted_top1_all_route_labeled", selected["hmm_top1_weighted_all_labels"], "All heuristic route labels; candidate misses count incorrect."),
        ("classifier_macro_top1_by_trip_all_route_labeled", selected["classifier_top1_macro_by_trip_all_labels"], "Unweighted mean across student-test trips; misses count incorrect."),
        ("hmm_macro_top1_by_trip_all_route_labeled", selected["hmm_top1_macro_by_trip_all_labels"], "Unweighted mean across student-test trips; misses count incorrect."),
        ("classifier_mean_continuity", selected["classifier_mean_continuity"], "Mean continuity over decoded-adjacent candidate observations."),
        ("hmm_mean_continuity", selected["hmm_mean_continuity"], "Mean continuity over decoded-adjacent candidate observations."),
    ]
    aggregate_path = destination.parent / "student_test_hmm_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value", "definition"])
        writer.writerows(aggregate_rows)
    print(f"Wrote student-test model comparison: {destination}")
    print(f"Wrote selected-model summaries: {selected_path} and {aggregate_path}")


if __name__ == "__main__":
    main()
