r"""Exploratory data checks and plots for train and student_test only.

Usage from the repository root:
    python eda.py --data-root .\data --output-dir .\eda

The evaluator partition is deliberately not read by this script.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from pathsnap_person1 import process_trip, project, read_rows

ROOT = Path(__file__).resolve().parent
PARTITIONS = ("train", "student_test")
COLORS = {"train": "#2463eb", "student_test": "#f97316"}


def to_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def network_quality(node_count, arcs):
    parent = list(range(node_count))
    rank = [0] * node_count
    degree = [0] * node_count
    invalid = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for u, v in arcs:
        if not (0 <= u < node_count and 0 <= v < node_count):
            invalid += 1
            continue
        degree[u] += 1
        degree[v] += 1
        union(u, v)
    components = len({find(i) for i in range(node_count)}) if node_count else 0
    return {
        "isolated_nodes": sum(d == 0 for d in degree),
        "weak_components": components,
        "invalid_arc_endpoints": invalid,
    }


def analyze_trip(folder, partition):
    tid = folder.name
    track_raw = read_rows(folder / f"{tid}.track", 3)
    node_raw = read_rows(folder / f"{tid}.nodes", 2)
    arc_raw = read_rows(folder / f"{tid}.arcs", 2)
    route_raw = read_rows(folder / f"{tid}.route", 1)

    track = []
    invalid_coordinates = 0
    invalid_timestamps = 0
    for row in track_raw:
        lon, lat, timestamp = (to_float(x) for x in row)
        if lon is None or lat is None or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            invalid_coordinates += 1
            continue
        if timestamp is None:
            invalid_timestamps += 1
            continue
        track.append((lon, lat, timestamp))

    nodes = []
    invalid_node_coordinates = 0
    for row in node_raw:
        lon, lat = (to_float(x) for x in row)
        if lon is None or lat is None or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            invalid_node_coordinates += 1
        else:
            nodes.append((lon, lat))
    arcs = [(int(row[0]), int(row[1])) for row in arc_raw]
    net = network_quality(len(node_raw), arcs)

    dts = [b[2] - a[2] for a, b in zip(track, track[1:])]
    positive_dt = [dt for dt in dts if dt > 0]
    nonincreasing = sum(dt <= 0 for dt in dts)
    lat0 = sum(p[1] for p in track) / len(track) if track else 0.0
    xy = [project(lon, lat, lat0) for lon, lat, _ in track]
    speeds = [
        math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]) / dt
        for i, dt in enumerate(dts, start=1) if dt > 0
    ]

    # Route alignment is computed only for the permitted development partitions.
    candidate_rows, label_summary = process_trip(folder, partition)
    route_labeled = label_summary["gps_observations"] - label_summary["unlabeled_observations"]
    positive_obs = {r["obs_idx"] for r in candidate_rows if r["is_route_label"]}
    labeled_candidate_obs = {r["obs_idx"] for r in candidate_rows if r["label_known"]}
    candidate_misses = max(0, route_labeled - len(positive_obs))

    valid_times = [p[2] for p in track]
    duration = max(valid_times) - min(valid_times) if valid_times else 0.0
    return {
        "partition": partition,
        "trip_id": tid,
        "gps_observations": len(track_raw),
        "valid_observations": len(track),
        "invalid_coordinates": invalid_coordinates,
        "invalid_timestamps": invalid_timestamps,
        "nonincreasing_timestamp_steps": nonincreasing,
        "nodes": len(node_raw),
        "invalid_node_coordinates": invalid_node_coordinates,
        "arcs": len(arcs),
        "invalid_arc_endpoints": net["invalid_arc_endpoints"],
        "isolated_nodes": net["isolated_nodes"],
        "weak_components": net["weak_components"],
        "route_arcs": len(route_raw),
        "duration_s": duration,
        "gaps_gt_1s": sum(dt > 1 for dt in positive_dt),
        "mean_speed_mps": sum(speeds) / len(speeds) if speeds else 0.0,
        "route_labeled_observations": route_labeled,
        "labeled_observations_with_candidates": len(labeled_candidate_obs),
        "positive_route_candidate_present": len(positive_obs),
        "candidate_misses": candidate_misses,
        "label_coverage_rate": route_labeled / max(1, len(track_raw)),
        "candidate_miss_rate": candidate_misses / max(1, route_labeled),
        "metric_track_xy": xy,
        "speeds_for_plot": speeds,
    }


def aggregate(rows):
    result = []
    additive = [
        "gps_observations", "valid_observations", "invalid_coordinates", "invalid_timestamps",
        "nodes", "arcs", "invalid_arc_endpoints", "isolated_nodes", "route_arcs",
        "route_labeled_observations", "labeled_observations_with_candidates",
        "positive_route_candidate_present", "candidate_misses",
    ]
    for part in PARTITIONS:
        subset = [r for r in rows if r["partition"] == part]
        item = {"partition": part, "trips": len(subset)}
        for key in additive:
            item[key] = sum(r[key] for r in subset)
        item["label_coverage_rate"] = item["route_labeled_observations"] / max(1, item["gps_observations"])
        item["candidate_miss_rate"] = item["candidate_misses"] / max(1, item["route_labeled_observations"])
        result.append(item)
    return result


def save_plots(rows, out):
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})

    # 1. Trip structure, with log y-axis because network sizes vary widely.
    metrics = [("gps_observations", "GPS observations"), ("nodes", "Road nodes"),
               ("arcs", "Directed arcs"), ("route_arcs", "Reference route arcs")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [[r[key] for r in rows if r["partition"] == part] for part in PARTITIONS]
        bp = ax.boxplot(values, tick_labels=PARTITIONS, patch_artist=True, showfliers=False)
        for patch, part in zip(bp["boxes"], PARTITIONS):
            patch.set_facecolor(COLORS[part]); patch.set_alpha(.65)
        ax.set_yscale("log"); ax.set_title(title); ax.set_ylabel("Count (log scale)")
        ax.grid(True, axis="y", alpha=.25)
    fig.suptitle("Trip and road-network structure", fontsize=15)
    fig.tight_layout(); fig.savefig(out / "01_trip_structure.png", dpi=170); plt.close(fig)

    # 2. GPS observations per trip.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for part in PARTITIONS:
        vals = [r["gps_observations"] for r in rows if r["partition"] == part]
        ax.hist(vals, bins=25, histtype="step", linewidth=2, color=COLORS[part], label=part)
    ax.set(title="GPS observations per trip", xlabel="Observations", ylabel="Number of trips")
    ax.legend(); ax.grid(True, alpha=.2); fig.tight_layout(); fig.savefig(out / "02_observations_per_trip.png", dpi=170); plt.close(fig)

    # 3. GPS-derived speed and trip duration.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for part in PARTITIONS:
        vals = [s for r in rows if r["partition"] == part for s in r["speeds_for_plot"] if 0 <= s <= 60]
        axes[0].hist(vals, bins=50, density=True, histtype="step", linewidth=2, color=COLORS[part], label=part)
        durations = [r["duration_s"] for r in rows if r["partition"] == part]
        axes[1].hist(durations, bins=25, histtype="step", linewidth=2, color=COLORS[part], label=part)
    axes[0].set(title="GPS-derived speed (0-60 m/s shown)", xlabel="Speed (m/s)", ylabel="Density")
    axes[1].set(title="Trip duration", xlabel="Seconds", ylabel="Number of trips")
    for ax in axes: ax.legend(); ax.grid(True, alpha=.2)
    fig.tight_layout(); fig.savefig(out / "03_motion_distributions.png", dpi=170); plt.close(fig)

    # 4. Sampling gaps by trip.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for part in PARTITIONS:
        subset = [r for r in rows if r["partition"] == part]
        ax.scatter([r["gps_observations"] for r in subset], [r["gaps_gt_1s"] for r in subset],
                   color=COLORS[part], alpha=.75, label=part)
    ax.set(title="Sampling gaps by trip", xlabel="GPS observations", ylabel="Time gaps > 1 second")
    ax.legend(); ax.grid(True, alpha=.2); fig.tight_layout(); fig.savefig(out / "04_sampling_gaps.png", dpi=170); plt.close(fig)

    # 5. Road-network health.
    fig, ax = plt.subplots(figsize=(8, 6))
    for part in PARTITIONS:
        subset = [r for r in rows if r["partition"] == part]
        ax.scatter([r["weak_components"] for r in subset], [r["isolated_nodes"] for r in subset],
                   color=COLORS[part], alpha=.75, label=part)
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set(title="Weakly connected components and isolated nodes", xlabel="Weak components (symlog)", ylabel="Isolated nodes (symlog)")
    ax.legend(); ax.grid(True, alpha=.2); fig.tight_layout(); fig.savefig(out / "05_network_quality.png", dpi=170); plt.close(fig)

    # 6. Label coverage and positive-candidate misses.
    summary = aggregate(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = list(range(len(PARTITIONS)))
    axes[0].bar(x, [s["label_coverage_rate"] * 100 for s in summary], color=[COLORS[p] for p in PARTITIONS])
    axes[0].set(xticks=x, xticklabels=PARTITIONS, ylim=(0, 100), ylabel="Percent", title="Route-label coverage")
    axes[1].bar(x, [s["candidate_miss_rate"] * 100 for s in summary], color=[COLORS[p] for p in PARTITIONS])
    axes[1].set(xticks=x, xticklabels=PARTITIONS, ylabel="Percent", title="Labeled observations missing positive candidate")
    for ax in axes: ax.grid(True, axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(out / "06_candidate_coverage.png", dpi=170); plt.close(fig)

    # 7. Representative traces in local metre coordinates (no raw degree axes).
    examples = [r for r in rows if r["partition"] == "train"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, row in zip(axes.flat, examples):
        xy = row["metric_track_xy"]
        if xy:
            x0, y0 = xy[0]
            ax.plot([x-x0 for x, _ in xy], [y-y0 for _, y in xy], color=COLORS["train"], linewidth=1.2)
            ax.scatter([0], [0], color="#16a34a", s=20, label="start")
            ax.scatter([xy[-1][0]-x0], [xy[-1][1]-y0], color="#dc2626", s=20, label="end")
        ax.set_title(f"Training trip {row['trip_id']}"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.axis("equal")
    for ax in axes.flat[len(examples):]: ax.axis("off")
    if examples: axes.flat[0].legend()
    fig.suptitle("Representative GPS tracks (local metric projection)", fontsize=14)
    fig.tight_layout(); fig.savefig(out / "07_representative_trajectories.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "eda"))
    args = parser.parse_args()
    data_root, out = Path(args.data_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Do not add evaluator_test here. Its coordinates and labels are reserved for evaluation.
    rows = []
    for part in PARTITIONS:
        folders = sorted(p for p in (data_root / part).iterdir() if p.is_dir())
        for folder in folders:
            record = analyze_trip(folder, part)
            rows.append(record)
            print(part, folder.name, "obs", record["gps_observations"], "isolated", record["isolated_nodes"],
                  "weak_components", record["weak_components"], "candidate_misses", record["candidate_misses"], flush=True)

    fields = [k for k in rows[0] if k not in ("metric_track_xy", "speeds_for_plot")]
    with (out / "eda_trip_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    part_summary = aggregate(rows)
    with (out / "eda_partition_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(part_summary[0])); writer.writeheader(); writer.writerows(part_summary)
    save_plots(rows, out)
    print(f"Wrote train/student_test EDA only to {out}")


if __name__ == "__main__":
    main()
