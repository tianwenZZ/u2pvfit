#!/usr/bin/env python3
"""
Generic CPU-vs-GPU ROOT TTree comparison for Allen monitor outputs.

Run inside an Allen runtime environment, for example:

  Allen/build.x86_64_v3-el9-gcc13+cuda12_4-opt+g/run python3 compare_root_outputs_cpu_gpu.py \
    --cpu-input histograms_1000evts_gausintegral_halffloat_cpu.root \
    --gpu-input histograms_1000evts_gausintegral_single_thread_pv_gpu.root \
    --output-dir /tmp/tzhou_cpu_gpu_compare_1000evt

The default configuration is the DEFAULT_CONFIG dictionary below. It can be
overridden with --config using JSON, or YAML when PyYAML is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CONFIG = {
    "global": {
        "comparison_mode": "relative",  # "relative" or "absolute"
        "relErr": 0.01,
        "absErr": 0.0,
        "epsilon": 1.0e-6,
        "stdErr": 0.01,
        "max_row_reports_per_variable": 50,
        "max_exact_reports_per_variable": 50,
        "max_nonfinite_reports_per_variable": 50,
    },
    "trees": {
        "tv_cheated_tracks": {
            "path": "tv_cheated_tracks/velo_tracks_tree",
            "event_key": "event_number",
            "keys": ["event_number", "track_index", "hit_index"],
            "exact_vars": ["local_event", "global_track_index", "n_hits", "lhcb_id"],
            "compare_vars": ["x", "y", "z", "t"],
        },
        "velo_kalman_filter": {
            "path": "velo_kalman_filter/velo_kalman_filter_tree",
            "event_key": "event_number",
            "keys": ["event_number", "track_index"],
            "exact_vars": ["local_event", "global_track_index", "backward"],
            "compare_vars": [
                "eta",
                "beamline_x",
                "beamline_y",
                "beamline_z",
                "beamline_t",
                "beamline_tx",
                "beamline_ty",
                "beamline_c00",
                "beamline_c20",
                "beamline_c22",
                "beamline_c11",
                "beamline_c31",
                "beamline_c33",
                "beamline_c55",
                "endvelo_x",
                "endvelo_y",
                "endvelo_z",
                "endvelo_t",
                "endvelo_tx",
                "endvelo_ty",
                "endvelo_c00",
                "endvelo_c20",
                "endvelo_c22",
                "endvelo_c11",
                "endvelo_c31",
                "endvelo_c33",
                "endvelo_c55",
            ],
        },
        "pv_beamline_prepare_tracks": {
            "path": "pv_beamline_prepare_tracks/prepare_tracks_tree",
            "event_key": "event_number",
            "keys": ["event_number", "track_index"],
            "exact_vars": ["local_event", "global_track_index"],
            "compare_vars": ["x", "y", "z", "t", "tx", "ty", "W_00", "W_11", "W_55"],
        },
        "pv_beamline_histo": {
            "path": "pv_beamline_histo/histo_tree",
            "event_key": "event_number",
            "keys": ["event_number", "iz", "it"],
            "exact_vars": [],
            "compare_vars": ["z_bin", "t_bin", "weight"],
        },
        "pv_beamline_merge_vertices": {
            "path": "pv_beamline_merge_vertices/merge_vtx_tree",
            "event_key": "event_number",
            # This tree has no stable vertex id and CPU/GPU can emit vertices in
            # a different order. Only compare the number of vertices per event.
            "match_mode": "event_count",
            "plot_event_count": True,
            "seed_count_path": "pv_beamline_peak/peak_tree",
            "plot_filename": "pv_beamline_merge_vertices__event_count_differences.png",
            "match_vertices": True,
            "vertex_match_sigma2": 0.1,
            "vertex_match_plot_filename": "pv_beamline_merge_vertices__vertex_match_distributions.png",
            "keys": ["event_number"],
            "exact_vars": [],
            "compare_vars": [],
        },
    },
}


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    config = deepcopy(DEFAULT_CONFIG)
    if not path:
        return config

    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requested, but PyYAML is not available. Use JSON instead.") from exc
        user_config = yaml.safe_load(text)
    else:
        user_config = json.loads(text)
    if user_config:
        deep_update(config, user_config)
    return config


def open_tree(root_path, tree_path):
    root_file = ROOT.TFile.Open(str(root_path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {root_path}")
    tree = root_file.Get(tree_path)
    if not tree or not tree.InheritsFrom("TTree"):
        root_file.Close()
        raise RuntimeError(f"Failed to find TTree '{tree_path}' in {root_path}")
    return root_file, tree


def branch_names(root_path, tree_path):
    root_file, tree = open_tree(root_path, tree_path)
    names = [branch.GetName() for branch in tree.GetListOfBranches()]
    root_file.Close()
    return names


def load_arrays(root_path, tree_path, branches):
    # ROOT owns the file while RDataFrame reads. Convert to numpy arrays before
    # returning so the caller is independent of the ROOT file lifetime.
    rdf = ROOT.RDataFrame(tree_path, str(root_path))
    arrays = rdf.AsNumpy(branches)
    return {name: np.asarray(values) for name, values in arrays.items()}


def filter_first_events(arrays, event_key, max_events):
    if max_events is None:
        return arrays
    if max_events < 0:
        raise RuntimeError("--max-events must be non-negative")
    events = np.unique(arrays[event_key])
    selected = set(events[:max_events])
    mask = np.fromiter((event in selected for event in arrays[event_key]), dtype=bool, count=len(arrays[event_key]))
    return {name: values[mask] for name, values in arrays.items()}


def lexsort_order(arrays, keys):
    sort_keys = [arrays[key] for key in reversed(keys)]
    return np.lexsort(sort_keys)


def adjacent_duplicate_count(key_arrays):
    if not key_arrays or len(key_arrays[0]) <= 1:
        return 0
    same = np.ones(len(key_arrays[0]) - 1, dtype=bool)
    for values in key_arrays:
        same &= values[1:] == values[:-1]
    return int(np.count_nonzero(same))


def tuple_key(arrays, keys, index):
    return tuple(to_python_scalar(arrays[key][index]) for key in keys)


def to_python_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def sorted_vertex_arrays(arrays, event_key, sort_by):
    order = np.lexsort([arrays[name] for name in reversed([event_key] + sort_by)])
    sorted_arrays = {name: values[order] for name, values in arrays.items()}
    events = sorted_arrays[event_key]
    ranks = np.zeros(len(events), dtype=np.int64)
    if len(events):
        starts = np.r_[0, np.nonzero(events[1:] != events[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(events)]
        for start, end in zip(starts, ends):
            ranks[start:end] = np.arange(end - start, dtype=np.int64)
    sorted_arrays["__sorted_vertex_index"] = ranks
    return sorted_arrays


def prepare_arrays(root_path, tree_config, max_events=None):
    path = tree_config["path"]
    keys = tree_config["keys"]
    branches = sorted(set(
        [tree_config["event_key"]] +
        [key for key in keys if not key.startswith("__")] +
        tree_config.get("sort_by", []) +
        tree_config.get("exact_vars", []) +
        tree_config.get("compare_vars", [])))
    available = set(branch_names(root_path, path))
    missing = sorted(set(branches) - available)
    if missing:
        raise RuntimeError(f"{root_path}: missing branches in {path}: {', '.join(missing)}")

    arrays = load_arrays(root_path, path, branches)
    arrays = filter_first_events(arrays, tree_config["event_key"], max_events)
    if tree_config.get("match_mode") == "sort_within_event":
        arrays = sorted_vertex_arrays(arrays, tree_config["event_key"], tree_config.get("sort_by", []))
        order = np.arange(len(next(iter(arrays.values()))), dtype=np.int64)
    else:
        order = lexsort_order(arrays, keys)
        arrays = {name: values[order] for name, values in arrays.items()}
    key_arrays = [arrays[key] for key in keys]
    return arrays, key_arrays, adjacent_duplicate_count(key_arrays)


def load_event_counts(root_path, tree_config, max_events=None):
    path = tree_config["path"]
    event_key = tree_config["event_key"]
    arrays = load_arrays(root_path, path, [event_key])
    arrays = filter_first_events(arrays, event_key, max_events)
    events = arrays[event_key]
    unique, counts = np.unique(events, return_counts=True)
    return dict(zip([to_python_scalar(event) for event in unique], [int(count) for count in counts])), int(len(events))


def load_event_counts_from_path(root_path, tree_path, event_key, max_events=None):
    arrays = load_arrays(root_path, tree_path, [event_key])
    arrays = filter_first_events(arrays, event_key, max_events)
    events = arrays[event_key]
    unique, counts = np.unique(events, return_counts=True)
    return dict(zip([to_python_scalar(event) for event in unique], [int(count) for count in counts]))


def diff_series(cpu_counts, gpu_counts):
    events = sorted(set(cpu_counts) | set(gpu_counts))
    diffs = np.array([cpu_counts.get(event, 0) - gpu_counts.get(event, 0) for event in events], dtype=float)
    return events, diffs


def plot_event_count_differences(output_path, seed_counts, reco_counts):
    seed_events, seed_diff = diff_series(seed_counts[0], seed_counts[1])
    reco_events, reco_diff = diff_series(reco_counts[0], reco_counts[1])
    all_events = seed_events + reco_events
    if all_events:
        xmin = min(all_events) - 1
        xmax = max(all_events) + 1
        event_span = max(all_events) - min(all_events) + 1
        tick_step = max(1, int(math.ceil(event_span / 25.0)))
        xticks = list(range(int(min(all_events)), int(max(all_events)) + 1, tick_step))
    else:
        xmin, xmax, xticks = -1, 1, []

    all_diff = np.concatenate([seed_diff, reco_diff]) if len(seed_diff) or len(reco_diff) else np.array([0.0])
    y_extent = max(3, int(math.ceil(float(np.max(np.abs(all_diff))))) if len(all_diff) else 3)
    y_lim = (-y_extent - 0.5, y_extent + 0.5)

    fig, (ax_seed, ax_reco, ax_hist) = plt.subplots(
        3, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [1, 1, 0.8]}, sharex=False)
    fig.suptitle("CPU vs GPU vertex count differences (event-by-event)", y=0.985)

    for ax, events, diffs, color, title in [
        (ax_seed, seed_events, seed_diff, "royalblue", "Seed vertices: CPU - GPU count per event"),
        (ax_reco, reco_events, reco_diff, "#ef3b3a", "Reco vertices: CPU - GPU count per event"),
    ]:
        ax.bar(events, diffs, width=0.72, color=color, edgecolor=color, linewidth=0.3)
        ax.axhline(0, color="0.45", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("CPU - GPU", fontsize=9)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(*y_lim)
        ax.set_yticks(range(-y_extent, y_extent + 1, 1))
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("event number", fontsize=9)

    hist_lo = min(-3, int(math.floor(float(np.min(all_diff)))) - 1)
    hist_hi = max(3, int(math.ceil(float(np.max(all_diff)))) + 1)
    bins = np.arange(hist_lo - 0.5, hist_hi + 1.5, 1)
    ax_hist.hist(seed_diff, bins=bins, alpha=0.75, color="royalblue", label="seed")
    ax_hist.hist(reco_diff, bins=bins, alpha=0.55, color="#ef3b3a", label="reco")
    ax_hist.set_title("Distribution of per-event count differences", fontsize=10)
    ax_hist.set_xlabel("CPU - GPU count difference", fontsize=9)
    ax_hist.set_ylabel("events", fontsize=9)
    ax_hist.grid(axis="y", alpha=0.25)
    ax_hist.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_vertex_match_arrays(root_path, tree_config, max_events=None):
    branches = [tree_config["event_key"], "z", "t", "cov22", "cov33"]
    arrays = load_arrays(root_path, tree_config["path"], branches)
    arrays = filter_first_events(arrays, tree_config["event_key"], max_events)
    events = arrays[tree_config["event_key"]]
    vertex_index = np.zeros(len(events), dtype=np.int64)
    by_event = defaultdict(list)
    for index, event in enumerate(events):
        by_event[to_python_scalar(event)].append(index)
    for indices in by_event.values():
        vertex_index[indices] = np.arange(len(indices), dtype=np.int64)
    arrays["__vertex_index"] = vertex_index
    return arrays


def vertex_candidate_chi2(cpu_arrays, gpu_arrays, cpu_index, gpu_index):
    dz = float(gpu_arrays["z"][gpu_index]) - float(cpu_arrays["z"][cpu_index])
    dt = float(gpu_arrays["t"][gpu_index]) - float(cpu_arrays["t"][cpu_index])
    sigma_z2 = float(cpu_arrays["cov22"][cpu_index]) + float(gpu_arrays["cov22"][gpu_index])
    sigma_t2 = float(cpu_arrays["cov33"][cpu_index]) + float(gpu_arrays["cov33"][gpu_index])
    if not (math.isfinite(dz) and math.isfinite(dt) and math.isfinite(sigma_z2) and math.isfinite(sigma_t2)):
        return math.nan
    if sigma_z2 <= 0.0 or sigma_t2 <= 0.0:
        return math.nan
    return dz * dz / sigma_z2 + dt * dt / sigma_t2


def plot_vertex_match_distributions(output_path, pair_rows):
    values = [
        {
            "data": [float(row["z_cpu_minus_gpu"]) for row in pair_rows],
            "xlabel": r"$z_{CPU}-z_{GPU}$ [mm]",
        },
        {
            "data": [float(row["t_cpu_minus_gpu"]) for row in pair_rows],
            "xlabel": r"$t_{CPU}-t_{GPU}$ [ns]",
        },
        {
            "data": [float(row["sigma_z_cpu_minus_gpu"]) for row in pair_rows],
            "xlabel": r"$\sigma_{z,CPU}-\sigma_{z,GPU}$ [mm]",
        },
        {
            "data": [float(row["sigma_t_cpu_minus_gpu"]) for row in pair_rows],
            "xlabel": r"$\sigma_{t,CPU}-\sigma_{t,GPU}$ [ns]",
            "xlim_quantiles": (0.001, 0.999),
        },
        {
            "data": [float(row["z_pull"]) for row in pair_rows],
            "xlabel": r"$(z_{CPU}-z_{GPU})/\sqrt{\sigma_{z,CPU}^{2}+\sigma_{z,GPU}^{2}}$",
        },
        {
            "data": [float(row["t_pull"]) for row in pair_rows],
            "xlabel": r"$(t_{CPU}-t_{GPU})/\sqrt{\sigma_{t,CPU}^{2}+\sigma_{t,GPU}^{2}}$",
        },
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("CPU-GPU reco PV comparison", y=0.985)
    for ax, spec in zip(axes.flat, values):
        data = np.asarray(spec["data"], dtype=float)
        finite = data[np.isfinite(data)]
        xlim = None
        if finite.size and "xlim_quantiles" in spec:
            low_q, high_q = spec["xlim_quantiles"]
            xlow, xhigh = np.quantile(finite, [low_q, high_q])
            if math.isfinite(xlow) and math.isfinite(xhigh) and xlow < xhigh:
                padding = 0.08 * (xhigh - xlow)
                xlim = (xlow - padding, xhigh + padding)
        if finite.size:
            ax.hist(finite, bins=80, range=xlim, color="steelblue", alpha=0.8)
            ax.axvline(0.0, color="0.35", linewidth=0.8)
            ax.text(
                0.98,
                0.95,
                f"n={finite.size}\nmean={finite.mean():.3g}\nstd={finite.std():.3g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        ax.set_xlabel(spec["xlabel"], fontsize=9)
        ax.set_ylabel("matched vertices", fontsize=9)
        if xlim:
            ax.set_xlim(*xlim)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compare_matched_vertices(cpu_input, gpu_input, tree_name, tree_config, output_dir, max_events=None):
    event_key = tree_config["event_key"]
    sigma2 = float(tree_config.get("vertex_match_sigma2", 0.01))
    cpu_arrays = load_vertex_match_arrays(cpu_input, tree_config, max_events)
    gpu_arrays = load_vertex_match_arrays(gpu_input, tree_config, max_events)
    events = sorted(set(map(to_python_scalar, cpu_arrays[event_key])) | set(map(to_python_scalar, gpu_arrays[event_key])))

    cpu_by_event = defaultdict(list)
    gpu_by_event = defaultdict(list)
    for index, event in enumerate(cpu_arrays[event_key]):
        cpu_by_event[to_python_scalar(event)].append(index)
    for index, event in enumerate(gpu_arrays[event_key]):
        gpu_by_event[to_python_scalar(event)].append(index)

    pair_rows = []
    ambiguity_rows = []
    unmatched_rows = []
    event_rows = []
    total_candidates = 0

    for event in events:
        cpu_indices = cpu_by_event.get(event, [])
        gpu_indices = gpu_by_event.get(event, [])
        candidates = []
        cpu_candidates = defaultdict(list)
        gpu_candidates = defaultdict(list)
        for ci in cpu_indices:
            for gi in gpu_indices:
                chi2 = vertex_candidate_chi2(cpu_arrays, gpu_arrays, ci, gi)
                if math.isfinite(chi2) and chi2 <= sigma2:
                    candidates.append((chi2, ci, gi))
                    cpu_candidates[ci].append((gi, chi2))
                    gpu_candidates[gi].append((ci, chi2))
        total_candidates += len(candidates)

        for ci, items in cpu_candidates.items():
            if len(items) > 1:
                ambiguity_rows.append({
                    "tree": tree_name,
                    "event_number": event,
                    "side": "cpu",
                    "vertex_index": int(cpu_arrays["__vertex_index"][ci]),
                    "n_candidates": len(items),
                    "candidate_indices": ";".join(str(int(gpu_arrays["__vertex_index"][gi])) for gi, _ in items),
                    "candidate_chi2": ";".join(f"{chi2:.8g}" for _, chi2 in items),
                })
        for gi, items in gpu_candidates.items():
            if len(items) > 1:
                ambiguity_rows.append({
                    "tree": tree_name,
                    "event_number": event,
                    "side": "gpu",
                    "vertex_index": int(gpu_arrays["__vertex_index"][gi]),
                    "n_candidates": len(items),
                    "candidate_indices": ";".join(str(int(cpu_arrays["__vertex_index"][ci])) for ci, _ in items),
                    "candidate_chi2": ";".join(f"{chi2:.8g}" for _, chi2 in items),
                })

        matched_cpu = set()
        matched_gpu = set()
        for chi2, ci, gi in sorted(candidates, key=lambda item: item[0]):
            if ci in matched_cpu or gi in matched_gpu:
                continue
            matched_cpu.add(ci)
            matched_gpu.add(gi)
            z_cpu = float(cpu_arrays["z"][ci])
            z_gpu = float(gpu_arrays["z"][gi])
            t_cpu = float(cpu_arrays["t"][ci])
            t_gpu = float(gpu_arrays["t"][gi])
            cov22_cpu = float(cpu_arrays["cov22"][ci])
            cov22_gpu = float(gpu_arrays["cov22"][gi])
            cov33_cpu = float(cpu_arrays["cov33"][ci])
            cov33_gpu = float(gpu_arrays["cov33"][gi])
            sigma_z2 = cov22_cpu + cov22_gpu
            sigma_t2 = cov33_cpu + cov33_gpu
            sigma_z_cpu = math.sqrt(cov22_cpu) if cov22_cpu >= 0.0 else math.nan
            sigma_z_gpu = math.sqrt(cov22_gpu) if cov22_gpu >= 0.0 else math.nan
            sigma_t_cpu = math.sqrt(cov33_cpu) if cov33_cpu >= 0.0 else math.nan
            sigma_t_gpu = math.sqrt(cov33_gpu) if cov33_gpu >= 0.0 else math.nan
            pair_rows.append({
                "tree": tree_name,
                "event_number": event,
                "cpu_vertex_index": int(cpu_arrays["__vertex_index"][ci]),
                "gpu_vertex_index": int(gpu_arrays["__vertex_index"][gi]),
                "match_chi2": chi2,
                "z_cpu": z_cpu,
                "z_gpu": z_gpu,
                "z_cpu_minus_gpu": z_cpu - z_gpu,
                "t_cpu": t_cpu,
                "t_gpu": t_gpu,
                "t_cpu_minus_gpu": t_cpu - t_gpu,
                "cov22_cpu": cov22_cpu,
                "cov22_gpu": cov22_gpu,
                "cov22_cpu_minus_gpu": cov22_cpu - cov22_gpu,
                "sigma_z_cpu": sigma_z_cpu,
                "sigma_z_gpu": sigma_z_gpu,
                "sigma_z_cpu_minus_gpu": sigma_z_cpu - sigma_z_gpu,
                "cov33_cpu": cov33_cpu,
                "cov33_gpu": cov33_gpu,
                "cov33_cpu_minus_gpu": cov33_cpu - cov33_gpu,
                "sigma_t_cpu": sigma_t_cpu,
                "sigma_t_gpu": sigma_t_gpu,
                "sigma_t_cpu_minus_gpu": sigma_t_cpu - sigma_t_gpu,
                "z_pull": (z_cpu - z_gpu) / math.sqrt(sigma_z2) if sigma_z2 > 0.0 else math.nan,
                "t_pull": (t_cpu - t_gpu) / math.sqrt(sigma_t2) if sigma_t2 > 0.0 else math.nan,
            })

        for ci in cpu_indices:
            if ci not in matched_cpu:
                unmatched_rows.append({
                    "tree": tree_name,
                    "side": "cpu",
                    "event_number": event,
                    "vertex_index": int(cpu_arrays["__vertex_index"][ci]),
                    "z": float(cpu_arrays["z"][ci]),
                    "t": float(cpu_arrays["t"][ci]),
                    "cov22": float(cpu_arrays["cov22"][ci]),
                    "cov33": float(cpu_arrays["cov33"][ci]),
                })
        for gi in gpu_indices:
            if gi not in matched_gpu:
                unmatched_rows.append({
                    "tree": tree_name,
                    "side": "gpu",
                    "event_number": event,
                    "vertex_index": int(gpu_arrays["__vertex_index"][gi]),
                    "z": float(gpu_arrays["z"][gi]),
                    "t": float(gpu_arrays["t"][gi]),
                    "cov22": float(gpu_arrays["cov22"][gi]),
                    "cov33": float(gpu_arrays["cov33"][gi]),
                })
        event_rows.append({
            "tree": tree_name,
            "event_number": event,
            "cpu_vertices": len(cpu_indices),
            "gpu_vertices": len(gpu_indices),
            "candidate_pairs": len(candidates),
            "matched_pairs": len(matched_cpu),
            "unmatched_cpu": len(cpu_indices) - len(matched_cpu),
            "unmatched_gpu": len(gpu_indices) - len(matched_gpu),
            "ambiguous_cpu_vertices": sum(1 for items in cpu_candidates.values() if len(items) > 1),
            "ambiguous_gpu_vertices": sum(1 for items in gpu_candidates.values() if len(items) > 1),
        })

    write_csv(
        output_dir / f"{tree_name}__vertex_match_pairs.csv",
        pair_rows,
        [
            "tree",
            "event_number",
            "cpu_vertex_index",
            "gpu_vertex_index",
            "match_chi2",
            "z_cpu",
            "z_gpu",
            "z_cpu_minus_gpu",
            "t_cpu",
            "t_gpu",
            "t_cpu_minus_gpu",
            "cov22_cpu",
            "cov22_gpu",
            "cov22_cpu_minus_gpu",
            "sigma_z_cpu",
            "sigma_z_gpu",
            "sigma_z_cpu_minus_gpu",
            "cov33_cpu",
            "cov33_gpu",
            "cov33_cpu_minus_gpu",
            "sigma_t_cpu",
            "sigma_t_gpu",
            "sigma_t_cpu_minus_gpu",
            "z_pull",
            "t_pull",
        ],
    )
    write_csv(
        output_dir / f"{tree_name}__vertex_match_ambiguities.csv",
        ambiguity_rows,
        ["tree", "event_number", "side", "vertex_index", "n_candidates", "candidate_indices", "candidate_chi2"],
    )
    write_csv(
        output_dir / f"{tree_name}__vertex_match_unmatched.csv",
        unmatched_rows,
        ["tree", "side", "event_number", "vertex_index", "z", "t", "cov22", "cov33"],
    )
    write_csv(
        output_dir / f"{tree_name}__vertex_match_event_summary.csv",
        event_rows,
        [
            "tree",
            "event_number",
            "cpu_vertices",
            "gpu_vertices",
            "candidate_pairs",
            "matched_pairs",
            "unmatched_cpu",
            "unmatched_gpu",
            "ambiguous_cpu_vertices",
            "ambiguous_gpu_vertices",
        ],
    )

    plot_path = output_dir / tree_config.get(
        "vertex_match_plot_filename", f"{tree_name}__vertex_match_distributions.png")
    plot_vertex_match_distributions(plot_path, pair_rows)

    return {
        "events": len(events),
        "candidate_pairs": total_candidates,
        "matched_pairs": len(pair_rows),
        "unmatched_cpu": sum(1 for row in unmatched_rows if row["side"] == "cpu"),
        "unmatched_gpu": sum(1 for row in unmatched_rows if row["side"] == "gpu"),
        "ambiguity_rows": len(ambiguity_rows),
        "plot_path": plot_path,
    }


def compare_event_counts(cpu_input, gpu_input, tree_name, tree_config, output_dir, max_events=None):
    path = tree_config["path"]
    event_key = tree_config["event_key"]
    print(f"\n[{tree_name}] {path}")
    print(f"  match_mode: event_count")

    cpu_counts, cpu_rows = load_event_counts(cpu_input, tree_config, max_events)
    gpu_counts, gpu_rows = load_event_counts(gpu_input, tree_config, max_events)
    events = sorted(set(cpu_counts) | set(gpu_counts))
    rows = []
    mismatch_rows = []
    for event in events:
        cpu_count = cpu_counts.get(event, 0)
        gpu_count = gpu_counts.get(event, 0)
        diff = cpu_count - gpu_count
        row = {
            "tree": tree_name,
            "event_number": event,
            "cpu_count": cpu_count,
            "gpu_count": gpu_count,
            "diff_cpu_minus_gpu": diff,
        }
        rows.append(row)
        if diff != 0:
            mismatch_rows.append(row)

    write_csv(
        output_dir / f"{tree_name}__event_count_comparison.csv",
        rows,
        ["tree", "event_number", "cpu_count", "gpu_count", "diff_cpu_minus_gpu"],
    )
    write_csv(
        output_dir / f"{tree_name}__event_count_mismatches.csv",
        mismatch_rows,
        ["tree", "event_number", "cpu_count", "gpu_count", "diff_cpu_minus_gpu"],
    )

    if tree_config.get("plot_event_count", False):
        seed_path = tree_config.get("seed_count_path")
        plot_path = output_dir / tree_config.get("plot_filename", f"{tree_name}__event_count_differences.png")
        if seed_path:
            seed_cpu_counts = load_event_counts_from_path(cpu_input, seed_path, event_key, max_events)
            seed_gpu_counts = load_event_counts_from_path(gpu_input, seed_path, event_key, max_events)
        else:
            seed_cpu_counts = {}
            seed_gpu_counts = {}
        plot_event_count_differences(plot_path, (seed_cpu_counts, seed_gpu_counts), (cpu_counts, gpu_counts))

    vertex_match_summary = None
    if tree_config.get("match_vertices", False):
        vertex_match_summary = compare_matched_vertices(cpu_input, gpu_input, tree_name, tree_config, output_dir, max_events)

    summary = [{
        "variable": "event_vertex_count",
        "kind": "event_count",
        "matched_rows": len(events),
        "nonfinite_rows": 0,
        "row_fail_rows": len(mismatch_rows),
        "row_fail_events": len(mismatch_rows),
        "event_std_fail_events": 0,
        "max_absdiff": max((abs(row["diff_cpu_minus_gpu"]) for row in rows), default=0),
        "mean_absdiff": float(np.mean([abs(row["diff_cpu_minus_gpu"]) for row in rows])) if rows else 0.0,
        "max_relerr": "",
        "max_event_stdErr": "",
    }]
    if vertex_match_summary:
        summary.append({
            "variable": "zt_vertex_match",
            "kind": "vertex_match",
            "matched_rows": vertex_match_summary["matched_pairs"],
            "nonfinite_rows": 0,
            "row_fail_rows": vertex_match_summary["unmatched_cpu"] + vertex_match_summary["unmatched_gpu"],
            "row_fail_events": "",
            "event_std_fail_events": vertex_match_summary["ambiguity_rows"],
            "max_absdiff": "",
            "mean_absdiff": "",
            "max_relerr": "",
            "max_event_stdErr": "",
        })
    write_csv(
        output_dir / f"{tree_name}__summary.csv",
        summary,
        [
            "variable",
            "kind",
            "matched_rows",
            "nonfinite_rows",
            "row_fail_rows",
            "row_fail_events",
            "event_std_fail_events",
            "max_absdiff",
            "mean_absdiff",
            "max_relerr",
            "max_event_stdErr",
        ],
    )
    for suffix, fields in [
        ("row_anomaly_samples", ["tree", "variable", "row_id", "event_number", "cpu", "gpu", "absdiff", "tolerance"]),
        ("exact_mismatch_samples", ["tree", "variable", "row_id", "event_number", "cpu", "gpu"]),
        ("nonfinite_samples", ["tree", "variable", "row_id", "event_number", "cpu", "gpu"]),
        ("event_stdErr_failures", ["tree", "variable", "event_number", "stdErr", "rows_in_event", "threshold"]),
        ("key_mismatch_samples", ["tree", "side", "row_id", "event_number"]),
    ]:
        write_csv(output_dir / f"{tree_name}__{suffix}.csv", [], fields)

    print(
        f"  events={len(events)} CPU vertices={cpu_rows} GPU vertices={gpu_rows} "
        f"count_mismatch_events={len(mismatch_rows)}")
    if tree_config.get("plot_event_count", False):
        print(f"  event count plot: {output_dir / tree_config.get('plot_filename', f'{tree_name}__event_count_differences.png')}")
    if vertex_match_summary:
        print(
            "  vertex z/t matching: "
            f"sigma2={tree_config.get('vertex_match_sigma2', 0.01)} "
            f"candidates={vertex_match_summary['candidate_pairs']} "
            f"matched={vertex_match_summary['matched_pairs']} "
            f"unmatched_cpu={vertex_match_summary['unmatched_cpu']} "
            f"unmatched_gpu={vertex_match_summary['unmatched_gpu']} "
            f"ambiguity_rows={vertex_match_summary['ambiguity_rows']}")
        print(f"  vertex match plot: {vertex_match_summary['plot_path']}")
    if mismatch_rows:
        print(f"  first mismatches: {mismatch_rows[:5]}")
    else:
        print("  OK: per-event vertex counts match")

    return {
        "tree": tree_name,
        "path": path,
        "cpu_rows": cpu_rows,
        "gpu_rows": gpu_rows,
        "matched_rows": len(events),
        "cpu_duplicate_keys": 0,
        "gpu_duplicate_keys": 0,
        "cpu_only_keys": len(set(cpu_counts) - set(gpu_counts)),
        "gpu_only_keys": len(set(gpu_counts) - set(cpu_counts)),
    }, summary, [], [], []


def key_arrays_equal(cpu_keys, gpu_keys):
    if len(cpu_keys) != len(gpu_keys):
        return False
    if not cpu_keys:
        return True
    if len(cpu_keys[0]) != len(gpu_keys[0]):
        return False
    return all(np.array_equal(left, right) for left, right in zip(cpu_keys, gpu_keys))


def key_record_array(arrays, keys):
    dtype = [(key, arrays[key].dtype) for key in keys]
    records = np.empty(len(arrays[keys[0]]), dtype=dtype)
    for key in keys:
        records[key] = arrays[key]
    return records


def record_to_key_string(record, keys):
    return ";".join(f"{key}={to_python_scalar(record[key])}" for key in keys)


def limited_key_set_diff(cpu_arrays, gpu_arrays, keys, limit=20):
    cpu_set = {tuple_key(cpu_arrays, keys, i) for i in range(len(cpu_arrays[keys[0]]))}
    gpu_set = {tuple_key(gpu_arrays, keys, i) for i in range(len(gpu_arrays[keys[0]]))}
    return sorted(cpu_set - gpu_set)[:limit], sorted(gpu_set - cpu_set)[:limit], len(cpu_set - gpu_set), len(gpu_set - cpu_set)


def align_by_key_intersection(cpu_arrays, gpu_arrays, keys):
    cpu_records = key_record_array(cpu_arrays, keys)
    gpu_records = key_record_array(gpu_arrays, keys)
    common, cpu_indices, gpu_indices = np.intersect1d(cpu_records, gpu_records, return_indices=True)
    cpu_only = np.setdiff1d(cpu_records, gpu_records)
    gpu_only = np.setdiff1d(gpu_records, cpu_records)
    aligned_cpu = {name: values[cpu_indices] for name, values in cpu_arrays.items()}
    aligned_gpu = {name: values[gpu_indices] for name, values in gpu_arrays.items()}
    return aligned_cpu, aligned_gpu, common, cpu_only, gpu_only


def safe_isfinite(values):
    try:
        return np.isfinite(values.astype(float, copy=False))
    except (TypeError, ValueError):
        return np.ones(len(values), dtype=bool)


def key_string(arrays, keys, index):
    return ";".join(f"{key}={to_python_scalar(arrays[key][index])}" for key in keys)


def event_groups(events):
    if len(events) == 0:
        return []
    order = np.argsort(events, kind="stable")
    sorted_events = events[order]
    starts = np.r_[0, np.nonzero(sorted_events[1:] != sorted_events[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(sorted_events)]
    return [(to_python_scalar(sorted_events[start]), order[start:end]) for start, end in zip(starts, ends)]


def compare_numeric_variable(cpu_arrays, gpu_arrays, keys, event_key, variable, global_cfg):
    cpu = cpu_arrays[variable].astype(float, copy=False)
    gpu = gpu_arrays[variable].astype(float, copy=False)
    diff = gpu - cpu
    absdiff = np.abs(diff)
    finite = safe_isfinite(cpu) & safe_isfinite(gpu)

    eps = float(global_cfg["epsilon"])
    rel_err = float(global_cfg["relErr"])
    abs_err = float(global_cfg["absErr"])
    mode = global_cfg["comparison_mode"]
    if mode == "absolute":
        tolerance = abs_err + eps
    elif mode == "relative":
        tolerance = rel_err * np.abs(cpu) + eps
    else:
        raise RuntimeError(f"Unknown comparison_mode: {mode}")

    row_pass = finite & (absdiff <= tolerance)
    row_fail = finite & ~row_pass
    nonfinite = ~finite

    denom = np.maximum(np.abs(cpu), eps)
    rel_residual = np.zeros(len(cpu), dtype=float)
    rel_residual[finite] = diff[finite] / denom[finite]

    event_failures = []
    max_std = 0.0
    event_fail_threshold = float(global_cfg["stdErr"])
    for event, indices in event_groups(cpu_arrays[event_key]):
        valid = finite[indices]
        if not np.any(valid):
            std_err = math.nan
        else:
            residual = rel_residual[indices][valid]
            std_err = float(math.sqrt(np.mean(residual * residual)))
            max_std = max(max_std, std_err)
        if math.isfinite(std_err) and std_err > event_fail_threshold:
            event_failures.append((event, std_err, len(indices)))

    max_rel = 0.0
    if np.any(finite):
        max_rel = float(np.max(absdiff[finite] / denom[finite]))

    sample_limit = int(global_cfg["max_row_reports_per_variable"])
    row_fail_indices = np.nonzero(row_fail)[0][:sample_limit]
    nonfinite_limit = int(global_cfg["max_nonfinite_reports_per_variable"])
    nonfinite_indices = np.nonzero(nonfinite)[0][:nonfinite_limit]

    row_samples = [
        {
            "variable": variable,
            "row_id": key_string(cpu_arrays, keys, int(index)),
            "event_number": to_python_scalar(cpu_arrays[event_key][index]),
            "cpu": cpu[index],
            "gpu": gpu[index],
            "absdiff": absdiff[index],
            "tolerance": tolerance[index] if np.ndim(tolerance) else tolerance,
        }
        for index in row_fail_indices
    ]
    nonfinite_samples = [
        {
            "variable": variable,
            "row_id": key_string(cpu_arrays, keys, int(index)),
            "event_number": to_python_scalar(cpu_arrays[event_key][index]),
            "cpu": cpu[index],
            "gpu": gpu[index],
        }
        for index in nonfinite_indices
    ]

    stats = {
        "variable": variable,
        "kind": "numeric",
        "matched_rows": len(cpu),
        "nonfinite_rows": int(np.count_nonzero(nonfinite)),
        "row_fail_rows": int(np.count_nonzero(row_fail)),
        "row_fail_events": len(set(to_python_scalar(cpu_arrays[event_key][i]) for i in np.nonzero(row_fail)[0])),
        "event_std_fail_events": len(event_failures),
        "max_absdiff": float(np.max(absdiff[finite])) if np.any(finite) else math.nan,
        "mean_absdiff": float(np.mean(absdiff[finite])) if np.any(finite) else math.nan,
        "max_relerr": max_rel,
        "max_event_stdErr": max_std,
    }
    return stats, row_samples, nonfinite_samples, event_failures


def compare_exact_variable(cpu_arrays, gpu_arrays, keys, event_key, variable, global_cfg):
    cpu = cpu_arrays[variable]
    gpu = gpu_arrays[variable]
    finite = safe_isfinite(cpu) & safe_isfinite(gpu)
    nonfinite = ~finite
    mismatch = finite & (cpu != gpu)
    limit = int(global_cfg["max_exact_reports_per_variable"])
    mismatch_indices = np.nonzero(mismatch)[0][:limit]
    nonfinite_limit = int(global_cfg["max_nonfinite_reports_per_variable"])
    nonfinite_indices = np.nonzero(nonfinite)[0][:nonfinite_limit]

    samples = [
        {
            "variable": variable,
            "row_id": key_string(cpu_arrays, keys, int(index)),
            "event_number": to_python_scalar(cpu_arrays[event_key][index]),
            "cpu": to_python_scalar(cpu[index]),
            "gpu": to_python_scalar(gpu[index]),
        }
        for index in mismatch_indices
    ]
    nonfinite_samples = [
        {
            "variable": variable,
            "row_id": key_string(cpu_arrays, keys, int(index)),
            "event_number": to_python_scalar(cpu_arrays[event_key][index]),
            "cpu": to_python_scalar(cpu[index]),
            "gpu": to_python_scalar(gpu[index]),
        }
        for index in nonfinite_indices
    ]
    stats = {
        "variable": variable,
        "kind": "exact",
        "matched_rows": len(cpu),
        "nonfinite_rows": int(np.count_nonzero(nonfinite)),
        "row_fail_rows": int(np.count_nonzero(mismatch)),
        "row_fail_events": len(set(to_python_scalar(cpu_arrays[event_key][i]) for i in np.nonzero(mismatch)[0])),
        "event_std_fail_events": 0,
        "max_absdiff": "",
        "mean_absdiff": "",
        "max_relerr": "",
        "max_event_stdErr": "",
    }
    return stats, samples, nonfinite_samples


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compare_tree(cpu_input, gpu_input, tree_name, tree_config, global_cfg, output_dir, max_events=None):
    path = tree_config["path"]
    keys = tree_config["keys"]
    event_key = tree_config["event_key"]
    print(f"\n[{tree_name}] {path}")
    print(f"  keys: {', '.join(keys)}")
    if tree_config.get("match_mode") == "sort_within_event":
        print(f"  match_mode: sort_within_event, sort_by={tree_config.get('sort_by', [])}")

    cpu_arrays, cpu_keys, cpu_dups = prepare_arrays(cpu_input, tree_config, max_events)
    gpu_arrays, gpu_keys, gpu_dups = prepare_arrays(gpu_input, tree_config, max_events)
    cpu_rows = len(cpu_keys[0]) if cpu_keys else 0
    gpu_rows = len(gpu_keys[0]) if gpu_keys else 0

    tree_meta = {
        "tree": tree_name,
        "path": path,
        "cpu_rows": cpu_rows,
        "gpu_rows": gpu_rows,
        "matched_rows": 0,
        "cpu_duplicate_keys": cpu_dups,
        "gpu_duplicate_keys": gpu_dups,
        "cpu_only_keys": 0,
        "gpu_only_keys": 0,
    }

    key_mismatch_rows = []
    if not key_arrays_equal(cpu_keys, gpu_keys):
        cpu_arrays, gpu_arrays, common_keys, cpu_only, gpu_only = align_by_key_intersection(cpu_arrays, gpu_arrays, keys)
        tree_meta["cpu_only_keys"] = len(cpu_only)
        tree_meta["gpu_only_keys"] = len(gpu_only)
        print(f"  key mismatch: CPU-only={len(cpu_only)}, GPU-only={len(gpu_only)}, common={len(common_keys)}")
        for record in cpu_only[:50]:
            key_mismatch_rows.append({
                "tree": tree_name,
                "side": "cpu_only",
                "row_id": record_to_key_string(record, keys),
                "event_number": to_python_scalar(record[event_key]) if event_key in record.dtype.names else "",
            })
        for record in gpu_only[:50]:
            key_mismatch_rows.append({
                "tree": tree_name,
                "side": "gpu_only",
                "row_id": record_to_key_string(record, keys),
                "event_number": to_python_scalar(record[event_key]) if event_key in record.dtype.names else "",
            })
        if len(cpu_only):
            print(f"  first CPU-only keys: {[record_to_key_string(record, keys) for record in cpu_only[:5]]}")
        if len(gpu_only):
            print(f"  first GPU-only keys: {[record_to_key_string(record, keys) for record in gpu_only[:5]]}")

    matched_rows = len(cpu_arrays[keys[0]]) if keys else 0
    tree_meta["matched_rows"] = matched_rows
    print(f"  rows: CPU={cpu_rows} GPU={gpu_rows} matched={matched_rows} duplicate_keys CPU/GPU={cpu_dups}/{gpu_dups}")

    summary_rows = []
    row_anomalies = []
    nonfinite_rows = []
    event_rows = []
    exact_rows = []

    for variable in tree_config.get("exact_vars", []):
        stats, samples, nonfinite_samples = compare_exact_variable(
            cpu_arrays, gpu_arrays, keys, event_key, variable, global_cfg)
        summary_rows.append(stats)
        exact_rows.extend({"tree": tree_name, **sample} for sample in samples)
        nonfinite_rows.extend({"tree": tree_name, **sample} for sample in nonfinite_samples)

    for variable in tree_config.get("compare_vars", []):
        stats, samples, nonfinite_samples, event_failures = compare_numeric_variable(
            cpu_arrays, gpu_arrays, keys, event_key, variable, global_cfg)
        summary_rows.append(stats)
        row_anomalies.extend({"tree": tree_name, **sample} for sample in samples)
        nonfinite_rows.extend({"tree": tree_name, **sample} for sample in nonfinite_samples)
        for event, std_err, n_rows in event_failures:
            event_rows.append({
                "tree": tree_name,
                "variable": variable,
                "event_number": event,
                "stdErr": std_err,
                "rows_in_event": n_rows,
                "threshold": global_cfg["stdErr"],
            })

    failed_vars = [
        row["variable"] for row in summary_rows
        if row["nonfinite_rows"] or row["row_fail_rows"] or row["event_std_fail_events"]
    ]
    if failed_vars:
        print(f"  variables with anomalies: {', '.join(failed_vars)}")
    else:
        print("  OK: no nonfinite rows, row tolerance failures, or event stdErr failures")

    stem = tree_name
    write_csv(
        output_dir / f"{stem}__summary.csv",
        summary_rows,
        [
            "variable",
            "kind",
            "matched_rows",
            "nonfinite_rows",
            "row_fail_rows",
            "row_fail_events",
            "event_std_fail_events",
            "max_absdiff",
            "mean_absdiff",
            "max_relerr",
            "max_event_stdErr",
        ],
    )
    write_csv(
        output_dir / f"{stem}__row_anomaly_samples.csv",
        row_anomalies,
        ["tree", "variable", "row_id", "event_number", "cpu", "gpu", "absdiff", "tolerance"],
    )
    write_csv(
        output_dir / f"{stem}__exact_mismatch_samples.csv",
        exact_rows,
        ["tree", "variable", "row_id", "event_number", "cpu", "gpu"],
    )
    write_csv(
        output_dir / f"{stem}__nonfinite_samples.csv",
        nonfinite_rows,
        ["tree", "variable", "row_id", "event_number", "cpu", "gpu"],
    )
    write_csv(
        output_dir / f"{stem}__event_stdErr_failures.csv",
        event_rows,
        ["tree", "variable", "event_number", "stdErr", "rows_in_event", "threshold"],
    )
    write_csv(
        output_dir / f"{stem}__key_mismatch_samples.csv",
        key_mismatch_rows,
        ["tree", "side", "row_id", "event_number"],
    )
    return tree_meta, summary_rows, row_anomalies, nonfinite_rows, event_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-input", required=True, help="CPU ROOT monitor file, used as reference.")
    parser.add_argument("--gpu-input", required=True, help="GPU ROOT monitor file.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary/anomaly CSV outputs.")
    parser.add_argument("--config", default=None, help="Optional JSON or YAML config overriding DEFAULT_CONFIG.")
    parser.add_argument("--trees", default=None, help="Comma-separated tree config names to compare. Default: all.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Compare only the first N event_number values in sorted order. Default: all events.")
    parser.add_argument("--print-config", action="store_true", help="Print effective config and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.print_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "effective_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    selected = list(config["trees"].keys())
    if args.trees:
        requested = [item.strip() for item in args.trees.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(config["trees"]))
        if unknown:
            raise RuntimeError(f"Unknown tree names in --trees: {', '.join(unknown)}")
        selected = requested

    print("CPU reference:", args.cpu_input)
    print("GPU compare:  ", args.gpu_input)
    print("Output dir:   ", output_dir)
    if args.max_events is not None:
        print("Max events:   ", args.max_events)
    print(
        "Tolerance:    "
        f"mode={config['global']['comparison_mode']} relErr={config['global']['relErr']} "
        f"absErr={config['global']['absErr']} epsilon={config['global']['epsilon']} "
        f"event stdErr={config['global']['stdErr']}")

    tree_rows = []
    for tree_name in selected:
        tree_config = config["trees"][tree_name]
        if tree_config.get("match_mode") == "event_count":
            tree_meta, _, _, _, _ = compare_event_counts(
                args.cpu_input, args.gpu_input, tree_name, tree_config, output_dir, args.max_events)
        else:
            tree_meta, _, _, _, _ = compare_tree(
                args.cpu_input,
                args.gpu_input,
                tree_name,
                tree_config,
                config["global"],
                output_dir,
                args.max_events)
        tree_rows.append(tree_meta)

    write_csv(
        output_dir / "tree_overview.csv",
        tree_rows,
        [
            "tree",
            "path",
            "cpu_rows",
            "gpu_rows",
            "matched_rows",
            "cpu_duplicate_keys",
            "gpu_duplicate_keys",
            "cpu_only_keys",
            "gpu_only_keys",
        ],
    )
    print("\nWrote:")
    print(f"  {output_dir / 'effective_config.json'}")
    print(f"  {output_dir / 'tree_overview.csv'}")
    print(f"  {output_dir}/*__summary.csv")
    print(f"  {output_dir}/*__row_anomaly_samples.csv")
    print(f"  {output_dir}/*__event_stdErr_failures.csv")
    print(f"  {output_dir}/*__key_mismatch_samples.csv")


if __name__ == "__main__":
    main()
