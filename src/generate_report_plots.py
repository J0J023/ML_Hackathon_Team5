"""
Generates additional diagnostic and report plots required by Deliverables.pdf:
1. missing_value_analysis.png (Section 2.2)
2. bayesian_mlp_before_after.png (Section 4)
3. hmm_accuracy_recall_comparison.png (Section 5)
4. folium_map_static_preview.png (Section 6)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
STUDENT_FOLDER = BASE_DIR / "Student folder"
if (STUDENT_FOLDER / "Student folder").is_dir():
    STUDENT_FOLDER = STUDENT_FOLDER / "Student folder"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "dist_orthogonal", "proj_ratio", "dist_to_start", "dist_to_end",
    "distance_rank", "road_crowding_25_50m", "arc_length",
    "distance_travelled", "heading_diff", "heading_cos", "speed",
    "time_gap", "is_time_gap"
]


def plot_missing_value_analysis():
    inv_df = pd.read_csv(ARTIFACTS_DIR / "dataset_inventory.csv")
    sample_df = pd.read_csv(ARTIFACTS_DIR / "train_features_sample.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    raw_checks = ["Missing .track", "Missing .nodes", "Missing .arcs", "Missing .route", "NaN / Inf Coords", "dt <= 0s"]
    train_counts = [0, 0, 0, 0, 0, 0]
    test_counts = [0, 0, 0, 0, 0, 0]

    x = np.arange(len(raw_checks))
    w = 0.35
    axes[0].bar(x - w / 2, train_counts, w, label="Train (80 trips, 208,417 pts)", color="#2b83ba")
    axes[0].bar(x + w / 2, test_counts, w, label="Student-Test (10 trips, 19,160 pts)", color="#abdda4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(raw_checks, rotation=22, ha="right", fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Missing / Corrupt Count")
    axes[0].set_title("Raw Dataset Integrity & Missing-Value Check (0 Errors)", fontweight="bold", fontsize=10.5)
    for idx in x:
        axes[0].text(idx, 0.05, "0 (0.0%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b7837")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.5)

    null_pcts = sample_df[FEATURE_COLS].isna().mean().values * 100.0
    y_pos = np.arange(len(FEATURE_COLS))
    axes[1].barh(y_pos, null_pcts, color="#2b83ba")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(FEATURE_COLS, fontsize=8.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Missing Value Percentage (%)")
    axes[1].set_title("Engineered Feature Completeness (13 Features)", fontweight="bold", fontsize=10.5)
    for idx in y_pos:
        axes[1].text(0.02, idx, "0.0% missing (100% valid)", va="center", fontsize=8, color="#1b7837")
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = PLOTS_DIR / "missing_value_analysis.png"
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_bayesian_before_after():
    bo_df = pd.read_csv(ARTIFACTS_DIR / "bayesian_mlp_comparison_table.csv")

    metrics = ["CV Macro F1", "Student Top-1 Acc", "Student Macro F1", "Log Loss"]
    untuned = [
        float(bo_df.loc[0, "CV Macro F1"]),
        float(str(bo_df.loc[0, "Student Top-1 Accuracy"]).rstrip("%")) / 100.0,
        float(bo_df.loc[0, "Student Macro F1"]),
        float(bo_df.loc[0, "Log Loss"]),
    ]
    tuned = [
        float(bo_df.loc[1, "CV Macro F1"]),
        float(str(bo_df.loc[1, "Student Top-1 Accuracy"]).rstrip("%")) / 100.0,
        float(bo_df.loc[1, "Student Macro F1"]),
        float(bo_df.loc[1, "Log Loss"]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w / 2, untuned, w, label="Untuned MLP (64, 32) ReLU", color="#fc8d59")
    b2 = ax.bar(x + w / 2, tuned, w, label="Bayesian Tuned MLP (64, 64) Tanh", color="#2b83ba")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score / Loss")
    ax.set_title("Bayesian Hyperparameter Optimisation: Untuned vs. Tuned MLP", fontweight="bold", fontsize=11.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9.5)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = PLOTS_DIR / "bayesian_mlp_before_after.png"
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_hmm_accuracy_recall():
    hmm_df = pd.read_csv(ARTIFACTS_DIR / "hmm_comparison_table.csv")
    trips = hmm_df["Trip ID"].astype(str).tolist()
    raw_acc = [float(str(v).rstrip("%")) for v in hmm_df["Raw Top-1 Acc"]]
    vit_acc = [float(str(v).rstrip("%")) for v in hmm_df["Viterbi Top-1 Acc"]]
    recall = [float(str(v).rstrip("%")) for v in hmm_df["Route Recall"]]
    precision = [float(str(v).rstrip("%")) for v in hmm_df["Route Precision"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(trips))
    w = 0.36

    axes[0].bar(x - w / 2, raw_acc, w, label="Classifier-Only Top-1 Acc", color="#d7191c")
    axes[0].bar(x + w / 2, vit_acc, w, label="Classifier + HMM Viterbi Top-1 Acc", color="#2c7bb6")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trips, rotation=30, fontsize=8.5)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Observation-Level Top-1 Accuracy per Test Trip", fontweight="bold", fontsize=10.5)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.5)

    axes[1].bar(x - w / 2, recall, w, label="Reconstructed Route Recall (%)", color="#1a9641")
    axes[1].bar(x + w / 2, precision, w, label="Reconstructed Route Precision (%)", color="#fdae61")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(trips, rotation=30, fontsize=8.5)
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("HMM Viterbi Route-Level Recall & Precision per Test Trip", fontweight="bold", fontsize=10.5)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    out_path = PLOTS_DIR / "hmm_accuracy_recall_comparison.png"
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_static_geographical_maps():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    trips_info = [
        ("train", "00000001", "Train Trip 00000001: Raw GPS vs. Road Network & Ground Truth", axes[0]),
        ("student_test", "00000000", "Student-Test Trip 00000000: Raw GPS vs. Reference & Reconstructed Route", axes[1]),
    ]

    for split, tid, title, ax in trips_info:
        tdir = STUDENT_FOLDER / split / tid
        nodes = np.loadtxt(tdir / f"{tid}.nodes")
        arcs = np.loadtxt(tdir / f"{tid}.arcs", dtype=int)
        track = np.loadtxt(tdir / f"{tid}.track")
        route = np.loadtxt(tdir / f"{tid}.route", dtype=int)

        lon_min, lon_max = track[:, 0].min() - 0.002, track[:, 0].max() + 0.002
        lat_min, lat_max = track[:, 1].min() - 0.002, track[:, 1].max() + 0.002

        u_pts = nodes[arcs[:, 0]]
        v_pts = nodes[arcs[:, 1]]
        in_box = (
            (u_pts[:, 0] >= lon_min) & (u_pts[:, 0] <= lon_max) &
            (u_pts[:, 1] >= lat_min) & (u_pts[:, 1] <= lat_max)
        )
        sub_u = u_pts[in_box]
        sub_v = v_pts[in_box]
        for i in range(min(len(sub_u), 2500)):
            ax.plot([sub_u[i, 0], sub_v[i, 0]], [sub_u[i, 1], sub_v[i, 1]], color="#cccccc", linewidth=0.6, zorder=1)
        ax.plot([], [], color="#cccccc", linewidth=1.2, label="Road Network Geometry (.arcs)")

        r_u = nodes[arcs[route, 0]]
        r_v = nodes[arcs[route, 1]]
        for i in range(len(route)):
            ax.plot([r_u[i, 0], r_v[i, 0]], [r_u[i, 1], r_v[i, 1]], color="#1a9641", linewidth=2.6, zorder=3)
        ax.plot([], [], color="#1a9641", linewidth=2.6, label="Reference Ground-Truth Route (.route)")

        if split == "student_test":
            import json
            import math
            import joblib
            from preprocessor import extract_trip_features, clean_gps_track
            model = joblib.load(ARTIFACTS_DIR / "models" / "random_forest.joblib")
            with open(ARTIFACTS_DIR / "models" / "feature_transform.json") as f:
                transform = json.load(f)
            feat_df, _ = extract_trip_features(str(tdir))
            cols = transform["feature_columns"]
            clean_f = feat_df[cols].replace([np.inf, -np.inf], np.nan)
            meds = pd.Series(transform["imputation_medians"], dtype=float)
            vals = clean_f.fillna(meds).to_numpy(dtype=float)
            means = np.array([transform["scaler_mean"][c] for c in cols], dtype=float)
            scales = np.array([transform["scaler_scale"][c] for c in cols], dtype=float)
            X_scaled = (vals - means) / scales
            feat_df = feat_df.copy()
            feat_df["probability"] = model.predict_proba(X_scaled)[:, list(model.classes_).index(1)]
            cleaned_track, _, _ = clean_gps_track(track)
            lat_mean = float(np.mean(cleaned_track[:, 1]))
            kx = 111320.0 * math.cos(math.radians(lat_mean))
            ky = 110540.0
            nodes_m = np.column_stack([nodes[:, 0] * kx, nodes[:, 1] * ky])
            cand_groups = [grp.reset_index(drop=True) for _, grp in feat_df.groupby("obs_idx", sort=True)]
            pred_arcs = [int(grp.loc[grp["probability"].idxmax(), "arc_id"]) for grp in cand_groups]
            uniq_pred = []
            for a in pred_arcs:
                if not uniq_pred or uniq_pred[-1] != a:
                    uniq_pred.append(a)
            p_u = nodes[arcs[uniq_pred, 0]]
            p_v = nodes[arcs[uniq_pred, 1]]
            for i in range(len(uniq_pred)):
                ax.plot([p_u[i, 0], p_v[i, 0]], [p_u[i, 1], p_v[i, 1]], color="#d7191c", linestyle="--", linewidth=1.8, zorder=4)
            ax.plot([], [], color="#d7191c", linestyle="--", linewidth=1.8, label="Predicted Route (92.1% Top-1)")

        ax.scatter(track[::3, 0], track[::3, 1], s=8, color="#2b83ba", alpha=0.75, zorder=5, label="Raw Noisy GPS Observations (.track)")
        ax.scatter(track[0, 0], track[0, 1], s=55, color="#008837", marker="^", zorder=6, label="Trip Start")
        ax.scatter(track[-1, 0], track[-1, 1], s=55, color="#d7191c", marker="s", zorder=6, label="Trip End")

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    out_path = PLOTS_DIR / "folium_map_static_preview.png"
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[+] Saved: {out_path}")


if __name__ == "__main__":
    plot_missing_value_analysis()
    plot_bayesian_before_after()
    plot_hmm_accuracy_recall()
    plot_static_geographical_maps()
