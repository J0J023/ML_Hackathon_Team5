"""
PathSnap: 5-Model Training, Cross-Validation, and Benchmarking Pipeline
Author: Team PathSnap
PES University - UE24CS352A Machine Learning Hackathon
"""

import os
import glob
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix

# Algorithm Families
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

FEATURE_COLS = [
    "dist_orthogonal",
    "proj_ratio",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "heading_cos",
    "speed",
    "time_gap",
    "is_time_gap"
]
TARGET_COL = "is_true_arc"


def load_datasets():
    """Loads preprocessed train and test feature matrices."""
    train_parquet = os.path.join(ARTIFACTS_DIR, "train_features.parquet")
    test_parquet = os.path.join(ARTIFACTS_DIR, "test_features.parquet")
    train_csv = os.path.join(ARTIFACTS_DIR, "train_features.csv.gz")
    test_csv = os.path.join(ARTIFACTS_DIR, "test_features.csv.gz")
    
    if os.path.exists(train_parquet):
        train_df = pd.read_parquet(train_parquet)
        test_df = pd.read_parquet(test_parquet)
    elif os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
    else:
        raise FileNotFoundError("Preprocessed feature files not found! Run src/preprocessor.py first.")
        
    return train_df, test_df


def compute_top1_observation_accuracy(df, y_pred_prob):
    """
    Computes observation-level Top-1 accuracy:
    For each (trip_id, obs_idx), picks the candidate with the highest predicted score.
    Returns 1.0 if that candidate has is_true_arc == 1, else 0.0.
    """
    temp_df = df[["trip_id", "obs_idx", "is_true_arc"]].copy()
    temp_df["prob"] = y_pred_prob
    
    # Pick max probability candidate per observation
    idx_max = temp_df.groupby(["trip_id", "obs_idx"])["prob"].idxmax()
    top1_choices = temp_df.loc[idx_max]
    
    top1_acc = top1_choices["is_true_arc"].mean()
    return top1_acc


def compute_route_continuity(df, y_pred_prob, arcs_dict):
    """
    Computes simple route continuity for classifier-only predictions:
    Percentage of consecutive observation top-1 arcs where arc_i -> arc_j is connected or identical.
    """
    temp_df = df[["trip_id", "obs_idx", "arc_id"]].copy()
    temp_df["prob"] = y_pred_prob
    idx_max = temp_df.groupby(["trip_id", "obs_idx"])["prob"].idxmax()
    top1 = temp_df.loc[idx_max].sort_values(["trip_id", "obs_idx"])
    
    total_transitions = 0
    valid_transitions = 0
    
    for tid, group in top1.groupby("trip_id"):
        arcs_seq = group["arc_id"].values
        trip_arcs = arcs_dict.get(tid)
        if trip_arcs is None or len(arcs_seq) < 2:
            continue
            
        for i in range(len(arcs_seq) - 1):
            a1, a2 = arcs_seq[i], arcs_seq[i+1]
            total_transitions += 1
            if a1 == a2:
                valid_transitions += 1
            elif a1 < len(trip_arcs) and a2 < len(trip_arcs):
                # Check if to_node of a1 == from_node of a2
                if trip_arcs[a1][1] == trip_arcs[a2][0]:
                    valid_transitions += 1
                    
    continuity = (valid_transitions / total_transitions * 100.0) if total_transitions > 0 else 0.0
    return continuity


def load_all_trip_arcs():
    """Loads arc endpoint mappings for all trips for continuity evaluation."""
    arcs_dict = {}
    for partition in ["train", "student_test"]:
        pdir = os.path.join(BASE_DIR, "Student folder", partition)
        for tdir in glob.glob(os.path.join(pdir, "*")):
            tid = os.path.basename(tdir)
            apath = os.path.join(tdir, f"{tid}.arcs")
            if os.path.exists(apath):
                with open(apath) as f:
                    arcs_dict[tid] = [list(map(int, l.strip().split())) for l in f if l.strip()]
    return arcs_dict


def get_model_definitions():
    """Defines 5 models representing distinct algorithm families."""
    models = {
        "Logistic Regression": {
            "family": "Linear and Probabilistic",
            "key_config": "C=1.0, solver='lbfgs', max_iter=500",
            "model": LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=42),
            "scale_features": True
        },
        "K-Nearest Neighbors": {
            "family": "Instance-based",
            "key_config": "n_neighbors=7, weights='distance'",
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
            "scale_features": True
        },
        "Decision Tree": {
            "family": "Tree-based",
            "key_config": "max_depth=12, min_samples_split=20",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_split=20, random_state=42),
            "scale_features": False
        },
        "Random Forest": {
            "family": "Ensemble",
            "key_config": "n_estimators=100, max_depth=14",
            "model": RandomForestClassifier(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42),
            "scale_features": False
        }
    }
    return models


def run_model_benchmark():
    print("=" * 60)
    print("PathSnap: Benchmarking 5 Distinct Algorithm Families...")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    train_df, test_df = load_datasets()
    print(f"Loaded Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    
    # Subsample training data if very large to ensure training completes comfortably before 11:30 AM
    if len(train_df) > 300000:
        print(f"Subsampling training dataset from {len(train_df):,} to 250,000 balanced rows for fast cross-validation...")
        pos = train_df[train_df[TARGET_COL] == 1]
        neg = train_df[train_df[TARGET_COL] == 0].sample(n=min(len(train_df) - len(pos), len(pos) * 5), random_state=42)
        train_df_sub = pd.concat([pos, neg], ignore_index=True)
    else:
        train_df_sub = train_df
        
    X_train_raw = train_df_sub[FEATURE_COLS].values
    y_train = train_df_sub[TARGET_COL].values
    groups_train = train_df_sub["trip_id"].values
    
    X_test_raw = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values
    
    # Feature scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    arcs_dict = load_all_trip_arcs()
    models = get_model_definitions()
    
    results = []
    trained_models = {}
    
    # GroupKFold (3-fold for speed during initial benchmark)
    gkf = GroupKFold(n_splits=3)
    
    for name, info in models.items():
        print(f"\n---> Training Model: {name} ({info['family']})")
        model = info["model"]
        use_scaled = info["scale_features"]
        X_tr = X_train_scaled if use_scaled else X_train_raw
        X_te = X_test_scaled if use_scaled else X_test_raw
        
        # 1. Group-Aware Cross-Validation
        cv_f1_scores = []
        cv_top1_scores = []
        t_cv_start = time.time()
        
        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tr, y_train, groups=groups_train)):
            X_fold_tr, y_fold_tr = X_tr[trn_idx], y_train[trn_idx]
            X_fold_val, y_fold_val = X_tr[val_idx], y_train[val_idx]
            val_sub_df = train_df_sub.iloc[val_idx]
            
            # Fit fold
            fold_clf = clone(model)
            if name == "K-Nearest Neighbors" and len(X_fold_tr) > 30000:
                sample_idx = np.random.choice(len(X_fold_tr), 30000, replace=False)
                fold_clf.fit(X_fold_tr[sample_idx], y_fold_tr[sample_idx])
            else:
                fold_clf.fit(X_fold_tr, y_fold_tr)
            
            val_probs = fold_clf.predict_proba(X_fold_val)[:, 1] if hasattr(fold_clf, "predict_proba") else fold_clf.decision_function(X_fold_val)
            val_preds = (val_probs >= 0.5).astype(int)
            
            f1 = f1_score(y_fold_val, val_preds, average="macro", zero_division=0)
            top1 = compute_top1_observation_accuracy(val_sub_df, val_probs)
            cv_f1_scores.append(f1)
            cv_top1_scores.append(top1)
            
        cv_macro_f1 = float(np.mean(cv_f1_scores))
        cv_top1_acc = float(np.mean(cv_top1_scores))
        
        # 2. Fit full model on all training data
        t_train_start = time.time()
        if name == "K-Nearest Neighbors" and len(X_tr) > 50000:
            sample_idx = np.random.choice(len(X_tr), 50000, replace=False)
            model.fit(X_tr[sample_idx], y_train[sample_idx])
        else:
            model.fit(X_tr, y_train)
        train_time = round(time.time() - t_train_start, 2)
        
        # 3. Student-Test Evaluation
        t_infer_start = time.time()
        test_probs = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_te)
        test_infer_time_ms = round((time.time() - t_infer_start) / len(X_te) * 1000.0, 4)
        
        test_preds = (test_probs >= 0.5).astype(int)
        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
        test_top1_acc = float(compute_top1_observation_accuracy(test_df, test_probs))
        test_log_loss = float(log_loss(y_test, test_probs)) if hasattr(model, "predict_proba") else np.nan
        route_continuity = float(compute_route_continuity(test_df, test_probs, arcs_dict))
        
        # Save model artifact
        model_filename = f"{name.lower().replace(' ', '_')}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        trained_models[name] = model
        
        res = {
            "Model": name,
            "Algorithm Family": info["family"],
            "Key Configuration": info["key_config"],
            "CV Macro F1-score": round(cv_macro_f1, 4),
            "Top-1 Accuracy": round(test_top1_acc, 4),
            "Route continuity": f"{round(route_continuity, 1)}%",
            "Student-Test Macro F1-score": round(test_macro_f1, 4),
            "Log Loss": round(test_log_loss, 4) if not np.isnan(test_log_loss) else "N/A",
            "Inference Time": f"{test_infer_time_ms:.3f} ms/obs",
            "Train Time (s)": train_time
        }
        results.append(res)
        print(f"  CV Macro F1: {cv_macro_f1:.4f} | Student Top-1 Acc: {test_top1_acc:.4f} | Test F1: {test_macro_f1:.4f} | Route Continuity: {route_continuity:.1f}%")
        
    # Build comparison DataFrame
    results_df = pd.DataFrame(results)
    
    # Save CSV and JSON
    table_csv = os.path.join(ARTIFACTS_DIR, "model_comparison_table.csv")
    results_df.to_csv(table_csv, index=False)
    
    with open(os.path.join(ARTIFACTS_DIR, "model_comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 90)
    print("CONSOLIDATED MODEL COMPARISON TABLE (Mandatory Section 3.1)")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)
    
    # Generate Visualizations
    plot_model_comparisons(results_df)
    plot_confusion_matrices(trained_models, models, X_test_raw, X_test_scaled, y_test)
    
    return results_df


def plot_model_comparisons(results_df):
    """Generates comparison bar plots required by Section 3.3."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Macro F1 comparison (CV vs Student-Test)
    plot_df = results_df[["Model", "CV Macro F1-score", "Student-Test Macro F1-score"]].melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette=["#4575b4", "#74add1"], ax=axes[0])
    axes[0].set_title("Cross-Validation vs. Student-Test Macro F1-Score", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha="right")
    
    # 2. Observation-level Top-1 Accuracy
    sns.barplot(data=results_df, x="Model", y="Top-1 Accuracy", palette="viridis", ax=axes[1])
    axes[1].set_title("Student-Testing Observation-Level Top-1 Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right")
    
    plt.tight_layout()
    comp_plot = os.path.join(PLOTS_DIR, "model_performance_comparison.png")
    plt.savefig(comp_plot, dpi=300)
    plt.close()
    print(f"[+] Comparison plot saved: {comp_plot}")


def plot_confusion_matrices(trained_models, models_info, X_test_raw, X_test_scaled, y_test):
    """Generates confusion matrices for every trained model."""
    fig, axes = plt.subplots(1, len(trained_models), figsize=(4 * len(trained_models), 3.5))
    
    for idx, (name, clf) in enumerate(trained_models.items()):
        use_scaled = models_info[name]["scale_features"]
        X_te = X_test_scaled if use_scaled else X_test_raw
        preds = clf.predict(X_te)
        cm = confusion_matrix(y_test, preds)
        
        ax = axes[idx] if len(trained_models) > 1 else axes
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if idx == 0 else "")
        ax.set_xticklabels(["Non-Arc (0)", "True Arc (1)"])
        ax.set_yticklabels(["Non-Arc (0)", "True Arc (1)"])
        
    plt.tight_layout()
    cm_plot = os.path.join(PLOTS_DIR, "model_confusion_matrices.png")
    plt.savefig(cm_plot, dpi=300)
    plt.close()
    print(f"[+] Confusion matrices saved: {cm_plot}")


if __name__ == "__main__":
    run_model_benchmark()
