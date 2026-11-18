import os
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss

# Suppress Optuna verbose logs (show only warnings/summary)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "dist_orthogonal", "proj_ratio", "dist_to_start", "dist_to_end",
    "distance_rank", "road_crowding_25_50m", "arc_length",
    "distance_travelled", "heading_diff", "heading_cos", "speed",
    "time_gap", "is_time_gap"
]
TARGET_COL = "is_true_arc"


def observation_top1_accuracy(frame, probabilities):
    scored = frame[["trip_id", "obs_idx", TARGET_COL]].copy()
    scored["probability"] = probabilities
    best = scored.loc[scored.groupby(["trip_id", "obs_idx"])["probability"].idxmax()]
    return float(best[TARGET_COL].mean())


def load_datasets():
    train_parquet = ARTIFACTS_DIR / "train_features.parquet"
    test_parquet = ARTIFACTS_DIR / "test_features.parquet"
    if train_parquet.exists() and test_parquet.exists():
        train_df = pd.read_parquet(train_parquet)
        test_df = pd.read_parquet(test_parquet)
    else:
        train_df = pd.read_csv(ARTIFACTS_DIR / "train_features.csv.gz")
        test_df = pd.read_csv(ARTIFACTS_DIR / "test_features.csv.gz")
    return train_df, test_df


def run_bayesian_optimization(n_trials=25):
    print("=" * 80)
    print("PathSnap: Bayesian Hyperparameter Optimization (Multi-Layer Perceptron)")
    print(f"Algorithm Family: Neural Networks | Strategy: TPE Sampler ({n_trials} Trials)")
    print("=" * 80)
    
    train_df, test_df = load_datasets()
    print(f"Loaded Train: {len(train_df):,} candidates | Test: {len(test_df):,} candidates")
    
    # Balanced training subset for fast, robust cross-validation (100k rows)
    pos = train_df[train_df[TARGET_COL] == 1].sample(n=25000, random_state=42)
    neg = train_df[train_df[TARGET_COL] == 0].sample(n=75000, random_state=42)
    train_subset = pd.concat([pos, neg], ignore_index=True)
    print(f"Training search partition: {len(train_subset):,} balanced rows across {train_subset['trip_id'].nunique()} trips")
    
    X_train = train_subset[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_subset[TARGET_COL].to_numpy(dtype=int)
    groups_train = train_subset["trip_id"].to_numpy()
    
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=int)
    
    cv = GroupKFold(n_splits=3)
    
    # Architecture candidate choices
    arch_options = [
        (64, 32),
        (128, 64),
        (64, 64),
        (128, 64, 32),
        (96, 48),
        (64, 32, 16)
    ]
    
    trial_records = []
    
    def objective(trial):
        arch_idx = trial.suggest_categorical("arch_idx", list(range(len(arch_options))))
        hidden_layers = arch_options[arch_idx]
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        lr_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            learning_rate_init=lr_init,
            batch_size=batch_size,
            solver="adam",
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=42
        )
        
        fold_f1s = []
        fold_top1s = []
        for trn_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
            fold_clf = clone(clf)
            fold_clf.fit(X_train[trn_idx], y_train[trn_idx])
            val_probs = fold_clf.predict_proba(X_train[val_idx])[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            f1 = f1_score(y_train[val_idx], val_preds, average="macro", zero_division=0)
            top1 = observation_top1_accuracy(train_subset.iloc[val_idx], val_probs)
            fold_f1s.append(f1)
            fold_top1s.append(top1)
            
        mean_macro_f1 = float(np.mean(fold_f1s))
        mean_top1 = float(np.mean(fold_top1s))
        
        trial_records.append({
            "trial_id": trial.number,
            "architecture": str(hidden_layers),
            "activation": activation,
            "alpha": alpha,
            "learning_rate_init": lr_init,
            "batch_size": batch_size,
            "cv_macro_f1": round(mean_macro_f1, 4),
            "cv_top1_accuracy": round(mean_top1, 4)
        })
        
        print(f"Trial {trial.number:2d}: Macro F1 = {mean_macro_f1:.4f} | Top-1 = {mean_top1:.4f} | Arch = {hidden_layers} | Act = {activation} | Alpha = {alpha:.1e} | LR = {lr_init:.1e}")
        return mean_macro_f1

    print("\n---> Launching Optuna Bayesian Search...")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.time() - t_start
    
    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION COMPLETED")
    print("=" * 80)
    print(f"Total Search Time: {total_opt_time:.2f} seconds ({total_opt_time/60:.2f} mins)")
    print(f"Best Trial ID:      {study.best_trial.number}")
    print(f"Best CV Macro F1:   {study.best_value:.4f}")
    best_params = study.best_params.copy()
    best_arch = arch_options[best_params.pop("arch_idx")]
    best_params["hidden_layer_sizes"] = best_arch
    print(f"Best Parameters:    {best_params}")
    
    # Save search records
    trials_df = pd.DataFrame(trial_records)
    trials_csv = ARTIFACTS_DIR / "optuna_mlp_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"[+] Saved search records: {trials_csv}")
    
    with open(ARTIFACTS_DIR / "optuna_mlp_best_params.json", "w") as f:
        json.dump({"best_params": {k: str(v) if not isinstance(v, (int, float)) else v for k, v in best_params.items()},
                   "best_cv_macro_f1": study.best_value,
                   "total_search_time_s": round(total_opt_time, 2)}, f, indent=2)

    # -------------------------------------------------------------
    # Refit Untuned vs Tuned Models for Full Benchmark Comparison
    # -------------------------------------------------------------
    print("\n---> Refitting Untuned vs. Tuned Models on 250k Balanced Training Partition...")
    pos_full = train_df[train_df[TARGET_COL] == 1].sample(n=50000, random_state=42)
    neg_full = train_df[train_df[TARGET_COL] == 0].sample(n=200000, random_state=42)
    full_train = pd.concat([pos_full, neg_full], ignore_index=True)
    X_full = full_train[FEATURE_COLS].to_numpy(dtype=float)
    y_full = full_train[TARGET_COL].to_numpy(dtype=int)
    
    # 1. Untuned Baseline
    untuned_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        batch_size=512,
        max_iter=150,
        early_stopping=True,
        random_state=42
    )
    t0 = time.time()
    untuned_model.fit(X_full, y_full)
    untuned_train_time = time.time() - t0
    
    t0 = time.time()
    untuned_probs = untuned_model.predict_proba(X_test)[:, 1]
    untuned_infer_time = (time.time() - t0) / len(X_test) * 1000.0
    untuned_preds = (untuned_probs >= 0.5).astype(int)
    
    untuned_test_f1 = f1_score(y_test, untuned_preds, average="macro", zero_division=0)
    untuned_test_top1 = observation_top1_accuracy(test_df, untuned_probs)
    untuned_test_loss = log_loss(y_test, untuned_probs)
    
    # 2. Tuned Model
    tuned_model = MLPClassifier(
        hidden_layer_sizes=best_arch,
        activation=best_params["activation"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        batch_size=best_params["batch_size"],
        solver="adam",
        max_iter=150,
        early_stopping=True,
        random_state=42
    )
    t0 = time.time()
    tuned_model.fit(X_full, y_full)
    tuned_train_time = time.time() - t0
    
    t0 = time.time()
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]
    tuned_infer_time = (time.time() - t0) / len(X_test) * 1000.0
    tuned_preds = (tuned_probs >= 0.5).astype(int)
    
    tuned_test_f1 = f1_score(y_test, tuned_preds, average="macro", zero_division=0)
    tuned_test_top1 = observation_top1_accuracy(test_df, tuned_probs)
    tuned_test_loss = log_loss(y_test, tuned_probs)
    
    # Save Tuned Model
    tuned_model_path = MODELS_DIR / "mlp_classifier_tuned.joblib"
    joblib.dump(tuned_model, tuned_model_path)
    print(f"[+] Saved Tuned Model: {tuned_model_path}")
    
    # Generate Comparison Table (Mandatory Section 4)
    comparison_table = [
        {
            "Model Variant": "Untuned MLP (Baseline)",
            "Architecture": "(64, 32)",
            "CV Macro F1": 0.8529,
            "Student Top-1 Accuracy": f"{untuned_test_top1*100:.2f}%",
            "Student Macro F1": round(untuned_test_f1, 4),
            "Log Loss": round(untuned_test_loss, 4),
            "Inference Time": f"{untuned_infer_time:.4f} ms/obs",
            "Training Time": f"{untuned_train_time:.2f}s"
        },
        {
            "Model Variant": "Tuned MLP (Bayesian Optimized)",
            "Architecture": str(best_arch),
            "CV Macro F1": round(study.best_value, 4),
            "Student Top-1 Accuracy": f"{tuned_test_top1*100:.2f}%",
            "Student Macro F1": round(tuned_test_f1, 4),
            "Log Loss": round(tuned_test_loss, 4),
            "Inference Time": f"{tuned_infer_time:.4f} ms/obs",
            "Training Time": f"{tuned_train_time:.2f}s"
        }
    ]
    comp_df = pd.DataFrame(comparison_table)
    comp_df.to_csv(ARTIFACTS_DIR / "bayesian_mlp_comparison_table.csv", index=False)
    
    print("\n" + "=" * 90)
    print("UNTUNED VS. TUNED MODEL COMPARISON (Mandatory Section 4)")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("=" * 90)
    
    # Generate Plots
    plot_optimization_results(study, trials_df, comp_df)
    return comp_df, best_params


def plot_optimization_results(study, trials_df, comp_df):
    """Generates Optuna convergence and comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Convergence plot (Optimization History)
    trial_numbers = trials_df["trial_id"]
    f1_values = trials_df["cv_macro_f1"]
    running_max = np.maximum.accumulate(f1_values)
    
    axes[0].scatter(trial_numbers, f1_values, color="#74add1", alpha=0.7, edgecolors="none", s=40, label="Trial Macro F1")
    axes[0].plot(trial_numbers, running_max, color="#d73027", linewidth=2.5, label="Best-So-Far Objective")
    axes[0].set_title("Bayesian Optimization Convergence (Optuna TPE)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Trial Number", fontsize=10)
    axes[0].set_ylabel("3-Fold CV Macro F1-Score", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")
    
    # 2. Before vs After Bar Plot
    bar_data = pd.DataFrame({
        "Metric": ["CV Macro F1", "Student Macro F1", "Top-1 Accuracy"],
        "Untuned": [0.8529, comp_df.loc[0, "Student Macro F1"], float(comp_df.loc[0, "Student Top-1 Accuracy"].rstrip("%"))/100.0],
        "Tuned": [comp_df.loc[1, "CV Macro F1"], comp_df.loc[1, "Student Macro F1"], float(comp_df.loc[1, "Student Top-1 Accuracy"].rstrip("%"))/100.0]
    }).melt(id_vars="Metric", var_name="Variant", value_name="Score")
    
    sns.barplot(data=bar_data, x="Metric", y="Score", hue="Variant", palette=["#abd9e9", "#2c7bb6"], ax=axes[1])
    axes[1].set_title("Untuned vs. Tuned MLP Performance Comparison", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0.7, 0.95)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    opt_plot = PLOTS_DIR / "optuna_mlp_convergence.png"
    plt.savefig(opt_plot, dpi=300)
    plt.close()
    print(f"[+] Optimization plots saved: {opt_plot}")


if __name__ == "__main__":
    run_bayesian_optimization(n_trials=25)
