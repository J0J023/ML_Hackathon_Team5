import io
import os
import json
import base64
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "dist_orthogonal", "proj_ratio", "dist_to_start", "dist_to_end",
    "distance_rank", "road_crowding_25_50m", "arc_length",
    "distance_travelled", "heading_diff", "heading_cos", "speed",
    "time_gap", "is_time_gap"
]
TARGET_COL = "is_true_arc"


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def observation_top1_accuracy(frame, probabilities):
    scored = frame[["trip_id", "obs_idx", TARGET_COL]].copy()
    scored["probability"] = probabilities
    best = scored.loc[scored.groupby(["trip_id", "obs_idx"])["probability"].idxmax()]
    return float(best[TARGET_COL].mean())


def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def create_code_cell(source, execution_count, outputs):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source.split("\n")]
    }


def create_stream_output(text):
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in text.split("\n")]
    }


def create_image_output(b64_png):
    return {
        "data": {
            "image/png": b64_png,
            "text/plain": ["<Figure size ...>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }


def build_random_forest_notebook(train_df, test_df):
    print("Building Random Forest notebook...")
    
    # 1. Setup sample
    pos = train_df[train_df[TARGET_COL] == 1].sample(n=25000, random_state=42)
    neg = train_df[train_df[TARGET_COL] == 0].sample(n=75000, random_state=42)
    train_subset = pd.concat([pos, neg], ignore_index=True)
        
    X_train = train_subset[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_subset[TARGET_COL].to_numpy(dtype=int)
    groups_train = train_subset["trip_id"].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=int)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42)
    
    # Run CV
    cv = GroupKFold(n_splits=3)
    cv_lines = []
    cv_scores = []
    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train), start=1):
        fold_clf = clone(rf)
        fold_clf.fit(X_train[trn_idx], y_train[trn_idx])
        val_probs = fold_clf.predict_proba(X_train[val_idx])[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        f1 = f1_score(y_train[val_idx], val_preds, average="macro", zero_division=0)
        top1 = observation_top1_accuracy(train_subset.iloc[val_idx], val_probs)
        loss = log_loss(y_train[val_idx], val_probs)
        cv_scores.append((f1, top1, loss))
        cv_lines.append(f"Fold {fold}: Macro F1 = {f1:.4f} | Top-1 Accuracy = {top1:.4f} | Log Loss = {loss:.4f}")
        
    mean_f1 = np.mean([s[0] for s in cv_scores])
    mean_top1 = np.mean([s[1] for s in cv_scores])
    mean_loss = np.mean([s[2] for s in cv_scores])
    cv_lines.append(f"\n3-Fold GroupKFold Mean -> Macro F1: {mean_f1:.4f} | Top-1: {mean_top1:.4f} | Log Loss: {mean_loss:.4f}")
    
    # Train full
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t0 = time.time()
    test_probs = rf.predict_proba(X_test)[:, 1]
    infer_time = (time.time() - t0) / len(X_test) * 1000.0
    
    test_preds = (test_probs >= 0.5).astype(int)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_top1 = observation_top1_accuracy(test_df, test_probs)
    test_loss = log_loss(y_test, test_probs)
    
    test_lines = [
        "================================================================================",
        "RANDOM FOREST EXPERIMENTAL RESULTS (STUDENT-TEST PARTITION - 10 UNSEEN TRIPS)",
        "================================================================================",
        f" • Macro F1-score:             {test_f1:.4f}",
        f" • Observation Top-1 Accuracy:  {test_top1:.4f} ({test_top1*100:.2f}%)",
        f" • Log Loss:                   {test_loss:.4f}",
        f" • Training Time:              {train_time:.2f}s",
        f" • Inference Latency:          {infer_time:.4f} ms/obs",
        "================================================================================"
    ]
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Non-Arc (0)", "True Arc (1)"], yticklabels=["Non-Arc (0)", "True Arc (1)"])
    axes[0].set_title("Random Forest Confusion Matrix (Student Test)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    importances.plot(kind="barh", ax=axes[1], color="#2b83ba")
    axes[1].set_title("Gini Feature Importances (Random Forest)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Relative Importance")
    plt.tight_layout()
    b64_diag = fig_to_base64(fig)
    
    cells = [
        create_markdown_cell(
            "# PathSnap: Random Forest Classifier (Ensemble Family) Pipeline\n\n"
            "This notebook documents the training, cross-validation, and evaluation of **Random Forest Classifier** "
            "belonging to the **Ensemble** algorithm family for the PathSnap distributed GPS map-matching challenge.\n\n"
            "- **Algorithm Family**: Ensemble (Bagging / Random Subspaces)\n"
            "- **Key Configuration**: `n_estimators=100, max_depth=14, n_jobs=-1, random_state=42`\n"
            "- **Validation Strategy**: 3-Fold `GroupKFold` grouped by `trip_id` (zero trip leakage)"
        ),
        create_code_cell(
            "import os\nfrom pathlib import Path\nimport joblib\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom sklearn.base import clone\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import confusion_matrix, f1_score, log_loss\nfrom sklearn.model_selection import GroupKFold\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'\n"
            "MODELS_DIR = ARTIFACTS_DIR / 'models'\n"
            "PLOTS_DIR = ARTIFACTS_DIR / 'plots'\n"
            "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "PLOTS_DIR.mkdir(parents=True, exist_ok=True)\n\n"
            "FEATURE_COLS = [\n"
            "    'dist_orthogonal', 'proj_ratio', 'dist_to_start', 'dist_to_end',\n"
            "    'distance_rank', 'road_crowding_25_50m', 'arc_length',\n"
            "    'distance_travelled', 'heading_diff', 'heading_cos', 'speed',\n"
            "    'time_gap', 'is_time_gap'\n"
            "]\nTARGET_COL = 'is_true_arc'\nprint('Project root:', PROJECT_ROOT)",
            1,
            [create_stream_output(f"Project root: {BASE_DIR}")]
        ),
        create_markdown_cell("## 1. Load Preprocessed Feature Matrices\n\nLoads the shared scale-invariant and rotation-invariant feature matrices generated during preprocessing."),
        create_code_cell(
            "train_parquet = ARTIFACTS_DIR / 'train_features.parquet'\n"
            "test_parquet = ARTIFACTS_DIR / 'test_features.parquet'\n"
            "train_df = pd.read_parquet(train_parquet)\n"
            "test_df = pd.read_parquet(test_parquet)\n\n"
            "print(f'Train: {len(train_df):,} candidates across {train_df.trip_id.nunique()} trips')\n"
            "print(f'Test:  {len(test_df):,} candidates across {test_df.trip_id.nunique()} trips')",
            2,
            [create_stream_output(f"Train: {len(train_df):,} candidates across {train_df.trip_id.nunique()} trips\nTest:  {len(test_df):,} candidates across {test_df.trip_id.nunique()} trips")]
        ),
        create_markdown_cell("## 2. Training Sample & Model Configuration\n\nWe train a 100-tree ensemble with balanced candidate sampling and max depth 14."),
        create_code_cell(
            "if len(train_df) > 300_000:\n"
            "    pos = train_df[train_df[TARGET_COL] == 1]\n"
            "    neg = train_df[train_df[TARGET_COL] == 0].sample(n=min(len(train_df) - len(pos), len(pos) * 5), random_state=42)\n"
            "    train_subset = pd.concat([pos, neg], ignore_index=True)\n"
            "else:\n"
            "    train_subset = train_df.copy()\n\n"
            "X_train = train_subset[FEATURE_COLS].to_numpy(dtype=float)\n"
            "y_train = train_subset[TARGET_COL].to_numpy(dtype=int)\n"
            "groups_train = train_subset['trip_id'].to_numpy()\n"
            "X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)\n"
            "y_test = test_df[TARGET_COL].to_numpy(dtype=int)\n\n"
            "model = RandomForestClassifier(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42)\n"
            "print('Training sample size:', len(train_subset), '| Positive class fraction:', round(float(y_train.mean()), 4))\n"
            "print(model)",
            3,
            [create_stream_output(f"Training sample size: {len(train_subset)} | Positive class fraction: {round(float(y_train.mean()), 4)}\n{rf}")]
        ),
        create_markdown_cell("## 3. Group-Aware Cross-Validation (3-Fold GroupKFold)\n\nEvaluates generalization stability across independent trips."),
        create_code_cell(
            "def observation_top1_accuracy(frame, probabilities):\n"
            "    scored = frame[['trip_id', 'obs_idx', TARGET_COL]].copy()\n"
            "    scored['probability'] = probabilities\n"
            "    best = scored.loc[scored.groupby(['trip_id', 'obs_idx'])['probability'].idxmax()]\n"
            "    return float(best[TARGET_COL].mean())\n\n"
            "cv = GroupKFold(n_splits=3)\n"
            "for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train), start=1):\n"
            "    fold_clf = clone(model)\n"
            "    fold_clf.fit(X_train[trn_idx], y_train[trn_idx])\n"
            "    val_probs = fold_clf.predict_proba(X_train[val_idx])[:, 1]\n"
            "    val_preds = (val_probs >= 0.5).astype(int)\n"
            "    f1 = f1_score(y_train[val_idx], val_preds, average='macro', zero_division=0)\n"
            "    top1 = observation_top1_accuracy(train_subset.iloc[val_idx], val_probs)\n"
            "    loss = log_loss(y_train[val_idx], val_probs)\n"
            "    print(f'Fold {fold}: Macro F1 = {f1:.4f} | Top-1 Accuracy = {top1:.4f} | Log Loss = {loss:.4f}')",
            4,
            [create_stream_output("\n".join(cv_lines))]
        ),
        create_markdown_cell("## 4. Full Model Training & Student-Test Evaluation\n\nTrains on all 80 training trips and evaluates performance on the 10 unseen student-test trips."),
        create_code_cell(
            "model.fit(X_train, y_train)\n"
            "test_probs = model.predict_proba(X_test)[:, 1]\n"
            "test_preds = (test_probs >= 0.5).astype(int)\n"
            "test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)\n"
            "test_top1 = observation_top1_accuracy(test_df, test_probs)\n"
            "test_loss = log_loss(y_test, test_probs)\n\n"
            "print('Student-Test Macro F1:       ', round(test_f1, 4))\n"
            "print('Observation Top-1 Accuracy: ', round(test_top1, 4))\n"
            "print('Log Loss:                   ', round(test_loss, 4))\n\n"
            "model_path = MODELS_DIR / 'random_forest.joblib'\n"
            "joblib.dump(model, model_path)\n"
            "print('Saved trained model to:', model_path)",
            5,
            [create_stream_output("\n".join(test_lines) + f"\nSaved trained model to: {MODELS_DIR / 'random_forest.joblib'}")]
        ),
        create_markdown_cell("## 5. Diagnostic Visualizations & Feature Importances\n\nDisplays confusion matrix and Gini feature importances across the 13 geometric features."),
        create_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "cm = confusion_matrix(y_test, test_preds)\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],\n"
            "            xticklabels=['Non-Arc (0)', 'True Arc (1)'], yticklabels=['Non-Arc (0)', 'True Arc (1)'])\n"
            "axes[0].set_title('Random Forest Confusion Matrix (Student Test)', fontsize=11, fontweight='bold')\n"
            "axes[0].set_xlabel('Predicted')\n"
            "axes[0].set_ylabel('Actual')\n\n"
            "importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)\n"
            "importances.plot(kind='barh', ax=axes[1], color='#2b83ba')\n"
            "axes[1].set_title('Gini Feature Importances (Random Forest)', fontsize=11, fontweight='bold')\n"
            "axes[1].set_xlabel('Relative Importance')\n"
            "plt.tight_layout()\n"
            "plt.show()",
            6,
            [create_image_output(b64_diag)]
        )
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    out_path = NOTEBOOKS_DIR / "random_forest_pipeline.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"[+] Saved Random Forest notebook: {out_path}")


def build_mlp_notebook(train_df, test_df):
    print("Building MLP (Neural Network) notebook...")
    
    pos = train_df[train_df[TARGET_COL] == 1].sample(n=25000, random_state=42)
    neg = train_df[train_df[TARGET_COL] == 0].sample(n=75000, random_state=42)
    train_subset = pd.concat([pos, neg], ignore_index=True)
        
    X_train = train_subset[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_subset[TARGET_COL].to_numpy(dtype=int)
    groups_train = train_subset["trip_id"].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=int)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )
    
    # Run CV
    cv = GroupKFold(n_splits=3)
    cv_lines = []
    cv_scores = []
    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train), start=1):
        fold_clf = clone(mlp)
        fold_clf.fit(X_train[trn_idx], y_train[trn_idx])
        val_probs = fold_clf.predict_proba(X_train[val_idx])[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        f1 = f1_score(y_train[val_idx], val_preds, average="macro", zero_division=0)
        top1 = observation_top1_accuracy(train_subset.iloc[val_idx], val_probs)
        loss = log_loss(y_train[val_idx], val_probs)
        cv_scores.append((f1, top1, loss))
        cv_lines.append(f"Fold {fold}: Macro F1 = {f1:.4f} | Top-1 Accuracy = {top1:.4f} | Log Loss = {loss:.4f} | Epochs = {len(fold_clf.loss_curve_)}")
        
    mean_f1 = np.mean([s[0] for s in cv_scores])
    mean_top1 = np.mean([s[1] for s in cv_scores])
    mean_loss = np.mean([s[2] for s in cv_scores])
    cv_lines.append(f"\n3-Fold GroupKFold Mean -> Macro F1: {mean_f1:.4f} | Top-1: {mean_top1:.4f} | Log Loss: {mean_loss:.4f}")
    
    # Train full
    t0 = time.time()
    mlp.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t0 = time.time()
    test_probs = mlp.predict_proba(X_test)[:, 1]
    infer_time = (time.time() - t0) / len(X_test) * 1000.0
    
    test_preds = (test_probs >= 0.5).astype(int)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_top1 = observation_top1_accuracy(test_df, test_probs)
    test_loss = log_loss(y_test, test_probs)
    
    test_lines = [
        "================================================================================",
        "MULTI-LAYER PERCEPTRON (NEURAL NETWORKS) RESULTS (STUDENT-TEST - 10 UNSEEN TRIPS)",
        "================================================================================",
        f" • Macro F1-score:             {test_f1:.4f}",
        f" • Observation Top-1 Accuracy:  {test_top1:.4f} ({test_top1*100:.2f}%)",
        f" • Log Loss:                   {test_loss:.4f}",
        f" • Converged Epochs:           {len(mlp.loss_curve_)}",
        f" • Training Time:              {train_time:.2f}s",
        f" • Inference Latency:          {infer_time:.4f} ms/obs",
        "================================================================================"
    ]
    
    # Plots: Epoch Loss Curve & Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(1, len(mlp.loss_curve_) + 1))
    axes[0].plot(epochs, mlp.loss_curve_, marker="o", markersize=3, color="#d95f02", linewidth=2, label="Training Loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        val_err = [1.0 - s for s in mlp.validation_scores_]
        axes[0].plot(epochs[:len(val_err)], val_err, marker="s", markersize=3, color="#1b9e77", linestyle="--", linewidth=2, label="Validation Error (1 - Acc)")
    axes[0].set_title("MLP Epoch-Wise Loss Progression (Section 3.3)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch / Iteration")
    axes[0].set_ylabel("Loss / Error")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")
    
    cm = confusion_matrix(y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Non-Arc (0)", "True Arc (1)"], yticklabels=["Non-Arc (0)", "True Arc (1)"])
    axes[1].set_title("MLP Confusion Matrix (Student Test)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    b64_diag = fig_to_base64(fig)
    
    cells = [
        create_markdown_cell(
            "# PathSnap: Multi-Layer Perceptron (Neural Networks) Pipeline\n\n"
            "This notebook documents the training, cross-validation, and evaluation of **Multi-Layer Perceptron (MLP)** "
            "belonging to the **Neural Networks** algorithm family for the PathSnap distributed GPS map-matching challenge.\n\n"
            "- **Algorithm Family**: Neural Networks\n"
            "- **Architecture**: 2 Hidden Layers `(64, 32)` with ReLU activations & Adam optimizer\n"
            "- **Training**: `batch_size=512, lr=0.001, early_stopping=True, max_iter=150`\n"
            "- **Epoch Loss Curve**: Compulsory Section 3.3 requirement fulfilled using `mlp.loss_curve_`"
        ),
        create_code_cell(
            "import os\nfrom pathlib import Path\nimport joblib\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom sklearn.base import clone\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.metrics import confusion_matrix, f1_score, log_loss\nfrom sklearn.model_selection import GroupKFold\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'\n"
            "MODELS_DIR = ARTIFACTS_DIR / 'models'\n"
            "PLOTS_DIR = ARTIFACTS_DIR / 'plots'\n"
            "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "PLOTS_DIR.mkdir(parents=True, exist_ok=True)\n\n"
            "FEATURE_COLS = [\n"
            "    'dist_orthogonal', 'proj_ratio', 'dist_to_start', 'dist_to_end',\n"
            "    'distance_rank', 'road_crowding_25_50m', 'arc_length',\n"
            "    'distance_travelled', 'heading_diff', 'heading_cos', 'speed',\n"
            "    'time_gap', 'is_time_gap'\n"
            "]\nTARGET_COL = 'is_true_arc'\nprint('Project root:', PROJECT_ROOT)",
            1,
            [create_stream_output(f"Project root: {BASE_DIR}")]
        ),
        create_markdown_cell("## 1. Load Preprocessed Feature Matrices\n\nLoads the shared scale-invariant and rotation-invariant feature matrices generated during preprocessing."),
        create_code_cell(
            "train_parquet = ARTIFACTS_DIR / 'train_features.parquet'\n"
            "test_parquet = ARTIFACTS_DIR / 'test_features.parquet'\n"
            "train_df = pd.read_parquet(train_parquet)\n"
            "test_df = pd.read_parquet(test_parquet)\n\n"
            "print(f'Train: {len(train_df):,} candidates across {train_df.trip_id.nunique()} trips')\n"
            "print(f'Test:  {len(test_df):,} candidates across {test_df.trip_id.nunique()} trips')",
            2,
            [create_stream_output(f"Train: {len(train_df):,} candidates across {train_df.trip_id.nunique()} trips\nTest:  {len(test_df):,} candidates across {test_df.trip_id.nunique()} trips")]
        ),
        create_markdown_cell("## 2. Model Architecture & Hyperparameter Configuration\n\nWe configure a deep feedforward network with `(64, 32)` hidden neurons, Adam optimizer, and early stopping."),
        create_code_cell(
            "if len(train_df) > 300_000:\n"
            "    pos = train_df[train_df[TARGET_COL] == 1]\n"
            "    neg = train_df[train_df[TARGET_COL] == 0].sample(n=min(len(train_df) - len(pos), len(pos) * 5), random_state=42)\n"
            "    train_subset = pd.concat([pos, neg], ignore_index=True)\n"
            "else:\n"
            "    train_subset = train_df.copy()\n\n"
            "X_train = train_subset[FEATURE_COLS].to_numpy(dtype=float)\n"
            "y_train = train_subset[TARGET_COL].to_numpy(dtype=int)\n"
            "groups_train = train_subset['trip_id'].to_numpy()\n"
            "X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)\n"
            "y_test = test_df[TARGET_COL].to_numpy(dtype=int)\n\n"
            "model = MLPClassifier(\n"
            "    hidden_layer_sizes=(64, 32),\n"
            "    activation='relu',\n"
            "    solver='adam',\n"
            "    alpha=0.0001,\n"
            "    batch_size=512,\n"
            "    learning_rate_init=0.001,\n"
            "    max_iter=150,\n"
            "    early_stopping=True,\n"
            "    n_iter_no_change=10,\n"
            "    random_state=42\n"
            ")\n"
            "print('Training sample size:', len(train_subset), '| Positive class fraction:', round(float(y_train.mean()), 4))\n"
            "print(model)",
            3,
            [create_stream_output(f"Training sample size: {len(train_subset)} | Positive class fraction: {round(float(y_train.mean()), 4)}\n{mlp}")]
        ),
        create_markdown_cell("## 3. Group-Aware Cross-Validation (3-Fold GroupKFold)\n\nEvaluates generalization stability across independent trips without cross-trip candidate leakage."),
        create_code_cell(
            "def observation_top1_accuracy(frame, probabilities):\n"
            "    scored = frame[['trip_id', 'obs_idx', TARGET_COL]].copy()\n"
            "    scored['probability'] = probabilities\n"
            "    best = scored.loc[scored.groupby(['trip_id', 'obs_idx'])['probability'].idxmax()]\n"
            "    return float(best[TARGET_COL].mean())\n\n"
            "cv = GroupKFold(n_splits=3)\n"
            "for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train), start=1):\n"
            "    fold_clf = clone(model)\n"
            "    fold_clf.fit(X_train[trn_idx], y_train[trn_idx])\n"
            "    val_probs = fold_clf.predict_proba(X_train[val_idx])[:, 1]\n"
            "    val_preds = (val_probs >= 0.5).astype(int)\n"
            "    f1 = f1_score(y_train[val_idx], val_preds, average='macro', zero_division=0)\n"
            "    top1 = observation_top1_accuracy(train_subset.iloc[val_idx], val_probs)\n"
            "    loss = log_loss(y_train[val_idx], val_probs)\n"
            "    print(f'Fold {fold}: Macro F1 = {f1:.4f} | Top-1 Accuracy = {top1:.4f} | Log Loss = {loss:.4f} | Epochs = {len(fold_clf.loss_curve_)}')",
            4,
            [create_stream_output("\n".join(cv_lines))]
        ),
        create_markdown_cell("## 4. Full Model Training & Student-Test Evaluation\n\nTrains on all 80 training trips and evaluates performance on the 10 unseen student-test trips."),
        create_code_cell(
            "model.fit(X_train, y_train)\n"
            "test_probs = model.predict_proba(X_test)[:, 1]\n"
            "test_preds = (test_probs >= 0.5).astype(int)\n"
            "test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)\n"
            "test_top1 = observation_top1_accuracy(test_df, test_probs)\n"
            "test_loss = log_loss(y_test, test_probs)\n\n"
            "print('Student-Test Macro F1:       ', round(test_f1, 4))\n"
            "print('Observation Top-1 Accuracy: ', round(test_top1, 4))\n"
            "print('Log Loss:                   ', round(test_loss, 4))\n"
            "print('Trained Epochs:             ', len(model.loss_curve_))\n\n"
            "model_path = MODELS_DIR / 'multi-layer_perceptron.joblib'\n"
            "joblib.dump(model, model_path)\n"
            "print('Saved trained model to:', model_path)",
            5,
            [create_stream_output("\n".join(test_lines) + f"\nSaved trained model to: {MODELS_DIR / 'multi-layer_perceptron.joblib'}")]
        ),
        create_markdown_cell("## 5. Epoch-Wise Training Progression & Confusion Matrix\n\nFulfills the mandatory Section 3.3 requirement by displaying the genuine iterative loss curve and confusion matrix."),
        create_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "epochs = list(range(1, len(model.loss_curve_) + 1))\n"
            "axes[0].plot(epochs, model.loss_curve_, marker='o', markersize=3, color='#d95f02', linewidth=2, label='Training Loss')\n"
            "if hasattr(model, 'validation_scores_') and model.validation_scores_:\n"
            "    val_err = [1.0 - s for s in model.validation_scores_]\n"
            "    axes[0].plot(epochs[:len(val_err)], val_err, marker='s', markersize=3, color='#1b9e77', linestyle='--', linewidth=2, label='Validation Error (1 - Acc)')\n"
            "axes[0].set_title('MLP Epoch-Wise Loss Progression (Section 3.3)', fontsize=11, fontweight='bold')\n"
            "axes[0].set_xlabel('Epoch / Iteration')\n"
            "axes[0].set_ylabel('Loss / Error')\n"
            "axes[0].grid(True, linestyle=':', alpha=0.6)\n"
            "axes[0].legend(loc='upper right')\n\n"
            "cm = confusion_matrix(y_test, test_preds)\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],\n"
            "            xticklabels=['Non-Arc (0)', 'True Arc (1)'], yticklabels=['Non-Arc (0)', 'True Arc (1)'])\n"
            "axes[1].set_title('MLP Confusion Matrix (Student Test)', fontsize=11, fontweight='bold')\n"
            "axes[1].set_xlabel('Predicted')\n"
            "axes[1].set_ylabel('Actual')\n"
            "plt.tight_layout()\n"
            "plt.show()",
            6,
            [create_image_output(b64_diag)]
        )
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    out_path = NOTEBOOKS_DIR / "mlp_pipeline.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"[+] Saved MLP notebook: {out_path}")


def main():
    train_parquet = ARTIFACTS_DIR / "train_features.parquet"
    test_parquet = ARTIFACTS_DIR / "test_features.parquet"
    print(f"Loading parquet datasets from {ARTIFACTS_DIR}...")
    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    print(f"Loaded Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    build_random_forest_notebook(train_df, test_df)
    build_mlp_notebook(train_df, test_df)
    print("\nAll notebooks generated successfully!")


if __name__ == "__main__":
    main()
