# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — Department of Computer Science and Engineering**  
**UE24CS352A: Machine Learning — Team 5**

---

## 1. Challenge Overview
GPS tracking systems suffer from urban building multipath reflections, signal dropouts, and ~1 Hz sampling noise, causing raw observations to drift onto parallel streets or across buildings.

**PathSnap** is an end-to-end distributed machine learning system that maps noisy GPS observations onto a directed road network and reconstructs continuous, topologically valid vehicle trajectories.

---

## 2. System Pipeline

```
[Raw GPS Track (.track) + Road Network (.nodes, .arcs)]
           │
           ▼
[1. Data Cleaning & Spatial Candidate Generation] (src/preprocessor.py — 5-point KD-Tree within 60m)
           │
           ▼
[2. Metric Projection & 13 Engineered Features] (dist_orthogonal, proj_ratio, heading_cos, speed, ...)
           │
           ▼
[3. 6-Model Supervised Benchmark] (src/train_models.py & notebooks/ — LR, GNB, KNN, DT, RF, MLP)
           │
           ▼
[4. Bayesian Hyperparameter Optimisation] (src/bayesian_opt_mlp.py — Optuna TPE, 25 trials)
           │
           ▼
[5. HMM & Viterbi Route Reconstruction] (src/hmm_decoder.py — Log-space trellis + Dijkstra transitions)
           │
           ▼
[6. Two-Laptop Deployment] (Laptop 1: FastAPI src/api.py <---> Laptop 2: Evaluator UI)
```

---

## 3. Repository Structure

```
ML_Hackathon_Team5/
├── docs/
│   ├── PathSnap_Technical_Report.md     # Full consolidated technical report (Markdown)
│   └── PathSnap_Technical_Report.pdf    # Consolidated technical report (PDF deliverable)
├── notebooks/                           # Individual model experimentation notebooks
│   ├── logistic_regression_pipeline.ipynb
│   ├── gaussian_naive_bayes_pipeline.ipynb
│   ├── knn_pathsnap.ipynb
│   ├── decision_tree_pipeline.ipynb
│   ├── random_forest_pipeline.ipynb
│   └── mlp_pipeline.ipynb
├── src/
│   ├── eda.py                           # Exploratory data analysis & Folium ground-truth map
│   ├── preprocessor.py                  # Cleaning, metric projection & 13-feature extraction
│   ├── train_models.py                  # GroupKFold cross-validation & multi-model benchmark
│   ├── bayesian_opt_mlp.py              # Optuna TPE Bayesian hyperparameter optimisation
│   ├── hmm_decoder.py                   # HMM + Viterbi route decoder & Folium route map
│   ├── generate_report_plots.py         # Diagnostic & comparison visualisations
│   └── api.py                           # FastAPI backend server for Laptop 1
├── artifacts/
│   ├── dataset_inventory.csv            # Audit of all 90 train/student_test trips
│   ├── eda_summary.json                 # Dataset summary statistics
│   ├── model_comparison_table.csv       # Consolidated model benchmark table
│   ├── bayesian_mlp_comparison_table.csv# Untuned vs. Bayesian-tuned MLP comparison
│   ├── optuna_mlp_trials.csv            # Full 25-trial Optuna search log
│   ├── optuna_mlp_best_params.json      # Best hyperparameters found by Optuna
│   ├── hmm_comparison_table.csv         # Classifier-only vs. HMM/Viterbi per-trip results
│   ├── maps/
│   │   ├── eda_map_trip_00000001.html
│   │   └── hmm_reconstructed_route_trip_00000000.html
│   ├── models/                          # Saved .joblib models & feature_transform.json
│   └── plots/                           # All EDA, model, Optuna, and HMM plots
├── requirements.txt
└── README.md
```

---

## 4. Summary of Experimental Results

### A. Model Comparison (`artifacts/model_comparison_table.csv`)
| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | `C=1.0, solver='lbfgs', max_iter=500` | 0.7837 | 0.7239 | 0.7954 | 0.2181 | 0.000 ms/obs |
| **Gaussian Naive Bayes** | Probabilistic | `var_smoothing=1e-9` | 0.5989 | 0.6918 | 0.6795 | 0.9653 | 0.000 ms/obs |
| **K-Nearest Neighbors** | Instance-based | `n_neighbors=7, weights='distance'` | 0.8436 | 0.7067 | 0.8201 | 0.7749 | 0.033 ms/obs |
| **Decision Tree** | Tree-based | `max_depth=12, min_samples_split=20` | 0.8534 | 0.7904 | 0.8433 | 0.2366 | 0.000 ms/obs |
| **Random Forest** | Ensemble | `n_estimators=100, max_depth=14` | **0.8551** | 0.7347 | 0.8490 | **0.1466** | 0.001 ms/obs |
| **Multi-Layer Perceptron** | Neural Networks | `hidden_layer_sizes=(64, 32), early_stopping=True` | 0.8529 | **0.8112** | **0.8508** | **0.1466** | 0.000 ms/obs |

### B. Bayesian Hyperparameter Optimisation (`artifacts/bayesian_mlp_comparison_table.csv`)
| Model Variant | Architecture | Activation | CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Log Loss | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned MLP** | `(64, 32)` | `relu` | 0.8529 | **80.18%** | **0.8458** | 0.1599 | **0.0002 ms/obs** |
| **Tuned MLP (Optuna Best)** | `(64, 64)` | `tanh` | **0.8900** | 77.61% | 0.8429 | **0.1594** | 0.0007 ms/obs |

---

## 5. Installation & Execution

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run EDA & Preprocessing
python src/eda.py
python src/preprocessor.py

# 3. Train & Benchmark Models + Bayesian Optimisation + HMM Decoding
python src/train_models.py
python src/bayesian_opt_mlp.py
python src/hmm_decoder.py
python src/generate_report_plots.py

# 4. Start the FastAPI Server on Laptop 1
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
