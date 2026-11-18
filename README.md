# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — Department of Computer Science and Engineering**  
**UE24CS352A: Machine Learning — Final Hackathon Submission**  
**Team 5 Repository**

---

## 1. Challenge Overview
GPS tracking systems inherently suffer from urban canyon multipath reflections, signal dropout, and low sampling frequencies (~1 Hz). As a result, noisy observations deviate onto parallel streets, jump across obstacles, or violate topological road network constraints.

**PathSnap** is an end-to-end, distributed machine learning system that accurately matches noisy raw GPS trajectories to an underlying road network graph and reconstructs valid, topologically continuous vehicle trajectories.

---

## 2. System Pipeline

```
[Raw GPS Track (.track) + Road Network (.nodes, .arcs)]
           │
           ▼
[1. Spatial Candidate Generation] (KD-Tree spatial indexing within search radius R = 60m)
           │
           ▼
[2. Invariant Feature Extraction] (Orthogonal distance, projection ratio, heading error, kinematics)
           │
           ▼
[3. ML Scoring Classifier] (Ensemble Random Forest & Decision Tree; Bayesian-Tuned MLP)
           │
           ▼ (Emission probabilities)
[4. Sequential HMM & Viterbi Decoder] (Topological arc connectivity & metric movement distance)
           │
           ▼
[5. Reconstructed Continuous Route]
           │
           ▼
[Two-Laptop Distributed Deployment]
  Laptop 1 (Model Server & API on :8000) <==== LAN ====> Laptop 2 (Evaluator Client UI on :3000)
```

---

## 3. Repository Structure

```
ML_Hackathon_Team5/
├── docs/
│   ├── PathSnap_Technical_Report.md     # Full consolidated technical report (Markdown)
│   └── PathSnap_Technical_Report.pdf    # Consolidated technical report (PDF deliverable)
├── report/
│   └── technical_report.pdf             # Deliverable 2: Consolidated Technical Report PDF
├── notebooks/                           # Executed model & analysis notebooks
│   ├── decision_tree_pipeline.ipynb
│   ├── random_forest_pipeline.ipynb
│   ├── mlp_pipeline.ipynb
│   ├── logistic_regression_pipeline.ipynb
│   ├── gaussian_naive_bayes_pipeline.ipynb
│   ├── knn_pathsnap.ipynb
│   ├── bayesian_optimization_mlp.ipynb
│   └── hmm_viterbi_pipeline.ipynb
├── ui/                                  # Evaluator interface (Laptop 2 Client)
│   ├── index.html                       # Responsive Tailwind + Leaflet dashboard
│   └── app.py                           # Lightweight local UI server runner
├── src/
│   ├── api.py                           # FastAPI distributed backend & Viterbi engine
│   ├── preprocessor.py                  # Candidate generator & feature extraction
│   ├── train_models.py                  # GroupKFold training across all 5 model families
│   ├── bayesian_opt_mlp.py              # Optuna Bayesian hyperparameter optimization
│   ├── hmm_decoder.py                   # HMM Viterbi trellis dynamic programming
│   ├── eda.py                           # Exploratory Data Analysis & Folium generator
│   ├── generate_report_plots.py         # Diagnostic & comparison visualisations
│   └── generate_pdf_report.py           # Automated compiler for Technical Report PDF
├── artifacts/
│   ├── dataset_inventory.csv            # Audit of all 90 trip partitions
│   ├── eda_summary.json                 # Key dataset statistics and sampling properties
│   ├── model_comparison_table.csv       # Phase 1 model comparison across 5 families
│   ├── hmm_comparison_table.csv         # HMM Viterbi evaluation across student-test trips
│   ├── bayesian_mlp_comparison_table.csv # Bayesian tuned vs untuned MLP comparison
│   ├── optuna_mlp_trials.csv            # Full 25-trial Optuna optimization history
│   ├── optuna_mlp_best_params.json      # Optimal MLP hyperparameters
│   ├── maps/                            # Interactive Folium/Leaflet maps (OpenStreetMap)
│   │   ├── eda_map_trip_00000001.html
│   │   └── hmm_reconstructed_route_trip_00000000.html
│   ├── models/                          # Serialized model weights & feature transform
│   │   ├── random_forest.joblib         # Active production inference model
│   │   ├── decision_tree.joblib
│   │   ├── multi-layer_perceptron.joblib
│   │   ├── mlp_classifier_tuned.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── k-nearest_neighbors.joblib
│   │   ├── gaussian_naive_bayes.joblib
│   │   └── feature_transform.json
│   └── plots/                           # Evaluation charts and diagnostics
│       ├── trajectory_distributions.png
│       ├── road_network_complexity.png
│       ├── sampling_interval_dist.png
│       ├── tree_feature_importances.png
│       ├── model_performance_comparison.png
│       ├── model_confusion_matrices.png
│       ├── optuna_mlp_convergence.png
│       ├── mlp_training_loss_curve.png
│       ├── hmm_continuity_comparison.png
│       ├── hmm_accuracy_recall_comparison.png
│       ├── missing_value_analysis.png
│       ├── bayesian_mlp_before_after.png
│       └── map_visualization_trip_00000015.png
├── evaluator_test/                      # 10 official reserved evaluation trips
│   └── [00000015, 00000017, 00000023, 00000030, 00000049,
│        00000059, 00000067, 00000073, 00000087, 00000094]
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Two-Laptop Deployment Setup (Evaluation Instructions)

As mandated by Section 3 & 6 of the Hackathon Guidelines, PathSnap is split across two distinct machines on the same local network:

### Laptop 1: Model Server & API Backend
Laptop 1 hosts the trained model, feature scalers, HMM/Viterbi engine, road network data, server-side evaluation, and terminal CMD logs.

1. **Find Laptop 1's Local IP Address**:
   ```bash
   # On macOS / Linux:
   ifconfig | grep "inet " | grep -v 127.0.0.1
   # On Windows (cmd/PowerShell):
   ipconfig
   ```
   *(e.g., `192.168.1.100`)*

2. **Start the FastAPI Backend**:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```
   *The server listens on all interfaces (`0.0.0.0`) with CORS enabled (`allow_origins=["*"]`).*

3. **Keep the Terminal (CMD) Visible for Evaluators**:
   Every time an inference request arrives, the terminal prints live formatted logs:
   ```
   ================================================================================
     [PATHSNAP INFERENCE REQUEST] 2026-11-18 14:48:01
     Client Host : 192.168.1.105 (Laptop 2)
     Target Trip : 00000015 (Resolved partition: evaluator_test)
   --------------------------------------------------------------------------------
     Nodes : 1,571 | Arcs : 3,225 | GPS Points : 6,854 | Candidates : 57,769
     Classifier-Only Top-1 Acc    : 53.63%
     HMM Viterbi Decoded Acc      : 73.72% (Gain: +20.09%)
     Reconstructed Route Continuity: 99.85%
     Total Processing Latency     : 5706.62 ms (Viterbi: 1936.34 ms)
   ================================================================================
   ```

---

### Laptop 2: Evaluator User Interface (Client)
Laptop 2 provides the evaluator interface displaying buttons for the **10 evaluation trips**, live scorecards, and interactive Leaflet map.

**Option A (Direct in Browser — Zero Install on Laptop 2)**:
1. Open Google Chrome or any modern browser on Laptop 2.
2. Navigate to: `http://<laptop1_ip>:8000/` *(e.g. `http://192.168.1.100:8000/`)*.
3. The full dashboard loads instantly.

**Option B (Dedicated UI Client Runner)**:
1. From the repository root on Laptop 2, run:
   ```bash
   python ui/app.py --server http://<laptop1_ip>:8000 --port 3000
   ```
2. Your browser will automatically open to `http://localhost:3000`.

---

## 5. Live Demonstration on the 10 Evaluator Trips

The evaluator may click **ANY** of the 10 evaluation trips randomly, or click the **🎲 Random Evaluator Trip** button:

| Evaluator Trip | Nodes | Road Arcs | GPS Pings | Classifier Acc | HMM Viterbi Acc | Accuracy Gain | Route Continuity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00000015** | 1,571 | 3,225 | 6,854 | 53.63% | **73.72%** | **+20.09%** | **99.85%** |
| **00000017** | 36,724 | 76,425 | 1,346 | 87.07% | **92.05%** | **+4.98%** | **91.52%** |
| **00000023** | 5,053 | 10,596 | 458 | 89.30% | **90.83%** | **+1.53%** | **95.84%** |
| **00000030** | 13,223 | 24,218 | 654 | 77.06% | **83.79%** | **+6.73%** | **93.72%** |
| **00000049** | 23,165 | 47,542 | 2,642 | 96.44% | **99.32%** | **+2.88%** | **99.43%** |
| **00000059** | 12,631 | 24,024 | 1,031 | 91.17% | **95.83%** | **+4.66%** | **95.73%** |
| **00000067** | 2,265 | 4,485 | 916 | 72.05% | **79.37%** | **+7.32%** | **99.67%** |
| **00000073** | 12,151 | 25,631 | 1,144 | 91.78% | **98.08%** | **+6.30%** | **99.56%** |
| **00000087** | 55,234 | 107,451 | 1,399 | 91.71% | **97.50%** | **+5.79%** | **98.43%** |
| **00000094** | 6,815 | 14,931 | 3,230 | 74.58% | **86.44%** | **+11.86%** | **98.61%** |

*All 10 evaluator trips achieve sub-second to low-second inference latency, route continuity up to 99.85%, and Viterbi accuracy up to 99.32%.*

---

## 6. Model Comparison & Bayesian Hyperparameter Optimization

### A. Phase 1 Model Benchmark (5 Model Families)
| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Log Loss | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | Tree-based | `max_depth=12, min_samples_split=20` | **0.8534** | **0.7904** | **0.8433** | 0.2366 | < 0.001 ms/obs |
| **Random Forest** | Ensemble | `n_estimators=100, max_depth=14` | **0.8551** | **0.7347** | **0.8490** | **0.1466** | 0.001 ms/obs |
| **Multi-Layer Perceptron** | Neural Networks | `hidden=(64, 32), Adam` | 0.8529 | 0.8112 | 0.8508 | 0.1466 | < 0.001 ms/obs |
| **Logistic Regression** | Linear / Probabilistic | `C=1.0, solver='lbfgs'` | 0.7837 | 0.7239 | 0.7954 | 0.2181 | < 0.001 ms/obs |
| **K-Nearest Neighbors** | Instance-based | `k=7, weights='distance'` | 0.8436 | 0.7067 | 0.8201 | 0.7749 | 0.033 ms/obs |

### B. Bayesian Hyperparameter Optimization (Optuna on MLP)
Bayesian optimization was executed across 25 trials using Optuna's Tree-structured Parzen Estimator (TPE) under 3-fold GroupKFold CV:
* **Best Architecture**: `hidden_layer_sizes=(64, 64)`, `activation='tanh'`, `alpha=9.08e-5`, `lr=1.07e-4`, `batch_size=256`.
* **Validation Macro F1**: Surged from **0.8529 to 0.8900 (+3.71% gain)**.
* Tuned model saved at: `artifacts/models/mlp_classifier_tuned.joblib`.

---

## 7. Interactive Maps & Tile Error Fix

All Folium and Leaflet maps in the repository strictly utilize **OpenStreetMap standard tiles** (`https://tile.openstreetmap.org/{z}/{x}/{y}.png`).
* This eliminates CartoDB API deprecation 403 Forbidden / "Access Blocked" errors.
* Interactive HTML maps can be opened directly in any browser:
  * `artifacts/maps/hmm_reconstructed_route_trip_00000000.html`
  * `artifacts/maps/eda_map_trip_00000001.html`

---

## 8. Deliverable 2: Technical Report (PDF)

The consolidated technical report adheres to all requirements in `Deliverables.pdf` and is available at:
* [`docs/PathSnap_Technical_Report.pdf`](docs/PathSnap_Technical_Report.pdf)
* [`report/technical_report.pdf`](report/technical_report.pdf)
