# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — Department of Computer Science and Engineering**  
**UE24CS352A: Machine Learning**  
**Hackathon Team Repository**

---

## 1. Challenge Overview
GPS tracking systems inherently suffer from urban canyon multipath reflections, signal dropout, and low sampling frequencies (~1 Hz). As a result, noisy observations deviate onto parallel streets or jump across obstacles.

**PathSnap** is an end-to-end, distributed machine learning system that accurately matches noisy raw GPS trajectories to a road network graph and reconstructs valid, topologically continuous vehicle trajectories.

---

## 2. System Architecture & Work Process

```
[Raw GPS Track (.track)] 
           │
           ▼
[1. Candidate Generation] (Spatial multi-point KD-Tree indexing within R = 60m)
           │
           ▼
[2. Invariant Feature Extraction] (Orthogonal distance, projection ratio, heading error, speed, kinematics)
           │
           ▼
[3. Tree-Based ML Scoring Model] (Decision Tree Classifier)
           │
           ▼ (Candidate emission probabilities)
[4. HMM & Viterbi Decoder] (Topological transition probabilities via network shortest path vs Euclidean distance)
           │
           ▼
[Reconstructed Route & Continuous Trajectory]
           │
           ▼
[FastAPI Backend (Laptop 1) <---> Evaluator UI (Laptop 2)]
```

---

## 3. Repository Structure
```
ML_Hackathon/
├── artifacts/
│   ├── dataset_inventory.csv          # Full audit of all 90 trip partitions
│   ├── eda_summary.json               # Key dataset statistics and sampling properties
│   ├── model_comparison_table.csv     # Phase 1 model comparison table
│   ├── model_comparison_results.json  # Raw metric logs
│   ├── maps/
│   │   └── eda_map_trip_00000001.html # Interactive Folium ground-truth & GPS map
│   ├── models/                        # Serialized model weights & feature scalers
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib
│   │   └── scaler.joblib
│   └── plots/                         # Diagnostic and EDA charts
│       ├── trajectory_distributions.png
│       ├── road_network_complexity.png
│       ├── sampling_interval_dist.png
│       ├── model_performance_comparison.png
│       ├── model_confusion_matrices.png
│       └── tree_feature_importances.png
├── src/
│   ├── eda.py                         # Exploratory Data Analysis & visualizer
│   ├── preprocessor.py                # Candidate generator & feature engineering
│   └── train_models.py                # GroupKFold cross-validation & model benchmarks
├── notebooks/
│   └── decision_tree_pipeline.ipynb   # Complete Decision Tree training & evaluation walkthrough
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Phase 1 Evaluation Results (Tree-Based Focus)

### A. Candidate Generation & Recall
* **Average Candidate Miss Rate**: $5.50\%$ (Candidate Recall: **$94.50\%$**).
* **Average Candidates per Observation**: $\approx 10\text{–}12$ road segment candidates.

### B. Tree-Based Model Comparison Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Log Loss | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | Tree-based | `max_depth=12, min_samples_split=20` | **0.8404** | **0.7094** | **0.8070** | 0.4633 | < 0.001 ms/obs |
| **Random Forest** | Ensemble | `n_estimators=100, max_depth=14` | **0.8441** | **0.7113** | **0.8202** | 0.1649 | 0.001 ms/obs |

### C. Feature Importance Analysis
From Random Forest and Decision Tree feature attributions:
1. `dist_orthogonal` ($33.9\%$): Perpendicular distance from ping to candidate segment line.
2. `proj_ratio` ($22.7\%$): Projection parameter along segment $[0, 1]$.
3. `heading_cos` & `heading_diff` ($19.0\%$ combined): Alignment between trajectory movement vector and road direction.
4. `arc_length` ($9.2\%$): Physical length of the segment.

---

## 5. Setup & Execution Instructions

### Installation
```bash
# Clone the repository
git clone https://github.com/J0J023/ML_Hackathon.git
cd ML_Hackathon

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Phase 1 (EDA & Model Training)
```bash
# 1. Run Exploratory Data Analysis & generate Folium map
python src/eda.py

# 2. Extract candidates, compute features, impute missing values, and scale using training trips
python src/preprocessor.py

# 3. Train models with GroupKFold cross-validation & benchmark using the exported shared matrices
python src/train_models.py
```

The preprocessor exports `artifacts/train_features.parquet` and
`artifacts/test_features.parquet` (or compressed CSV fallbacks). Feature
imputation and scaling are fitted on the training trips only; all models consume
the same exported feature columns. The fitted medians and scaling parameters are
saved in `artifacts/models/feature_transform.json`.
The benchmark trains Logistic Regression, Gaussian Naive Bayes, K-Nearest
Neighbors, Decision Tree, and Random Forest models.
