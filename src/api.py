"""FastAPI inference service for distributed PathSnap map-matching and route reconstruction."""

import json
import logging
import math
import os
import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pathsnap.api")

# Rolling live logs buffer for server and client inspection (last 200 events)
SERVER_LOGS = deque(maxlen=200)


def live_log(stage: str, message: str, color_code: str = "36"):
    """Output real-time timestamped ANSI colored logs to terminal and store in buffer."""
    timestamp = time.strftime("%H:%M:%S")
    # ANSI escape sequences: 31=red, 32=green, 33=yellow, 34=blue, 35=magenta, 36=cyan
    formatted_console = f"\033[1;{color_code}m[{timestamp}] [{stage}]\033[0m {message}"
    print(formatted_console)
    sys.stdout.flush()

    SERVER_LOGS.append(
        {
            "timestamp": timestamp,
            "stage": stage,
            "message": message,
            "color": color_code,
        }
    )


try:
    from .preprocessor import (
        ARTIFACTS_DIR,
        BASE_DIR,
        FEATURE_COLS,
        TEST_DIR,
        TRAIN_DIR,
        clean_gps_track,
        extract_trip_features,
        load_trip_data,
        match_track_to_route,
    )
except ImportError:
    from preprocessor import (
        ARTIFACTS_DIR,
        BASE_DIR,
        FEATURE_COLS,
        TEST_DIR,
        TRAIN_DIR,
        clean_gps_track,
        extract_trip_features,
        load_trip_data,
        match_track_to_route,
    )

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
MODEL_NAME = "Multi-Layer Perceptron"
MODEL_PATH = os.path.join(MODELS_DIR, "multi-layer_perceptron.joblib")
TRANSFORM_PATH = os.path.join(MODELS_DIR, "feature_transform.json")
EVALUATOR_DIR = os.path.join(REPO_ROOT, "evaluator_test")
UI_DIR = os.path.join(REPO_ROOT, "ui")

EVALUATOR_TRIP_IDS = [
    "00000015",
    "00000017",
    "00000023",
    "00000030",
    "00000049",
    "00000059",
    "00000067",
    "00000073",
    "00000087",
    "00000094",
]

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed GPS map-matching and sequential HMM route reconstruction service.",
    version="2.0.0",
)

# Enable CORS for cross-laptop communication (Laptop 1 API <-> Laptop 2 UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TripRequest(BaseModel):
    trip_id: str = Field(
        min_length=1,
        max_length=64,
        description="Name or ID of a trip directory (e.g. '00000015' from evaluator_test).",
        examples=["00000015", "00000000"],
    )


class PredictionMetrics(BaseModel):
    classifier_only_accuracy: Optional[float] = None
    hmm_viterbi_accuracy: Optional[float] = None
    route_continuity: Optional[float] = None
    accuracy_improvement: Optional[float] = None
    inference_time_ms: float
    total_observations: int
    total_candidates: int
    partition: str


class TripPredictionResponse(BaseModel):
    trip_id: str
    partition: str
    predicted_arc_ids: list[int]
    predicted_coordinates: list[list[float]]
    raw_gps_coordinates: list[list[float]]
    reference_coordinates: Optional[list[list[float]]] = None
    metrics: PredictionMetrics


@lru_cache(maxsize=1)
def load_inference_assets():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained {MODEL_NAME} not found at {MODEL_PATH}. Run src/train_models.py first."
        )
    if not os.path.isfile(TRANSFORM_PATH):
        raise FileNotFoundError(
            f"Feature transform not found at {TRANSFORM_PATH}. Run src/preprocessor.py first."
        )

    model = joblib.load(MODEL_PATH)
    with open(TRANSFORM_PATH, encoding="utf-8") as transform_file:
        transform = json.load(transform_file)
    if transform.get("feature_columns") != FEATURE_COLS:
        raise ValueError("Saved feature transform does not match the API feature schema.")
    if not hasattr(model, "predict_proba") or 1 not in model.classes_:
        raise ValueError("Saved API classifier must support predict_proba and class 1.")
    if getattr(model, "n_features_in_", None) != len(FEATURE_COLS):
        raise ValueError(
            f"Saved {MODEL_NAME} was trained with a different feature schema. "
            "Run src/train_models.py to retrain it."
        )
    return model, transform


def resolve_trip_dir(trip_id: str) -> tuple[str, str]:
    """Resolve a trip_id to its absolute directory across evaluator_test, student_test, and train.

    Returns:
        tuple[str, str]: (absolute_trip_directory, partition_name)
    """
    clean_id = trip_id.strip()
    candidate_names = [clean_id]

    digits_only = "".join(filter(str.isdigit, clean_id))
    if digits_only:
        num = int(digits_only)
        candidate_names.extend([f"{num:08d}", f"{num}"])

    search_locations = [
        (EVALUATOR_DIR, "evaluator_test"),
        (os.path.join(STUDENT_FOLDER, "student_test"), "student_test"),
        (os.path.join(STUDENT_FOLDER, "train"), "train"),
    ]

    for base_dir, partition in search_locations:
        if not os.path.isdir(base_dir):
            continue
        for name in candidate_names:
            trip_path = os.path.join(base_dir, name)
            if os.path.isdir(trip_path):
                return trip_path, partition

    raise HTTPException(
        status_code=404,
        detail=(
            f"Trip '{trip_id}' was not found in evaluator_test, student_test, or train data. "
            f"Available evaluator trips: {EVALUATOR_TRIP_IDS}"
        ),
    )


def transform_candidate_features(features: pd.DataFrame, transform: dict) -> np.ndarray:
    columns = transform["feature_columns"]
    missing_columns = set(columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"Candidate features are missing columns: {sorted(missing_columns)}")

    clean = features[columns].replace([np.inf, -np.inf], np.nan)
    medians = pd.Series(transform["imputation_medians"], dtype=float)
    values = clean.fillna(medians).to_numpy(dtype=float)
    means = np.array([transform["scaler_mean"][name] for name in columns], dtype=float)
    scales = np.array([transform["scaler_scale"][name] for name in columns], dtype=float)
    return (values - means) / scales


def _project_candidate_points(
    candidates: pd.DataFrame, arcs: np.ndarray, nodes_m: np.ndarray
) -> np.ndarray:
    arc_ids = candidates["arc_id"].to_numpy(dtype=np.int64)
    ratios = candidates["proj_ratio"].to_numpy(dtype=float)
    starts = nodes_m[arcs[arc_ids, 0]]
    vectors = nodes_m[arcs[arc_ids, 1]] - starts
    return starts + ratios[:, None] * vectors


def viterbi_decode(
    candidate_groups: list[pd.DataFrame],
    arcs: np.ndarray,
    nodes_m: np.ndarray,
) -> list[int]:
    """Decode candidate arcs using model probabilities, arc connectivity, and movement distance."""
    if not candidate_groups:
        return []

    projected_groups = [_project_candidate_points(candidate_groups[0], arcs, nodes_m)]
    score = np.log(np.maximum(candidate_groups[0]["probability"].to_numpy(dtype=float), 1e-12))
    backpointers = [None]

    for observation_index in range(1, len(candidate_groups)):
        previous = candidate_groups[observation_index - 1]
        current = candidate_groups[observation_index]

        previous_arc_ids = previous["arc_id"].to_numpy(dtype=np.int64)
        current_arc_ids = current["arc_id"].to_numpy(dtype=np.int64)
        previous_points = projected_groups[-1]
        current_points = _project_candidate_points(current, arcs, nodes_m)
        expected_distance = float(current["distance_travelled"].iloc[0])
        distance_scale = max(15.0, expected_distance * 0.5 + 5.0)

        transition_scores = np.empty((len(previous), len(current)), dtype=float)
        for previous_index, previous_arc in enumerate(previous_arc_ids):
            for current_index, current_arc in enumerate(current_arc_ids):
                if previous_arc == current_arc:
                    topology_score = 0.0
                elif arcs[previous_arc, 1] == arcs[current_arc, 0]:
                    topology_score = -1.25
                else:
                    topology_score = -5.0

                projected_distance = np.linalg.norm(
                    current_points[current_index] - previous_points[previous_index]
                )
                distance_score = -abs(projected_distance - expected_distance) / distance_scale
                transition_scores[previous_index, current_index] = topology_score + distance_score

        candidate_scores = score[:, None] + transition_scores
        best_previous = np.argmax(candidate_scores, axis=0)
        emission_scores = np.log(
            np.maximum(current["probability"].to_numpy(dtype=float), 1e-12)
        )
        score = candidate_scores[best_previous, np.arange(len(current))] + emission_scores
        backpointers.append(best_previous)
        projected_groups.append(current_points)

    state = int(np.argmax(score))
    selected_states = [state]
    for observation_index in range(len(candidate_groups) - 1, 0, -1):
        state = int(backpointers[observation_index][state])
        selected_states.append(state)
    selected_states.reverse()

    return [
        int(group.iloc[state]["arc_id"])
        for group, state in zip(candidate_groups, selected_states)
    ]


def project_predicted_coordinates(
    predicted_arc_ids: list[int],
    candidate_groups: list[pd.DataFrame],
    arcs: np.ndarray,
    nodes: np.ndarray,
    track: np.ndarray,
) -> list[list[float]]:
    lat_mean = float(np.mean(track[:, 1]))
    kx = 111320.0 * math.cos(math.radians(lat_mean))
    ky = 110540.0
    if abs(kx) < 1e-8:
        raise HTTPException(status_code=422, detail="Trip latitude is too close to a pole.")

    nodes_m = np.empty((len(nodes), 2), dtype=float)
    nodes_m[:, 0] = nodes[:, 0] * kx
    nodes_m[:, 1] = nodes[:, 1] * ky
    output = []
    for observation_index, arc_id in enumerate(predicted_arc_ids):
        group = candidate_groups[observation_index]
        matching = group.loc[group["arc_id"] == arc_id]
        if not matching.empty:
            ratio = float(matching.iloc[0]["proj_ratio"])
        else:
            ratio = 0.5

        start = nodes_m[arcs[arc_id, 0]]
        end = nodes_m[arcs[arc_id, 1]]
        projected = start + ratio * (end - start)
        output.append([float(projected[1] / ky), float(projected[0] / kx)])
    return output


def extract_reference_coordinates(
    route: list[int],
    arcs: np.ndarray,
    nodes: np.ndarray,
) -> list[list[float]]:
    """Extract reference polyline coordinates [lat, lon] from ground truth route arcs."""
    if not route:
        return []
    coords = []
    for arc_id in route:
        u_idx = arcs[arc_id, 0]
        v_idx = arcs[arc_id, 1]
        u_lat, u_lon = float(nodes[u_idx, 1]), float(nodes[u_idx, 0])
        v_lat, v_lon = float(nodes[v_idx, 1]), float(nodes[v_idx, 0])
        if not coords or coords[-1] != [u_lat, u_lon]:
            coords.append([u_lat, u_lon])
        coords.append([v_lat, v_lon])
    return coords


def route_continuity(predicted_arc_ids: list[int], arcs: np.ndarray) -> Optional[float]:
    if len(predicted_arc_ids) < 2:
        return None
    connected = sum(
        first == second or arcs[first, 1] == arcs[second, 0]
        for first, second in zip(predicted_arc_ids, predicted_arc_ids[1:])
    )
    return float(connected / (len(predicted_arc_ids) - 1))


@app.get("/health")
def health(req: Request):
    client_ip = req.client.host if req.client else "unknown"
    live_log("HEALTH PING", f"Ping from {client_ip} -> Server status: ONLINE (Random Forest Model Ready)", "32")
    try:
        load_inference_assets()
    except (FileNotFoundError, ValueError) as exc:
        live_log("HEALTH ERROR", str(exc), "31")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "service": "PathSnap Map-Matching API",
        "model": "Random Forest Classifier (Ensemble)",
        "evaluator_trips_available": len(EVALUATOR_TRIP_IDS),
        "evaluator_trip_ids": EVALUATOR_TRIP_IDS,
    }


@app.get("/api/logs")
def get_live_logs():
    """Return rolling log entries for the live UI terminal stream."""
    return {"logs": list(SERVER_LOGS)}


@app.get("/api/trips")
def list_available_trips(req: Request):
    client_ip = req.client.host if req.client else "unknown"
    live_log("TRIP QUERY", f"Available trips list fetched by client {client_ip}", "36")
    trips_metadata = []
    for tid in EVALUATOR_TRIP_IDS:
        trip_dir = os.path.join(EVALUATOR_DIR, tid)
        has_route = os.path.isfile(os.path.join(trip_dir, f"{tid}.route"))
        trips_metadata.append(
            {
                "trip_id": tid,
                "partition": "evaluator_test",
                "is_evaluator_trip": True,
                "has_ground_truth": has_route,
            }
        )

    return {
        "evaluator_trips": EVALUATOR_TRIP_IDS,
        "trips": trips_metadata,
        "default_trip": "00000015",
    }


@app.post("/predict", response_model=TripPredictionResponse)
def predict_trip(request: TripRequest, req: Request):
    started_at = time.perf_counter()
    client_ip = req.client.host if req.client else "unknown"

    trip_dir, partition = resolve_trip_dir(request.trip_id)

    # Real-time incremental terminal & UI logging
    print("\n" + "=" * 80)
    live_log("INCOMING REQUEST", f"POST /predict from {client_ip} | Trip ID: '{request.trip_id}' (Resolved: {partition})", "35")

    try:
        model, transform = load_inference_assets()
    except (FileNotFoundError, ValueError) as exc:
        live_log("ASSET ERROR", f"Model/Transform load failed: {exc}", "31")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        live_log("STAGE 1/4", f"Loading road network and GPS trajectory from: {trip_dir}", "34")
        trip_id, nodes, arcs, raw_track, route = load_trip_data(trip_dir)
        track, _, _ = clean_gps_track(raw_track)
        live_log("STAGE 1/4", f"Loaded {len(nodes):,} nodes, {len(arcs):,} road arcs, {len(track):,} valid GPS observations", "34")

        live_log("STAGE 2/4", "Generating candidate road segments via KD-Tree spatial indexing (R = 60m)...", "36")
        cand_t0 = time.perf_counter()
        features, miss_rate = extract_trip_features(trip_dir, include_true_arc=False)
        cand_dt = (time.perf_counter() - cand_t0) * 1000.0
        live_log("STAGE 2/4", f"Generated {len(features):,} candidates in {cand_dt:.1f} ms (Candidate Miss Rate: {miss_rate:.2%})", "36")
    except OSError as exc:
        live_log("DATA ERROR", f"Trip data loading failed: {exc}", "31")
        raise HTTPException(status_code=422, detail=f"Could not load trip data: {exc}") from exc

    if features.empty or len(track) == 0:
        live_log("DATA ERROR", f"Trip '{trip_id}' has no usable GPS candidates.", "31")
        raise HTTPException(
            status_code=422, detail=f"Trip '{trip_id}' has no usable GPS candidates."
        )

    total_candidates = len(features)
    live_log("STAGE 3/4", "Transforming 13 invariant features and scoring with Random Forest...", "33")
    score_t0 = time.perf_counter()
    transformed = transform_candidate_features(features, transform)
    class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(transformed)[:, class_index]
    features = features.copy()
    features["probability"] = probabilities
    features = features.sort_values(["obs_idx", "dist_orthogonal"], kind="stable").copy()

    candidate_groups = [
        group.reset_index(drop=True)
        for _, group in features.groupby("obs_idx", sort=True)
    ]
    classifier_arc_ids = [
        int(group.iloc[int(np.argmax(group["probability"].to_numpy()))]["arc_id"])
        for group in candidate_groups
    ]
    score_dt = (time.perf_counter() - score_t0) * 1000.0
    live_log("STAGE 3/4", f"Model probability scoring completed in {score_dt:.1f} ms", "33")

    live_log("STAGE 4/4", "Executing HMM Viterbi trellis dynamic programming in log-space...", "32")
    lat_mean = float(np.mean(track[:, 1]))
    kx = 111320.0 * math.cos(math.radians(lat_mean))
    ky = 110540.0
    track_m = np.empty((len(track), 2), dtype=float)
    track_m[:, 0] = track[:, 0] * kx
    track_m[:, 1] = track[:, 1] * ky
    nodes_m = np.empty((len(nodes), 2), dtype=float)
    nodes_m[:, 0] = nodes[:, 0] * kx
    nodes_m[:, 1] = nodes[:, 1] * ky

    viterbi_t0 = time.perf_counter()
    predicted_arc_ids = viterbi_decode(candidate_groups, arcs, nodes_m)
    viterbi_ms = (time.perf_counter() - viterbi_t0) * 1000.0
    continuity = route_continuity(predicted_arc_ids, arcs)
    live_log("STAGE 4/4", f"Viterbi decoded {len(predicted_arc_ids):,} arcs in {viterbi_ms:.1f} ms | Route Continuity: {continuity * 100:.2f}%", "32")

    predicted_coordinates = project_predicted_coordinates(
        predicted_arc_ids, candidate_groups, arcs, nodes, track
    )

    classifier_accuracy = None
    viterbi_accuracy = None
    improvement = None
    reference_coords = None

    if route:
        u_m = nodes_m[arcs[:, 0]]
        v_m = nodes_m[arcs[:, 1]]
        true_arc_ids = match_track_to_route(track_m, u_m, v_m, route)
        classifier_accuracy = float(np.mean(np.asarray(classifier_arc_ids) == true_arc_ids))
        viterbi_accuracy = float(np.mean(np.asarray(predicted_arc_ids) == true_arc_ids))
        improvement = float(viterbi_accuracy - classifier_accuracy)
        reference_coords = extract_reference_coordinates(route, arcs, nodes)
        live_log("EVALUATION", f"Ground Truth Match: Classifier {classifier_accuracy*100:.2f}% -> Viterbi {viterbi_accuracy*100:.2f}% (Gain: {improvement*100:+.2f}%)", "32")

    total_time = round((time.perf_counter() - started_at) * 1000.0, 3)
    live_log("DISPATCH 200", f"Dispatched {len(predicted_coordinates):,} points to {client_ip} in {total_time:.1f} ms total", "36")
    print("=" * 80 + "\n")

    metrics = PredictionMetrics(
        classifier_only_accuracy=classifier_accuracy,
        hmm_viterbi_accuracy=viterbi_accuracy,
        route_continuity=continuity,
        accuracy_improvement=improvement,
        inference_time_ms=total_time,
        total_observations=len(track),
        total_candidates=total_candidates,
        partition=partition,
    )

    return TripPredictionResponse(
        trip_id=trip_id,
        partition=partition,
        predicted_arc_ids=predicted_arc_ids,
        predicted_coordinates=predicted_coordinates,
        raw_gps_coordinates=[
            [float(point[1]), float(point[0])] for point in track
        ],
        reference_coordinates=reference_coords,
        metrics=metrics,
    )


# Serve Static UI Frontend
if os.path.isdir(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/")
    @app.get("/ui")
    def serve_ui(req: Request):
        client_ip = req.client.host if req.client else "unknown"
        live_log("UI CONNECT", f"Evaluator UI dashboard accessed from {client_ip}", "35")
        index_file = os.path.join(UI_DIR, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
        return {"message": "PathSnap API is online. UI directory found but index.html missing."}
