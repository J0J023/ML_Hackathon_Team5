"""FastAPI inference service for stored PathSnap trips."""

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .preprocessor import (
        ARTIFACTS_DIR,
        FEATURE_COLS,
        STUDENT_FOLDER,
        clean_gps_track,
        extract_trip_features,
        load_trip_data,
        match_track_to_route,
    )
except ImportError:
    from preprocessor import (
        ARTIFACTS_DIR,
        FEATURE_COLS,
        STUDENT_FOLDER,
        clean_gps_track,
        extract_trip_features,
        load_trip_data,
        match_track_to_route,
    )

MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
TRANSFORM_PATH = os.path.join(MODELS_DIR, "feature_transform.json")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Scores road candidates and decodes a trip's predicted road sequence.",
    version="1.0.0",
)


class TripRequest(BaseModel):
    trip_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Name of a trip directory in the local train or student_test dataset.",
        examples=["00000000"],
    )


class PredictionMetrics(BaseModel):
    classifier_only_accuracy: Optional[float] = None
    hmm_viterbi_accuracy: Optional[float] = None
    route_continuity: Optional[float] = None
    inference_time_ms: float


class TripPredictionResponse(BaseModel):
    trip_id: str
    predicted_arc_ids: list[int]
    predicted_coordinates: list[list[float]]
    raw_gps_coordinates: list[list[float]]
    metrics: PredictionMetrics


@lru_cache(maxsize=1)
def load_inference_assets():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained Random Forest not found at {MODEL_PATH}. Run src/train_models.py first."
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
            "Saved Random Forest was trained with a different feature schema. "
            "Run src/train_models.py to retrain it."
        )
    return model, transform


def resolve_trip_dir(trip_id: str) -> str:
    for partition in ("train", "student_test"):
        trip_dir = os.path.join(STUDENT_FOLDER, partition, trip_id)
        if os.path.isdir(trip_dir):
            return trip_dir
    raise HTTPException(
        status_code=404,
        detail=f"Trip '{trip_id}' was not found in the local train or student_test data.",
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


def viterbi_decode(
    candidate_groups: list[pd.DataFrame],
    arcs: np.ndarray,
    nodes_m: np.ndarray,
) -> list[int]:
    """Decode candidate arcs using emissions, arc connectivity, and movement distance."""
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


def _project_candidate_points(
    candidates: pd.DataFrame, arcs: np.ndarray, nodes_m: np.ndarray
) -> np.ndarray:
    arc_ids = candidates["arc_id"].to_numpy(dtype=np.int64)
    ratios = candidates["proj_ratio"].to_numpy(dtype=float)
    starts = nodes_m[arcs[arc_ids, 0]]
    vectors = nodes_m[arcs[arc_ids, 1]] - starts
    return starts + ratios[:, None] * vectors


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
        matching = candidate_groups[observation_index].loc[
            candidate_groups[observation_index]["arc_id"] == arc_id
        ]
        if matching.empty:
            raise RuntimeError(
                f"Decoded arc {arc_id} is not a candidate for observation {observation_index}."
            )
        ratio = float(matching.iloc[0]["proj_ratio"])
        start = nodes_m[arcs[arc_id, 0]]
        end = nodes_m[arcs[arc_id, 1]]
        projected = start + ratio * (end - start)
        output.append([float(projected[1] / ky), float(projected[0] / kx)])
    return output


def route_continuity(predicted_arc_ids: list[int], arcs: np.ndarray) -> Optional[float]:
    if len(predicted_arc_ids) < 2:
        return None
    connected = sum(
        first == second or arcs[first, 1] == arcs[second, 0]
        for first, second in zip(predicted_arc_ids, predicted_arc_ids[1:])
    )
    return float(connected / (len(predicted_arc_ids) - 1))


@app.get("/health")
def health():
    try:
        load_inference_assets()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "model": "random_forest"}


@app.post("/predict", response_model=TripPredictionResponse)
def predict_trip(request: TripRequest):
    started_at = time.perf_counter()
    trip_dir = resolve_trip_dir(request.trip_id)

    try:
        model, transform = load_inference_assets()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        trip_id, nodes, arcs, raw_track, route = load_trip_data(trip_dir)
        track, _, _ = clean_gps_track(raw_track)
        features, _ = extract_trip_features(trip_dir, include_true_arc=False)
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Could not load trip data: {exc}") from exc

    if features.empty or len(track) == 0:
        raise HTTPException(status_code=422, detail=f"Trip '{trip_id}' has no usable GPS candidates.")

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

    lat_mean = float(np.mean(track[:, 1]))
    kx = 111320.0 * math.cos(math.radians(lat_mean))
    ky = 110540.0
    track_m = np.empty((len(track), 2), dtype=float)
    track_m[:, 0] = track[:, 0] * kx
    track_m[:, 1] = track[:, 1] * ky
    nodes_m = np.empty((len(nodes), 2), dtype=float)
    nodes_m[:, 0] = nodes[:, 0] * kx
    nodes_m[:, 1] = nodes[:, 1] * ky
    predicted_arc_ids = viterbi_decode(candidate_groups, arcs, nodes_m)
    predicted_coordinates = project_predicted_coordinates(
        predicted_arc_ids, candidate_groups, arcs, nodes, track
    )

    classifier_accuracy = None
    viterbi_accuracy = None
    if route:
        u_m = nodes_m[arcs[:, 0]]
        v_m = nodes_m[arcs[:, 1]]
        true_arc_ids = match_track_to_route(track_m, u_m, v_m, route)
        classifier_accuracy = float(np.mean(np.asarray(classifier_arc_ids) == true_arc_ids))
        viterbi_accuracy = float(np.mean(np.asarray(predicted_arc_ids) == true_arc_ids))

    metrics = PredictionMetrics(
        classifier_only_accuracy=classifier_accuracy,
        hmm_viterbi_accuracy=viterbi_accuracy,
        route_continuity=route_continuity(predicted_arc_ids, arcs),
        inference_time_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
    )
    return TripPredictionResponse(
        trip_id=trip_id,
        predicted_arc_ids=predicted_arc_ids,
        predicted_coordinates=predicted_coordinates,
        raw_gps_coordinates=[
            [float(point[1]), float(point[0])] for point in track
        ],
        metrics=metrics,
    )
