"""Run the map-matching API against every labeled evaluator trip."""

import json
from pathlib import Path

import pandas as pd

try:
    from .api import MODEL_NAME, TripRequest, predict_trip
except ImportError:
    from api import MODEL_NAME, TripRequest, predict_trip

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = PROJECT_ROOT / "evaluator_test"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():
    if not EVALUATOR_DIR.is_dir():
        raise FileNotFoundError(f"Evaluator dataset not found: {EVALUATOR_DIR}")

    trip_dirs = sorted(path for path in EVALUATOR_DIR.iterdir() if path.is_dir())
    if not trip_dirs:
        raise FileNotFoundError(f"No evaluator trips found under {EVALUATOR_DIR}")

    rows = []
    for trip_dir in trip_dirs:
        response = predict_trip(TripRequest(trip_id=trip_dir.name))
        metrics = response.metrics
        if metrics.classifier_only_accuracy is None or metrics.hmm_viterbi_accuracy is None:
            raise ValueError(f"Trip {trip_dir.name} has no route ground truth for evaluation.")
        row = {
            "model": MODEL_NAME,
            "trip_id": response.trip_id,
            "observations": len(response.predicted_arc_ids),
            "classifier_only_accuracy": metrics.classifier_only_accuracy,
            "hmm_viterbi_accuracy": metrics.hmm_viterbi_accuracy,
            "route_continuity": metrics.route_continuity,
            "inference_time_ms": metrics.inference_time_ms,
        }
        rows.append(row)
        print(
            f"{row['trip_id']}: classifier={row['classifier_only_accuracy']:.4f}, "
            f"Viterbi={row['hmm_viterbi_accuracy']:.4f}, "
            f"continuity={row['route_continuity']:.4f}, "
            f"time={row['inference_time_ms']:.1f} ms"
        )

    results = pd.DataFrame(rows)
    summary = {
        "model": MODEL_NAME,
        "trip_count": len(results),
        "mean_classifier_only_accuracy": float(results["classifier_only_accuracy"].mean()),
        "mean_hmm_viterbi_accuracy": float(results["hmm_viterbi_accuracy"].mean()),
        "mean_route_continuity": float(results["route_continuity"].mean()),
        "mean_inference_time_ms": float(results["inference_time_ms"].mean()),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS_DIR / "evaluator_test_results.csv"
    json_path = ARTIFACTS_DIR / "evaluator_test_results.json"
    results.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as results_file:
        json.dump({"summary": summary, "trips": rows}, results_file, indent=2)

    print("\nEvaluator summary:")
    print(pd.Series(summary).to_string())
    print(f"\nSaved results:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
