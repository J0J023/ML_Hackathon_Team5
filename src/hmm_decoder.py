import os
import glob
import time
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import networkx as nx
import folium
from folium import plugins
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MAPS_DIR = ARTIFACTS_DIR / "maps"
STUDENT_FOLDER = BASE_DIR / "Student folder"
nested_student = STUDENT_FOLDER / "Student folder"
if nested_student.is_dir():
    STUDENT_FOLDER = nested_student

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "dist_orthogonal", "proj_ratio", "dist_to_start", "dist_to_end",
    "distance_rank", "road_crowding_25_50m", "arc_length",
    "distance_travelled", "heading_diff", "heading_cos", "speed",
    "time_gap", "is_time_gap"
]
TARGET_COL = "is_true_arc"


class HMMRouteDecoder:
    """
    Hidden Markov Model (HMM) with Viterbi Trellis dynamic programming for
    sequential map-matching and continuous vehicle route reconstruction.
    
    Hidden States: Candidate road segments C(p_t) for GPS observation p_t.
    Emission Probability: P(p_t | a_k) ~ P_model(y=1 | x_tk) * exp(-d_ortho^2 / (2*sigma_z^2))
    Transition Probability: P(a_j | a_i) ~ exp(-|d_route - d_gps| / beta)
    Gap Handling: Softens or resets topological penalty when is_time_gap == 1 (Delta t > 3s).
    """
    def __init__(self, beta=15.0, sigma_z=15.0, max_graph_dist=500.0, disconnected_penalty=-25.0):
        self.beta = beta
        self.sigma_z = sigma_z
        self.max_graph_dist = max_graph_dist
        self.disconnected_penalty = disconnected_penalty
        self.dijkstra_cache = {}

    def build_network_graph(self, arcs, nodes):
        """Constructs a metric directed road network graph."""
        nodes_arr = np.array(nodes)
        lat0 = float(np.mean(nodes_arr[:, 1]))
        R = 6371000.0
        x = np.radians(nodes_arr[:, 0]) * R * np.cos(np.radians(lat0))
        y = np.radians(nodes_arr[:, 1]) * R
        pts_m = np.column_stack([x, y])
        
        G = nx.DiGraph()
        arc_lens = {}
        for idx, (u, v) in enumerate(arcs):
            length = float(np.linalg.norm(pts_m[v] - pts_m[u]))
            arc_lens[idx] = max(length, 0.1)
            G.add_edge(u, v, arc_id=idx, weight=arc_lens[idx])
            
        return G, arc_lens, pts_m

    def decode_observations(self, trip_df, model, arcs, nodes, G=None, arc_lens=None):
        """
        Runs Viterbi dynamic programming in log-space across all observations of a trip.
        Returns:
            decoded_arcs: List of chosen arc_id for each observation.
            viterbi_probs: List of emission probabilities.
            latency_ms_per_obs: Average execution latency per observation ping.
        """
        if G is None or arc_lens is None:
            G, arc_lens, _ = self.build_network_graph(arcs, nodes)

        # 1. Compute model emission likelihoods
        X = trip_df[FEATURE_COLS].to_numpy(dtype=float)
        probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
        trip_df = trip_df.copy()
        trip_df["model_prob"] = probs
        
        # 2. Group candidates by observation index
        grouped = [g for _, g in trip_df.groupby("obs_idx")]
        T = len(grouped)
        if T == 0:
            return [], [], 0.0

        t_start = time.time()
        
        # Initialize Trellis (t=0)
        c0 = grouped[0]
        c0_probs = np.maximum(c0["model_prob"].values, 1e-6)
        c0_ortho = c0["dist_orthogonal"].values
        V_prev = np.log(c0_probs) - (c0_ortho**2) / (2 * self.sigma_z**2)
        cands_prev = c0["arc_id"].values
        proj_prev = c0["proj_ratio"].values
        
        backpointers = []
        
        # Forward Trellis recurrence (t = 1 .. T-1)
        for t in range(1, T):
            ct = grouped[t]
            cands_curr = ct["arc_id"].values
            proj_curr = ct["proj_ratio"].values
            ct_probs = np.maximum(ct["model_prob"].values, 1e-6)
            ct_ortho = ct["dist_orthogonal"].values
            emissions = np.log(ct_probs) - (ct_ortho**2) / (2 * self.sigma_z**2)
            
            d_gps = ct["distance_travelled"].values[0] if "distance_travelled" in ct else 10.0
            is_gap = bool(ct["is_time_gap"].values[0]) if "is_time_gap" in ct else False
            
            K_prev = len(cands_prev)
            K_curr = len(cands_curr)
            
            V_curr = np.full(K_curr, -1e9)
            bp_curr = np.zeros(K_curr, dtype=int)
            
            for j in range(K_curr):
                aj = cands_curr[j]
                rj = proj_curr[j]
                uj, vj = arcs[aj]
                lj = arc_lens[aj]
                
                best_val = -1e9
                best_i = 0
                
                for i in range(K_prev):
                    ai = cands_prev[i]
                    ri = proj_prev[i]
                    ui, vi = arcs[ai]
                    li = arc_lens[ai]
                    
                    if is_gap or d_gps > 100.0:
                        # Large gap: soften topological transition penalty
                        trans_cost = 0.0
                    elif ai == aj:
                        # Vehicle continues along the same road arc
                        d_route = abs(rj - ri) * li
                        trans_cost = -abs(d_route - d_gps) / self.beta
                    elif vi == uj:
                        # Direct adjacent topological successor
                        d_route = (1.0 - ri) * li + rj * lj
                        trans_cost = -abs(d_route - d_gps) / self.beta
                    else:
                        # Shortest path through road network
                        cache_key = (vi, uj)
                        if cache_key not in self.dijkstra_cache:
                            try:
                                d_graph = nx.shortest_path_length(G, source=vi, target=uj, weight="weight")
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                d_graph = 1e6
                            self.dijkstra_cache[cache_key] = d_graph
                        else:
                            d_graph = self.dijkstra_cache[cache_key]
                            
                        if d_graph < self.max_graph_dist:
                            d_route = (1.0 - ri) * li + d_graph + rj * lj
                            trans_cost = -abs(d_route - d_gps) / self.beta
                        else:
                            trans_cost = self.disconnected_penalty
                    
                    score = V_prev[i] + trans_cost
                    if score > best_val:
                        best_val = score
                        best_i = i
                        
                V_curr[j] = best_val + emissions[j]
                bp_curr[j] = best_i
                
            backpointers.append(bp_curr)
            V_prev = V_curr
            cands_prev = cands_curr
            proj_prev = proj_curr
            
        # Backtracking MAP path
        opt_idx = int(np.argmax(V_prev))
        opt_indices = [opt_idx]
        for t in range(T - 2, -1, -1):
            opt_idx = int(backpointers[t][opt_idx])
            opt_indices.append(opt_idx)
        opt_indices.reverse()
        
        total_time = time.time() - t_start
        latency_ms_per_obs = (total_time / T) * 1000.0
        
        decoded_arcs = [int(grouped[t]["arc_id"].values[opt_indices[t]]) for t in range(T)]
        viterbi_probs = [float(grouped[t]["model_prob"].values[opt_indices[t]]) for t in range(T)]
        
        return decoded_arcs, viterbi_probs, latency_ms_per_obs

    @staticmethod
    def compute_continuity(arc_sequence, arcs):
        """
        Computes route continuity (% of consecutive steps that are identical or directly connected).
        """
        if len(arc_sequence) < 2:
            return 100.0
        total = 0
        valid = 0
        for i in range(len(arc_sequence) - 1):
            a1, a2 = arc_sequence[i], arc_sequence[i+1]
            total += 1
            if a1 == a2:
                valid += 1
            elif a1 < len(arcs) and a2 < len(arcs) and arcs[a1][1] == arcs[a2][0]:
                valid += 1
        return (valid / total * 100.0) if total > 0 else 0.0

    @staticmethod
    def reconstruct_route_sequence(decoded_arcs):
        """Collapses consecutive identical arcs into a sequential route."""
        route = []
        for a in decoded_arcs:
            if not route or route[-1] != a:
                route.append(a)
        return route


def generate_folium_map(trip_id, raw_track, nodes, arcs, true_route, predicted_arcs, out_html):
    """
    Renders interactive Folium map required by Section 6:
    - Layer 1: Raw GPS observations (blue circle markers)
    - Layer 2: Predicted route sequence (red polyline)
    - Layer 3: Ground-truth reference route (green polyline)
    - Layer 4: Candidate road network context (light gray edges)
    """
    nodes_arr = np.array(nodes)
    raw_track_arr = np.array(raw_track)
    
    center_lat = float(np.mean(raw_track_arr[:, 1]))
    center_lon = float(np.mean(raw_track_arr[:, 0]))
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )
    
    # 1. Road Network Layer (subgraph around trip)
    network_group = folium.FeatureGroup(name="Road Network Graph", show=False)
    # Collect arcs that appear near the trip
    sampled_arcs = list(set(predicted_arcs + (true_route if true_route else [])))
    for a in sampled_arcs[:200]:
        if a < len(arcs):
            u, v = arcs[a]
            p1 = [nodes[u][1], nodes[u][0]]
            p2 = [nodes[v][1], nodes[v][0]]
            folium.PolyLine([p1, p2], color="#999999", weight=2, opacity=0.5).add_to(network_group)
    network_group.add_to(m)
    
    # 2. Raw GPS Layer
    gps_group = folium.FeatureGroup(name="Original GPS Observations", show=True)
    for idx, pt in enumerate(raw_track_arr):
        lat, lon = pt[1], pt[0]
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#2b83ba",
            fill=True,
            fill_color="#2b83ba",
            fill_opacity=0.8,
            popup=f"Obs #{idx}<br>Lon: {lon:.6f}<br>Lat: {lat:.6f}"
        ).add_to(gps_group)
    gps_group.add_to(m)
    
    # 3. Reference Route (Ground Truth)
    if true_route:
        ref_group = folium.FeatureGroup(name="Ground-Truth Route (Green)", show=True)
        ref_coords = []
        for a in true_route:
            if a < len(arcs):
                u, v = arcs[a]
                ref_coords.append([nodes[u][1], nodes[u][0]])
                ref_coords.append([nodes[v][1], nodes[v][0]])
        if ref_coords:
            folium.PolyLine(
                ref_coords,
                color="#2ca25f",
                weight=6,
                opacity=0.7,
                tooltip="Ground-Truth Route"
            ).add_to(ref_group)
        ref_group.add_to(m)
        
    # 4. Predicted Route (HMM Decoded)
    pred_group = folium.FeatureGroup(name="Predicted Route (Red)", show=True)
    pred_coords = []
    unique_pred_arcs = HMMRouteDecoder.reconstruct_route_sequence(predicted_arcs)
    for a in unique_pred_arcs:
        if a < len(arcs):
            u, v = arcs[a]
            pred_coords.append([nodes[u][1], nodes[u][0]])
            pred_coords.append([nodes[v][1], nodes[v][0]])
    if pred_coords:
        folium.PolyLine(
            pred_coords,
            color="#d73027",
            weight=4,
            opacity=0.9,
            dash_array="6, 6",
            tooltip="HMM Decoded Route"
        ).add_to(pred_group)
    pred_group.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))
    print(f"[+] Saved interactive Folium map: {out_html}")


def run_hmm_benchmark():
    print("=" * 85)
    print("PathSnap: HMM and Viterbi Sequential Route Reconstruction Benchmark")
    print("Evaluating Classifier-Only vs. Classifier + HMM across 10 Student-Test Trips")
    print("=" * 85)
    
    test_parquet = ARTIFACTS_DIR / "test_features.parquet"
    test_csv = ARTIFACTS_DIR / "test_features.csv.gz"
    if test_parquet.exists():
        test_df = pd.read_parquet(test_parquet)
    else:
        test_df = pd.read_csv(test_csv)
        
    # Load Decision Tree model
    dt_model_path = MODELS_DIR / "decision_tree.joblib"
    if not dt_model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {dt_model_path}")
    model = joblib.load(dt_model_path)
    print(f"Loaded trained classifier: {dt_model_path.name}")
    
    decoder = HMMRouteDecoder(beta=15.0, sigma_z=15.0, max_graph_dist=500.0)
    
    results = []
    trip_dirs = sorted(glob.glob(str(STUDENT_FOLDER / "student_test" / "*")))
    
    print("\nProcessing trips...")
    for trip_path in trip_dirs:
        tid_str = os.path.basename(trip_path)
        tid_int = int(tid_str)
        
        trip_sub = test_df[test_df["trip_id"] == tid_int].copy()
        if trip_sub.empty:
            continue
            
        arcs_file = os.path.join(trip_path, f"{tid_str}.arcs")
        nodes_file = os.path.join(trip_path, f"{tid_str}.nodes")
        route_file = os.path.join(trip_path, f"{tid_str}.route")
        track_file = os.path.join(trip_path, f"{tid_str}.track")
        
        with open(arcs_file) as f:
            arcs = [list(map(int, l.strip().split())) for l in f if l.strip()]
        with open(nodes_file) as f:
            nodes = [list(map(float, l.strip().split())) for l in f if l.strip()]
        with open(track_file) as f:
            raw_track = [list(map(float, l.strip().split())) for l in f if l.strip()]
            
        true_route = []
        if os.path.exists(route_file):
            with open(route_file) as f:
                true_route = [int(l.strip()) for l in f if l.strip()]
                
        # 1. Classifier-Only Evaluation
        X_trip = trip_sub[FEATURE_COLS].to_numpy(dtype=float)
        raw_probs = model.predict_proba(X_trip)[:, 1]
        trip_sub["prob"] = raw_probs
        raw_max_idx = trip_sub.groupby("obs_idx")["prob"].idxmax()
        raw_top1_df = trip_sub.loc[raw_max_idx].sort_values("obs_idx")
        raw_arcs = raw_top1_df["arc_id"].values
        raw_top1_acc = float(raw_top1_df["is_true_arc"].mean() * 100.0)
        raw_continuity = decoder.compute_continuity(raw_arcs, arcs)
        
        # 2. HMM + Viterbi Evaluation
        G, arc_lens, _ = decoder.build_network_graph(arcs, nodes)
        viterbi_arcs, _, lat_ms = decoder.decode_observations(trip_sub, model, arcs, nodes, G, arc_lens)
        
        # Map decoded arcs back to check if is_true_arc
        vit_acc_list = []
        for t_idx, arc in enumerate(viterbi_arcs):
            match_row = trip_sub[(trip_sub["obs_idx"] == t_idx) & (trip_sub["arc_id"] == arc)]
            if not match_row.empty:
                vit_acc_list.append(int(match_row["is_true_arc"].values[0]))
            else:
                vit_acc_list.append(0)
        vit_top1_acc = float(np.mean(vit_acc_list) * 100.0) if vit_acc_list else 0.0
        vit_continuity = decoder.compute_continuity(viterbi_arcs, arcs)
        
        # Route Sequence comparison vs true route
        vit_route = decoder.reconstruct_route_sequence(viterbi_arcs)
        true_set = set(true_route)
        vit_set = set(vit_route)
        recall = len(true_set.intersection(vit_set)) / len(true_set) * 100.0 if true_set else 0.0
        precision = len(true_set.intersection(vit_set)) / len(vit_set) * 100.0 if vit_set else 0.0
        
        row = {
            "Trip ID": tid_str,
            "Observations": len(raw_arcs),
            "Raw Top-1 Acc": f"{raw_top1_acc:.2f}%",
            "Viterbi Top-1 Acc": f"{vit_top1_acc:.2f}%",
            "Raw Continuity": f"{raw_continuity:.2f}%",
            "Viterbi Continuity": f"{vit_continuity:.2f}%",
            "Route Recall": f"{recall:.2f}%",
            "Route Precision": f"{precision:.2f}%",
            "Latency (ms/obs)": round(lat_ms, 3)
        }
        results.append(row)
        print(f"Trip {tid_str} ({len(raw_arcs):4d} obs): Raw Cont={raw_continuity:5.2f}% -> Vit Cont={vit_continuity:5.2f}% | Top-1={vit_top1_acc:5.2f}% | Latency={lat_ms:.3f} ms")
        
        # Generate representative Folium map for the first trip (00000000)
        if tid_str == "00000000":
            map_path = MAPS_DIR / "hmm_reconstructed_route_trip_00000000.html"
            generate_folium_map(tid_str, raw_track, nodes, arcs, true_route, viterbi_arcs, map_path)

    res_df = pd.DataFrame(results)
    table_csv = ARTIFACTS_DIR / "hmm_comparison_table.csv"
    res_df.to_csv(table_csv, index=False)
    print(f"\n[+] Saved HMM comparison table: {table_csv}")
    
    print("\n" + "=" * 95)
    print("HMM VITERBI ROUTE RECONSTRUCTION BENCHMARK RESULTS (Mandatory Section 5)")
    print("=" * 95)
    print(res_df.to_string(index=False))
    print("=" * 95)
    
    # Generate Comparison Plot
    plot_hmm_comparison(res_df)
    return res_df


def plot_hmm_comparison(res_df):
    """Plots Classifier-Only vs Classifier + HMM continuity and accuracy comparison."""
    plot_data = []
    for _, r in res_df.iterrows():
        tid = r["Trip ID"]
        raw_c = float(r["Raw Continuity"].rstrip("%"))
        vit_c = float(r["Viterbi Continuity"].rstrip("%"))
        plot_data.append({"Trip ID": tid, "Method": "Classifier Only", "Continuity": raw_c})
        plot_data.append({"Trip ID": tid, "Method": "Classifier + HMM", "Continuity": vit_c})
    
    p_df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=p_df, x="Trip ID", y="Continuity", hue="Method", palette=["#d7191c", "#2c7bb6"], ax=ax)
    ax.set_title("Route Continuity Comparison: Classifier-Only vs. Classifier + HMM Viterbi", fontsize=12, fontweight="bold")
    ax.set_xlabel("Student-Test Trip ID", fontsize=11)
    ax.set_ylabel("Route Continuity (%)", fontsize=11)
    ax.set_ylim(70, 102)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "hmm_continuity_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved continuity comparison plot: {plot_path}")


if __name__ == "__main__":
    run_hmm_benchmark()
