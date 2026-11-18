"""
PathSnap: Preprocessor and Feature Engineering Pipeline
Author: Team PathSnap
PES University - UE24CS352A Machine Learning Hackathon
"""

import os
import glob
import math
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_FOLDER = os.path.join(BASE_DIR, "Student folder")
TRAIN_DIR = os.path.join(STUDENT_FOLDER, "train")
TEST_DIR = os.path.join(STUDENT_FOLDER, "student_test")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_trip_data(trip_dir):
    """Loads nodes, arcs, track, and route (if exists) for a trip."""
    tid = os.path.basename(trip_dir)
    nodes_path = os.path.join(trip_dir, f"{tid}.nodes")
    arcs_path = os.path.join(trip_dir, f"{tid}.arcs")
    track_path = os.path.join(trip_dir, f"{tid}.track")
    route_path = os.path.join(trip_dir, f"{tid}.route")
    
    with open(nodes_path) as f:
        nodes = np.array([[float(p) for p in l.strip().split()] for l in f if l.strip()])
        
    with open(arcs_path) as f:
        arcs = np.array([[int(p) for p in l.strip().split()] for l in f if l.strip()])
        
    with open(track_path) as f:
        track = np.array([[float(p) for p in l.strip().split()] for l in f if l.strip()])
        
    route = None
    if os.path.exists(route_path):
        with open(route_path) as f:
            route = [int(l.strip()) for l in f if l.strip()]
            
    return tid, nodes, arcs, track, route


def match_track_to_route(track_m, u_m, v_m, route):
    """
    Matches each GPS point to its active ground-truth arc in the route sequence,
    preserving forward temporal monotonicity along the vehicle's trajectory.
    """
    n_pts = len(track_m)
    n_route = len(route)
    true_arcs = np.empty(n_pts, dtype=np.int32)
    
    r_idx = 0
    for i in range(n_pts):
        p = track_m[i]
        # Look within a local forward window around the current route position
        window = range(max(0, r_idx - 2), min(n_route, r_idx + 15))
        best_d = 1e9
        best_r = r_idx
        best_arc = route[r_idx]
        
        for r in window:
            a = route[r]
            u = u_m[a]
            v = v_m[a]
            seg = v - u
            seg_len_sq = np.dot(seg, seg)
            if seg_len_sq == 0:
                d = np.linalg.norm(p - u)
            else:
                t = np.clip(np.dot(p - u, seg) / seg_len_sq, 0.0, 1.0)
                proj = u + t * seg
                d = np.linalg.norm(p - proj)
                
            if d < best_d:
                best_d = d
                best_r = r
                best_arc = a
                
        r_idx = best_r
        true_arcs[i] = best_arc
        
    return true_arcs


def clean_gps_track(track, gap_threshold=3.0):
    """
    Cleans raw GPS track:
    1. Removes broken/invalid coordinates (NaNs, outside valid lat/lon ranges).
    2. Resolves out-of-order timestamps by sorting, and removes duplicate timestamps.
    3. Flags large time gaps where GPS dropped out (dt > gap_threshold).
    """
    if len(track) == 0:
        return track, np.array([]), np.array([])
        
    # 1. Filter valid coordinates
    valid_mask = (
        ~np.isnan(track[:, 0]) & ~np.isnan(track[:, 1]) & ~np.isnan(track[:, 2]) &
        (track[:, 0] >= -180.0) & (track[:, 0] <= 180.0) &
        (track[:, 1] >= -90.0) & (track[:, 1] <= 90.0)
    )
    cleaned = track[valid_mask]
    
    # 2. Sort chronologically and drop duplicate timestamps
    sort_idx = np.argsort(cleaned[:, 2])
    cleaned = cleaned[sort_idx]
    _, unique_idx = np.unique(cleaned[:, 2], return_index=True)
    cleaned = cleaned[np.sort(unique_idx)]
    
    # 3. Compute time delta and flag dropout gaps
    n_pts = len(cleaned)
    dt = np.ones(n_pts)
    if n_pts > 1:
        dt[1:] = np.diff(cleaned[:, 2])
    dt = np.where(dt <= 0, 1.0, dt)
    is_time_gap = (dt > gap_threshold).astype(np.int32)
    
    return cleaned, dt, is_time_gap


def extract_trip_features(trip_dir, search_radius=60.0, max_cands_per_pt=15):
    """
    Extracts candidate road segments and rotation/scale-invariant features
    for every GPS observation in a trip after cleaning and metric projection.
    """
    tid, nodes, arcs, raw_track, route = load_trip_data(trip_dir)
    
    # Step A: Clean GPS points (filter invalid, sort/dedup timestamps, flag gaps)
    track, dt, is_time_gap = clean_gps_track(raw_track, gap_threshold=3.0)
    n_pts = len(track)
    if n_pts == 0:
        return pd.DataFrame(), 0.0
    
    # Step B: Convert degrees to metric coordinates (meters)
    lat_mean = np.mean(track[:, 1])
    kx = 111320.0 * math.cos(math.radians(lat_mean))
    ky = 110540.0
    
    nodes_m = np.empty_like(nodes)
    nodes_m[:, 0] = nodes[:, 0] * kx
    nodes_m[:, 1] = nodes[:, 1] * ky
    
    track_m = np.empty((n_pts, 2))
    track_m[:, 0] = track[:, 0] * kx
    track_m[:, 1] = track[:, 1] * ky
    
    u_m = nodes_m[arcs[:, 0]]
    v_m = nodes_m[arcs[:, 1]]
    arc_vec = v_m - u_m
    arc_len = np.linalg.norm(arc_vec, axis=1)
    
    # 2. Ground truth matching if route exists
    true_arcs = match_track_to_route(track_m, u_m, v_m, route) if route is not None else None
    
    # 3. Fast multi-point segment indexing for high recall
    pts_list = []
    arc_indices = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pts_list.append(u_m + frac * arc_vec)
        arc_indices.append(np.arange(len(arcs)))
    sample_pts = np.vstack(pts_list)
    sample_arc_ids = np.concatenate(arc_indices)
    
    tree = cKDTree(sample_pts)
    query_res = tree.query_ball_point(track_m, r=search_radius)
    
    # 4. Trajectory kinematics (speed, time_gap, heading)
    d_pos = np.zeros((n_pts, 2))
    d_pos[1:] = np.diff(track_m, axis=0)
    speed = np.linalg.norm(d_pos, axis=1) / dt
    
    # Trajectory heading in radians [-pi, pi]
    traj_heading = np.arctan2(d_pos[:, 1], d_pos[:, 0])
    if n_pts > 1:
        traj_heading[0] = traj_heading[1]
    # Arc heading in radians
    arc_heading = np.arctan2(arc_vec[:, 1], arc_vec[:, 0])
    
    # 5. Extract candidate rows
    rows = []
    miss_count = 0
    
    for i in range(n_pts):
        p = track_m[i]
        cands = set(sample_arc_ids[query_res[i]])
        
        # Fallback to nearest arcs if query returned very few
        if len(cands) < 3:
            _, near_idx = tree.query(p, k=min(5, len(sample_pts)))
            cands.update(sample_arc_ids[near_idx] if isinstance(near_idx, (list, np.ndarray)) else [sample_arc_ids[near_idx]])
            
        cands = list(cands)
        
        # Check recall
        is_hit = False
        if true_arcs is not None:
            t_arc = true_arcs[i]
            if t_arc in cands:
                is_hit = True
            else:
                miss_count += 1
                # To train effectively without degenerate sample, add true arc if missed
                cands.append(t_arc)
                
        # Compute exact orthogonal distance and features for all candidates
        cand_arr = np.array(cands, dtype=np.int32)
        u_cand = u_m[cand_arr]
        v_cand = v_m[cand_arr]
        vec_cand = arc_vec[cand_arr]
        len_cand = arc_len[cand_arr]
        len_sq_cand = len_cand ** 2
        
        p_u = p - u_cand
        dot_p_u = np.sum(p_u * vec_cand, axis=1)
        
        # Projection ratio t in [0, 1]
        t_proj = np.where(len_sq_cand > 0, np.clip(dot_p_u / np.maximum(len_sq_cand, 1e-6), 0.0, 1.0), 0.0)
        proj_pts = u_cand + t_proj[:, None] * vec_cand
        dist_ortho = np.linalg.norm(p - proj_pts, axis=1)
        dist_start = np.linalg.norm(p_u, axis=1)
        dist_end = np.linalg.norm(p - v_cand, axis=1)
        
        # Angular difference between trajectory and candidate arc
        theta_diff = np.abs(traj_heading[i] - arc_heading[cand_arr])
        theta_diff = np.where(theta_diff > np.pi, 2 * np.pi - theta_diff, theta_diff)
        cos_theta = np.cos(theta_diff)
        
        # Subsample negative candidates to cap maximum per point and keep balanced
        # Sort by distance
        sort_order = np.argsort(dist_ortho)
        if len(sort_order) > max_cands_per_pt:
            # Keep top candidates + true arc if present
            keep_indices = list(sort_order[:max_cands_per_pt])
            if true_arcs is not None:
                t_pos = np.where(cand_arr == true_arcs[i])[0]
                if len(t_pos) > 0 and t_pos[0] not in keep_indices:
                    keep_indices[-1] = t_pos[0]
            sort_order = np.array(keep_indices)
            
        for idx in sort_order:
            a_id = cand_arr[idx]
            label = 1 if (true_arcs is not None and a_id == true_arcs[i]) else 0
            
            rows.append({
                "trip_id": tid,
                "obs_idx": i,
                "arc_id": a_id,
                "dist_orthogonal": float(dist_ortho[idx]),
                "proj_ratio": float(t_proj[idx]),
                "dist_to_start": float(dist_start[idx]),
                "dist_to_end": float(dist_end[idx]),
                "arc_length": float(len_cand[idx]),
                "heading_diff": float(theta_diff[idx]),
                "heading_cos": float(cos_theta[idx]),
                "speed": float(speed[i]),
                "time_gap": float(dt[i]),
                "is_time_gap": int(is_time_gap[i]),
                "is_true_arc": label
            })
            
    df = pd.DataFrame(rows)
    miss_rate = (miss_count / n_pts) if n_pts > 0 else 0.0
    return df, miss_rate


def build_preprocessed_dataset(max_train_trips=80, max_test_trips=10):
    """
    Builds the complete preprocessed dataset for all train and test trips,
    saving the engineered feature matrices for model experimentation.
    """
    print("=" * 60)
    print("PathSnap: Building Feature-Engineered Dataset...")
    print("=" * 60)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    train_trips = sorted(glob.glob(os.path.join(TRAIN_DIR, "*")))[:max_train_trips]
    test_trips = sorted(glob.glob(os.path.join(TEST_DIR, "*")))[:max_test_trips]
    
    print(f"Processing {len(train_trips)} Training trips...")
    train_dfs = []
    train_misses = []
    t0 = time.time()
    
    for i, tdir in enumerate(train_trips):
        df, m_rate = extract_trip_features(tdir)
        train_dfs.append(df)
        train_misses.append(m_rate)
        if (i + 1) % 10 == 0 or (i + 1) == len(train_trips):
            print(f"  Processed {i + 1}/{len(train_trips)} train trips (Elapsed: {time.time() - t0:.1f}s)")
            
    train_data = pd.concat(train_dfs, ignore_index=True)
    avg_train_miss = np.mean(train_misses) * 100
    print(f"\n[+] Train feature matrix built: {train_data.shape[0]:,} candidate rows")
    print(f"[+] Average Train Candidate Miss Rate: {avg_train_miss:.2f}% (Recall: {100 - avg_train_miss:.2f}%)")
    
    print(f"\nProcessing {len(test_trips)} Student-Test trips...")
    test_dfs = []
    test_misses = []
    for i, tdir in enumerate(test_trips):
        df, m_rate = extract_trip_features(tdir)
        test_dfs.append(df)
        test_misses.append(m_rate)
        
    test_data = pd.concat(test_dfs, ignore_index=True)
    avg_test_miss = np.mean(test_misses) * 100
    print(f"[+] Test feature matrix built: {test_data.shape[0]:,} candidate rows")
    print(f"[+] Average Test Candidate Miss Rate: {avg_test_miss:.2f}% (Recall: {100 - avg_test_miss:.2f}%)")
    
    # Save feature matrices
    try:
        train_out = os.path.join(ARTIFACTS_DIR, "train_features.parquet")
        test_out = os.path.join(ARTIFACTS_DIR, "test_features.parquet")
        train_data.to_parquet(train_out, index=False)
        test_data.to_parquet(test_out, index=False)
        print(f"[+] Saved {train_out}")
        print(f"[+] Saved {test_out}")
    except Exception as e:
        print(f"[*] Parquet export skipped ({e}), falling back to compressed CSV...")
        train_out = os.path.join(ARTIFACTS_DIR, "train_features.csv.gz")
        test_out = os.path.join(ARTIFACTS_DIR, "test_features.csv.gz")
        train_data.to_csv(train_out, index=False, compression="gzip")
        test_data.to_csv(test_out, index=False, compression="gzip")
        print(f"[+] Saved {train_out}")
        print(f"[+] Saved {test_out}")
    
    # Also save small CSV sample for review
    train_sample_csv = os.path.join(ARTIFACTS_DIR, "train_features_sample.csv")
    train_data.head(1000).to_csv(train_sample_csv, index=False)
    print(f"[+] Saved feature sample: {train_sample_csv}")
    
    return train_data, test_data


if __name__ == "__main__":
    build_preprocessed_dataset()
