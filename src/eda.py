"""
PathSnap: Exploratory Data Analysis (EDA) Module
Author: Team PathSnap
PES University - UE24CS352A Machine Learning Hackathon
"""

import os
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_FOLDER = os.path.join(BASE_DIR, "Student folder")
TRAIN_DIR = os.path.join(STUDENT_FOLDER, "train")
TEST_DIR = os.path.join(STUDENT_FOLDER, "student_test")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
MAPS_DIR = os.path.join(ARTIFACTS_DIR, "maps")


def analyze_trips(partition_dir, partition_name="train"):
    """
    Parses and summarizes all trips in a given partition.
    Returns:
        DataFrame with trip-level metrics, and raw delta_t array.
    """
    trip_dirs = sorted(glob.glob(os.path.join(partition_dir, "*")))
    records = []
    all_dt = []
    
    for tdir in trip_dirs:
        if not os.path.isdir(tdir):
            continue
        tid = os.path.basename(tdir)
        track_path = os.path.join(tdir, f"{tid}.track")
        nodes_path = os.path.join(tdir, f"{tid}.nodes")
        arcs_path = os.path.join(tdir, f"{tid}.arcs")
        route_path = os.path.join(tdir, f"{tid}.route")
        
        # 1. Track Analysis
        track_pts = []
        with open(track_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        track_pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    except ValueError:
                        pass
        
        n_points = len(track_pts)
        if n_points == 0:
            continue
            
        timestamps = [p[2] for p in track_pts]
        lons = [p[0] for p in track_pts]
        lats = [p[1] for p in track_pts]
        
        duration = timestamps[-1] - timestamps[0]
        dt_list = [timestamps[i+1] - timestamps[i] for i in range(n_points - 1)]
        all_dt.extend(dt_list)
        
        # Missing / invalid check
        nan_count = sum(1 for p in track_pts if math.isnan(p[0]) or math.isnan(p[1]) or math.isnan(p[2]))
        
        # 2. Nodes Analysis
        n_nodes = 0
        with open(nodes_path, 'r') as f:
            for _ in f:
                n_nodes += 1
                
        # 3. Arcs Analysis
        n_arcs = 0
        with open(arcs_path, 'r') as f:
            for _ in f:
                n_arcs += 1
                
        # 4. Route Analysis
        n_route_arcs = 0
        has_route = os.path.exists(route_path)
        if has_route:
            with open(route_path, 'r') as f:
                for _ in f:
                    n_route_arcs += 1
                    
        records.append({
            "partition": partition_name,
            "trip_id": tid,
            "gps_points": n_points,
            "duration_sec": duration,
            "duration_min": duration / 60.0,
            "min_lon": min(lons),
            "max_lon": max(lons),
            "min_lat": min(lats),
            "max_lat": max(lats),
            "mean_dt": np.mean(dt_list) if dt_list else 0.0,
            "max_dt": max(dt_list) if dt_list else 0.0,
            "pct_1s_dt": (sum(1 for dt in dt_list if 0.99 <= dt <= 1.01) / len(dt_list) * 100) if dt_list else 0.0,
            "nan_count": nan_count,
            "num_nodes": n_nodes,
            "num_arcs": n_arcs,
            "num_route_arcs": n_route_arcs,
            "has_route": has_route
        })
        
    df = pd.DataFrame(records)
    return df, np.array(all_dt)


def generate_eda_report():
    print("=" * 60)
    print("PathSnap: Running Exploratory Data Analysis...")
    print("=" * 60)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)
    
    train_df, train_dt = analyze_trips(TRAIN_DIR, "train")
    test_df, test_dt = analyze_trips(TEST_DIR, "student_test")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save CSV inventory
    inventory_path = os.path.join(ARTIFACTS_DIR, "dataset_inventory.csv")
    combined_df.to_csv(inventory_path, index=False)
    print(f"[+] Dataset inventory saved to {inventory_path}")
    
    # Overall summary stats
    summary = {
        "train_trips": len(train_df),
        "test_trips": len(test_df),
        "total_trips": len(combined_df),
        "train_total_gps_points": int(train_df["gps_points"].sum()),
        "test_total_gps_points": int(test_df["gps_points"].sum()),
        "gps_points_per_trip_train": {
            "mean": float(train_df["gps_points"].mean()),
            "std": float(train_df["gps_points"].std()),
            "min": int(train_df["gps_points"].min()),
            "median": float(train_df["gps_points"].median()),
            "max": int(train_df["gps_points"].max()),
        },
        "road_network_nodes_train": {
            "mean": float(train_df["num_nodes"].mean()),
            "min": int(train_df["num_nodes"].min()),
            "max": int(train_df["num_nodes"].max())
        },
        "road_network_arcs_train": {
            "mean": float(train_df["num_arcs"].mean()),
            "min": int(train_df["num_arcs"].min()),
            "max": int(train_df["num_arcs"].max())
        },
        "route_arcs_train": {
            "mean": float(train_df["num_route_arcs"].mean()),
            "min": int(train_df["num_route_arcs"].min()),
            "max": int(train_df["num_route_arcs"].max())
        },
        "sampling_dt_percentage_1s": float((np.isclose(train_dt, 1.0, atol=0.05).sum() / len(train_dt)) * 100),
        "missing_invalid_values": int(combined_df["nan_count"].sum())
    }
    
    summary_path = os.path.join(ARTIFACTS_DIR, "eda_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[+] EDA summary statistics saved to {summary_path}")
    
    # Print clean summary table
    print("\n--- Summary Statistics (Train vs Student Test) ---")
    print(f"Trips: Train={summary['train_trips']}, Test={summary['test_trips']}")
    print(f"Total Observations: Train={summary['train_total_gps_points']:,}, Test={summary['test_total_gps_points']:,}")
    print(f"GPS points / trip (Train): Min={summary['gps_points_per_trip_train']['min']}, Median={summary['gps_points_per_trip_train']['median']:.0f}, Max={summary['gps_points_per_trip_train']['max']}")
    print(f"Sampling interval 1 Hz adherence: {summary['sampling_dt_percentage_1s']:.2f}%")
    print(f"Missing / corrupted values: {summary['missing_invalid_values']} (Clean)")
    
    # ----------------- Visualizations -----------------
    # Plot 1: GPS Points and Duration Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(train_df['gps_points'], bins=25, kde=True, color='#2b5c8f', ax=axes[0])
    axes[0].set_title("Distribution of GPS Points per Trip (Train)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Number of GPS Observations")
    axes[0].set_ylabel("Trip Count")
    
    sns.histplot(train_df['duration_min'], bins=25, kde=True, color='#2ca02c', ax=axes[1])
    axes[1].set_title("Distribution of Trip Duration (Minutes)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Duration (Minutes)")
    axes[1].set_ylabel("Trip Count")
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "trajectory_distributions.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] Plot saved: {plot1_path}")
    
    # Plot 2: Road Network Sizes (Nodes vs Arcs)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=train_df, x='num_nodes', y='num_arcs', color='#e6550d', alpha=0.8, s=50, ax=ax)
    ax.set_title("Road Network Complexity: Arcs vs Nodes per Trip", fontsize=12, fontweight='bold')
    ax.set_xlabel("Number of Road Nodes")
    ax.set_ylabel("Number of Road Segments (Arcs)")
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "road_network_complexity.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] Plot saved: {plot2_path}")
    
    # Plot 3: Sampling Interval Distribution (log scale)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dt_series = pd.Series(train_dt).clip(0, 10)
    sns.histplot(dt_series, bins=20, color='#756bb1', discrete=True, ax=ax)
    ax.set_title("GPS Sampling Interval Distribution (Δt in seconds)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time Delta (seconds)")
    ax.set_ylabel("Observation Count")
    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "sampling_interval_dist.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] Plot saved: {plot3_path}")
    
    # ----------------- Folium Map Generator -----------------
    generate_sample_folium_map(os.path.join(TRAIN_DIR, "00000001"))
    

def generate_sample_folium_map(trip_dir):
    tid = os.path.basename(trip_dir)
    track_path = os.path.join(trip_dir, f"{tid}.track")
    nodes_path = os.path.join(trip_dir, f"{tid}.nodes")
    arcs_path = os.path.join(trip_dir, f"{tid}.arcs")
    route_path = os.path.join(trip_dir, f"{tid}.route")
    
    # Load nodes
    nodes = []
    with open(nodes_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                nodes.append((float(parts[0]), float(parts[1])))
                
    # Load arcs
    arcs = []
    with open(arcs_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                arcs.append((int(parts[0]), int(parts[1])))
                
    # Load track
    track = []
    with open(track_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                track.append((float(parts[0]), float(parts[1]), float(parts[2])))
                
    # Load route
    route_arcs = []
    if os.path.exists(route_path):
        with open(route_path) as f:
            for line in f:
                l = line.strip()
                if l:
                    route_arcs.append(int(l))
                    
    # Center map on median GPS point
    lons = [p[0] for p in track]
    lats = [p[1] for p in track]
    center_lat = np.median(lats)
    center_lon = np.median(lons)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
    
    # 1. Road network layer (sample arcs within bounding box + margin to keep HTML lightweight)
    min_lon, max_lon = min(lons) - 0.01, max(lons) + 0.01
    min_lat, max_lat = min(lats) - 0.01, max(lats) + 0.01
    
    network_group = folium.FeatureGroup(name="Road Network (Arcs)", show=True)
    # Add relevant arcs
    route_set = set(route_arcs)
    arc_count = 0
    for idx, (u, v) in enumerate(arcs):
        if idx not in route_set:
            lon_u, lat_u = nodes[u]
            lon_v, lat_v = nodes[v]
            if (min_lon <= lon_u <= max_lon and min_lat <= lat_u <= max_lat) or \
               (min_lon <= lon_v <= max_lon and min_lat <= lat_v <= max_lat):
                folium.PolyLine(
                    locations=[[lat_u, lon_u], [lat_v, lon_v]],
                    color="#aaaaaa",
                    weight=1.5,
                    opacity=0.6
                ).add_to(network_group)
                arc_count += 1
                if arc_count > 3000:  # limit for browser rendering performance
                    break
    network_group.add_to(m)
    
    # 2. Ground truth route layer
    if route_arcs:
        route_group = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
        route_coords = []
        for a_id in route_arcs:
            u, v = arcs[a_id]
            route_coords.append([nodes[u][1], nodes[u][0]])
        # Add last node
        last_v = arcs[route_arcs[-1]][1]
        route_coords.append([nodes[last_v][1], nodes[last_v][0]])
        
        folium.PolyLine(
            locations=route_coords,
            color="#2ca02c",
            weight=4.5,
            opacity=0.85,
            tooltip=f"Ground Truth Route (Trip {tid})"
        ).add_to(route_group)
        route_group.add_to(m)
        
    # 3. Raw GPS track layer
    track_group = folium.FeatureGroup(name="Raw GPS Observations (Blue Dots)", show=True)
    for i, (lon, lat, ts) in enumerate(track):
        # sample dots if trajectory is long to keep map smooth
        if len(track) > 1000 and i % 2 != 0:
            continue
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            popup=f"Obs #{i}<br>Time: {ts:.1f}s<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
        ).add_to(track_group)
    track_group.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    out_map = os.path.join(MAPS_DIR, f"eda_map_trip_{tid}.html")
    m.save(out_map)
    print(f"[+] Folium interactive map generated: {out_map}")


if __name__ == "__main__":
    generate_eda_report()
