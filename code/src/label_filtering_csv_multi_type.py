import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from pathlib import Path
from collections import deque
from sklearn.cluster import DBSCAN
from collections import Counter

def filter_labels_pca_clustering(input_file, output_file, k_neighbors=50,
                                 verticality_thresh=0.99, max_neighbor_distance=0.175,
                                 min_cluster_size=1, min_height=2.4, min_category_purity=0.95, max_pca_ratio=10.0):

    df = pd.read_csv(input_file, sep=";")

    if "label" not in df.columns or "categoryID" not in df.columns:
        print(f"[SKIP] Missing required columns in: {input_file}")
        return

    original_len = len(df)
    mismatch_df = df[df["label"] == 1].copy()

    if mismatch_df.empty:
        print(f"[SKIP] No mismatched points in: {input_file}")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, sep=";", index=False)
        return

    points = mismatch_df[["X", "Y", "Z"]].values
    categoryIDs = mismatch_df["categoryID"].values

    # DBSCAN
    db = DBSCAN(eps=max_neighbor_distance, min_samples=3).fit(points)
    labels = db.labels_
    unique_labels = set(labels)
    kept_indices = []

    for label in unique_labels:
        if label == -1:
            continue  # skip noise

        cluster_mask = labels == label
        cluster_points = points[cluster_mask]
        cluster_categoryIDs = categoryIDs[cluster_mask]

        most_common, count = Counter(cluster_categoryIDs).most_common(1)[0]
        purity = count / len(cluster_categoryIDs)

        if len(cluster_points) < min_cluster_size:
            print(f"[REJECT] Cluster too small ({len(cluster_points)} < {min_cluster_size})")
            continue

        if purity < min_category_purity:
            print(f"[REJECT] Cluster purity too low ({purity:.2f} < {min_category_purity})")
            continue

        z_extent = cluster_points[:, 2].max() - cluster_points[:, 2].min()
        if z_extent < min_height:
            print(f"[REJECT] Cluster z-extent too small ({z_extent:.2f} < {min_height})")
            continue

        pca_xy = PCA(n_components=2).fit(cluster_points[:, :2])
        ratio = pca_xy.explained_variance_[0] / (pca_xy.explained_variance_[1] + 1e-6)
        if ratio > max_pca_ratio:
            print(f"[REJECT] PCA XY ratio too high ({ratio:.2f} > 10.0)")
            continue

        # Verticality check using 3D PCA
        pca_3d = PCA(n_components=3).fit(cluster_points)
        vertical_axis = np.array([0, 0, 1])
        verticality = np.abs(np.dot(pca_3d.components_[0], vertical_axis))

        if verticality < verticality_thresh:
            print(f"[REJECT] Cluster verticality too low ({verticality:.2f} < {verticality_thresh})")
            continue

        print(f"[KEEP] Cluster passed with size={len(cluster_points)}, z-extent={z_extent:.2f}, XY ratio={ratio:.2f}")
        kept_indices.extend(mismatch_df[cluster_mask].index.tolist())

    df["label"] = 0
    df.loc[kept_indices, "label"] = 1
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep=";", index=False)
    print(f"[DONE] {Path(input_file).name} | kept {len(kept_indices)} / {original_len} points")

def filter_labels_pca_clustering_brute(input_file, output_file,
                                       x_range, y_range, z_range):
    """ In cases where label filtering on pca_clustering fails, 
    this function can be used to manually cut away falsely labeled mismatched points 
    by viewing the labels with view_labels_csv_multi_type.py and filling in the desired 
    x,y,z ranges where mismatched points may occur. """
    df = pd.read_csv(input_file, sep=";")
    original_len = len(df)

    if not {"X", "Y", "Z", "label"}.issubset(df.columns):
        print(f"[SKIP] Required columns missing in: {input_file}")
        return

    # Define bounding box conditions
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    # Apply mask for brute-force mismatch area
    in_box = (
        (df["X"] >= x_min) & (df["X"] <= x_max) &
        (df["Y"] >= y_min) & (df["Y"] <= y_max) &
        (df["Z"] >= z_min) & (df["Z"] <= z_max)
    )

    # Reset all labels to 0 (unchanged), and restore label=1 for points in box
    df["label"] = 0
    df.loc[in_box, "label"] = 1

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep=";", index=False)

    kept = df["label"].sum()
    print(f"[BRUTE] {Path(input_file).name} | kept {kept} / {original_len} points")

if __name__ == "__main__":
    # === MODIFY PER RUN ===
    dt_label_file = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv\room_1_occluded_dt_scan_chaos_frames_17_to_17_object_filtered_preprocessed_labels.csv"
    dt_output_folder = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv_filtered_pillar\room_1_occluded_dt_scan_chaos_frames_17_to_17_object_filtered_preprocessed_labels.csv"

    robot_label_file = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv\room_1_occluded_robot_scan_chaos_frames_17_to_17_object_filtered_preprocessed_labels.csv"
    robot_output_folder = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv_filtered_pillar\room_1_occluded_robot_scan_chaos_frames_17_to_17_object_filtered_preprocessed_labels.csv"

    # filter_labels_pca_clustering(dt_label_file, dt_output_folder, min_cluster_size=10, min_height=0.2, max_neighbor_distance=0.175, min_category_purity=0.2, verticality_thresh=0.95,  max_pca_ratio=10.0)
    # print("Done with DT-scan")
    # filter_labels_pca_clustering(robot_label_file, robot_output_folder, min_cluster_size=10, min_height=0.2, max_neighbor_distance=0.175, min_category_purity=0.99, verticality_thresh=0.95,  max_pca_ratio=1000)
    # print("Done with robot-scan")

    # Brute-force override: keep only points inside bounding box
    filter_labels_pca_clustering_brute(
        input_file=dt_output_folder,
        output_file=dt_output_folder,  # overwrite
        x_range=(3, 4),
        y_range=(-1.3, -0.6),
        z_range=(-1.5, 1.5)
    )

    print("Done with DT-scan")
    
    # filter_labels_pca_clustering_brute(
    #     input_file=robot_output_folder,
    #     output_file=robot_output_folder,  # overwrite
    #     x_range=(-4, 6),
    #     y_range=(-2.4, -2.2),
    #     z_range=(-1.5, 1.5)
    # )

    # print("Done with Robot-scan")