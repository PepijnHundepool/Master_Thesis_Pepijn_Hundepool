import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from pathlib import Path
from collections import deque

def grow_cluster(points, categoryIDs, kdtree, assigned, start_idx, k_neighbors, verticality_thresh, max_neighbor_distance):
    """
    Grow a cluster starting from seed point, using distance + PCA-based shape checks + categoryID purity.
    """
    cluster_indices = []
    queue = deque()
    queue.append(start_idx)

    while queue:
        idx = queue.popleft()
        if assigned[idx]:
            continue

        assigned[idx] = True
        cluster_indices.append(idx)

        distances, indices = kdtree.query(points[idx].reshape(1, -1), k=k_neighbors, return_distance=True)
        distances = distances[0]
        indices = indices[0]

        for nbr_idx, dist in zip(indices, distances):
            if assigned[nbr_idx]:
                continue

            # 1. Distance filter
            if dist > max_neighbor_distance:
                continue

            # 2. CategoryID purity check
            tentative_indices = cluster_indices + [nbr_idx]
            tentative_categoryIDs = categoryIDs[tentative_indices]
            if not np.allclose(tentative_categoryIDs, tentative_categoryIDs[0], atol=1e-4):
                continue  # ❌ Reject neighbor if categoryID differs

            # 3. Tentative PCA verticality check
            temp_cluster_points = points[tentative_indices]

            if len(temp_cluster_points) >= 3:
                pca = PCA(n_components=3)
                pca.fit(temp_cluster_points)
                dominant_axis = pca.components_[0]
                verticality = np.abs(np.dot(dominant_axis, np.array([0, 0, 1])))
                if verticality < verticality_thresh:
                    continue

            # ✅ All checks passed: accept neighbor
            queue.append(nbr_idx)

    return cluster_indices

def filter_labels_pca_clustering(input_dir, output_dir, k_neighbors=50,
                                  verticality_thresh = 0.99, max_neighbor_distance=0.15,
                                  min_cluster_size=10, min_height=1.5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.csv"):
        df = pd.read_csv(file, sep=";")
        if "label" not in df.columns or "categoryID" not in df.columns:
            print(f"[SKIP] Missing required columns in: {file.name}")
            continue

        original_len = len(df)
        mismatch_mask = df["label"] == 1
        mismatch_df = df[mismatch_mask].copy()

        if mismatch_df.empty:
            df.to_csv(output_dir / file.name, sep=";", index=False)
            print(f"[SKIP] No mismatched points in: {file.name}")
            continue

        points = mismatch_df[["X", "Y", "Z"]].values
        categoryIDs = mismatch_df["categoryID"].values
        kdtree = KDTree(points)
        assigned = np.zeros(len(points), dtype=bool)

        num_kept_clusters = 0  # Counter for kept clusters
        kept_indices = []

        for idx in range(len(points)):
            if assigned[idx]:
                continue

            cluster_indices = grow_cluster(points, categoryIDs, kdtree, assigned, idx,
                                           k_neighbors, verticality_thresh, max_neighbor_distance)

            if len(cluster_indices) < min_cluster_size:
                continue

            cluster_points = points[cluster_indices]
            z_extent = cluster_points[:, 2].max() - cluster_points[:, 2].min()

            if z_extent < min_height:
                continue

            num_kept_clusters += 1  # Count this valid cluster

            # Passed all checks
            kept_indices.extend(mismatch_df.iloc[cluster_indices].index.tolist())

        # Overwrite labels
        df["label"] = 0
        df.loc[kept_indices, "label"] = 1
        df.to_csv(output_dir / file.name, sep=";", index=False)

        print(f"[DONE] {file.name} | kept {len(kept_indices)} / {original_len} points")
        print(f" → Clusters kept: {num_kept_clusters}")

if __name__ == "__main__":
    # csv_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_wall_removed\clean\preprocessed_csv_files", # change this per run
    # label_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_wall_removed\clean\labels_csv" # change this per run
    geometry_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\labels_csv" # change this per run
    output_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\labels_csv_filtered" # change this per run

    filter_labels_pca_clustering(geometry_folder, output_folder,
                                 k_neighbors=50,
                                 verticality_thresh=0.99,
                                 max_neighbor_distance=0.1349,
                                 min_cluster_size=30,
                                 min_height=2.5)
