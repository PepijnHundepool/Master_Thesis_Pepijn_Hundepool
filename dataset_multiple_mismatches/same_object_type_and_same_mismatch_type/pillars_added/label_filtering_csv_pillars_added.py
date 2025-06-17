"""
label_filtering_csv.py
Filters false mismatch labels (shadows) in robot-scan or dt-scan based on categoryID consistency.
Used on .CSV files BEFORE conversion to .NPY.

Assumes the following:
- You have a .csv file with X, Y, Z, categoryID columns
- You have a corresponding .csv label file with one label per point (0 or 1)

This script maps labels back to 0 if their categoryID appears in a mixed cluster (shadow).
"""

import numpy as np
import pandas as pd
import os
from collections import defaultdict

def filter_shadow_labels(csv_folder, label_folder, min_points=30):
    for fname in os.listdir(label_folder):
        if not fname.endswith("_labels.csv"):
            continue

        label_path = os.path.join(label_folder, fname)
        scan_name = fname.replace("_labels.csv", ".csv")
        csv_path = os.path.join(csv_folder, scan_name)

        if not os.path.exists(csv_path):
            print(f"[SKIP] Missing CSV for: {fname}")
            continue

        df = pd.read_csv(label_path, sep=';')
        labels = df["label"].values
        category_ids = df["categoryID"].values.astype(np.float32)

        if "categoryID" not in df.columns:
            print(f"[ERROR] categoryID not found in: {csv_path}")
            continue

        category_ids = df["categoryID"].values.astype(np.float32)

        if len(category_ids) != len(labels):
            print(f"[ERROR] Length mismatch in: {fname}")
            continue

        # Find cluster size per categoryID
        mismatched_indices = np.where(labels == 1)[0]
        cluster_counts = defaultdict(int)
        for idx in mismatched_indices:
            cid = category_ids[idx]
            cluster_counts[cid] += 1

        # Identify valid categoryIDs
        valid_cids = [cid for cid, count in cluster_counts.items() if count >= min_points]

        # Filter labels
        for idx in mismatched_indices:
            if category_ids[idx] not in valid_cids:
                labels[idx] = 0

        df["label"] = labels  # update labels column
        df.to_csv(label_path, index=False, sep=";")
        print(f"[SAVED] Filtered labels written to: {label_path}")

if __name__ == "__main__":
    # Example usage — customize this
    filter_shadow_labels(
        # csv_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\occluded\preprocessed_csv_files", # change this per run
        # label_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\occluded\labels_csv" # change this per run

        # csv_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\preprocessed_csv_files", # change this per run
        # label_folder=r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\labels_csv_filtered" # change this per run

        csv_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\pillars_added\clean\labels",
        label_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\pillars_added\clean\labels_csv_filtered"
    )
