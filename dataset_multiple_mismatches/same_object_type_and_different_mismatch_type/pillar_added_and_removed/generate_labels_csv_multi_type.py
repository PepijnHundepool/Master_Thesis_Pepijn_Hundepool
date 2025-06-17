import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def compute_labels_added(dt_scan, robot_scan, threshold):
    dt_tree = cKDTree(dt_scan)
    distances, _ = dt_tree.query(robot_scan)
    labels = np.zeros(len(robot_scan), dtype=np.float32)
    labels[distances > threshold] = 1
    return labels

def compute_labels_removed(dt_scan, robot_scan, threshold):
    robot_tree = cKDTree(robot_scan)
    distances, _ = robot_tree.query(dt_scan)
    labels = np.zeros(len(dt_scan), dtype=np.float32)
    labels[distances > threshold] = 1
    return labels

def process_pointcloud_pairs(csv_folder, labels_folder_added, labels_folder_removed, threshold=0.09):
    os.makedirs(labels_folder_added, exist_ok=True)
    os.makedirs(labels_folder_removed, exist_ok=True)

    files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    for file in files:
        robot_path = os.path.join(csv_folder, file.replace("dt_scan", "robot_scan"))
        dt_path = os.path.join(csv_folder, file.replace("robot_scan", "dt_scan"))

        is_robot = "robot_scan" in file
        is_dt = "dt_scan" in file

        if is_robot:
            if not os.path.exists(dt_path):
                print(f"[SKIP] DT-scan not found for {file} → {dt_path}")
                continue

            print(f"[INFO] Processing ADDED mismatch (robot-scan): {file}")
            dt_df = pd.read_csv(dt_path, sep=';')
            robot_df = pd.read_csv(os.path.join(csv_folder, file), sep=';')

            dt_scan = dt_df[["X", "Y", "Z"]].values
            robot_scan = robot_df[["X", "Y", "Z"]].values
            labels = compute_labels_added(dt_scan, robot_scan, threshold)

            robot_df["label"] = labels
            save_path = os.path.join(labels_folder_added, file.replace(".csv", "_labels.csv"))
            robot_df.to_csv(save_path, sep=';', index=False)
            print(f"[SAVED] ADDED labels → {save_path}")

        elif is_dt:
            if not os.path.exists(robot_path):
                print(f"[SKIP] Robot scan not found for {file} → {robot_path}")
                continue

            print(f"[INFO] Processing REMOVED mismatch (dt-scan): {file}")
            dt_df = pd.read_csv(os.path.join(csv_folder, file), sep=';')
            robot_df = pd.read_csv(robot_path, sep=';')

            dt_scan = dt_df[["X", "Y", "Z"]].values
            robot_scan = robot_df[["X", "Y", "Z"]].values
            labels = compute_labels_removed(dt_scan, robot_scan, threshold)

            dt_df["label"] = labels
            save_path = os.path.join(labels_folder_removed, file.replace(".csv", "_labels.csv"))
            dt_df.to_csv(save_path, sep=';', index=False)
            print(f"[SAVED] REMOVED labels → {save_path}")

        else:
            print(f"[SKIP] Unrecognized file: {file}")

def main():
    csv_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\preprocessed_csv_files"  # change as needed
    labels_folder_added = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv"
    labels_folder_removed = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv"

    process_pointcloud_pairs(csv_folder, labels_folder_added, labels_folder_removed, threshold=0.09)

if __name__ == "__main__":
    main()
