import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def compute_per_point_labels(dt_scan, robot_scan, threshold=0.09):
    robot_tree = cKDTree(robot_scan)  # robot scan is the reference
    distances, _ = robot_tree.query(dt_scan)  # query FROM dt_scan TO robot_scan
    labels = np.zeros(len(dt_scan), dtype=np.float32)
    labels[distances > threshold] = 1
    return labels

def process_pointcloud_pairs(csv_folder, labels_folder, threshold=0.09):
    os.makedirs(labels_folder, exist_ok=True)

    files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    for file in files:
        if "dt_scan" in file:
            # This is a mismatched DT scan (the object is still here, but gone in robot scan)
            robot_file = file.replace("dt_scan", "robot_scan")
            dt_path = os.path.join(csv_folder, file)
            robot_path = os.path.join(csv_folder, robot_file)
            labels_path = os.path.join(labels_folder, file.replace(".csv", "_labels.csv"))

            if not os.path.exists(robot_path):
                print(f"[SKIP] Robot scan not found for {file} → {robot_path}")
                continue

            print(f"[INFO] Processing REMOVED mismatch sample: {file}")
        else:
            # This is an unchanged DT scan
            dt_path = os.path.join(csv_folder, file)
            robot_path = dt_path
            labels_path = os.path.join(labels_folder, file.replace(".csv", "_labels.csv"))

            print(f"[INFO] Processing unchanged sample: {file}")

        # Read CSVs with semicolon separator and column names
        dt_df = pd.read_csv(dt_path, sep=";")
        robot_df = pd.read_csv(robot_path, sep=";")

        dt_points = dt_df[["X", "Y", "Z"]].values
        robot_points = robot_df[["X", "Y", "Z"]].values

        # Compute mismatch labels for DT-scan points (robot scan = reference)
        labels = compute_per_point_labels(dt_points, robot_points, threshold)

        if dt_path == robot_path:
            if np.any(labels != 0):
                print(f"[WARNING] Unchanged sample {file} contains non-zero labels!")
            else:
                print(f"[INFO] Verified: unchanged sample {file} has all-zero labels.")

        # Save label column alongside the DT scan
        dt_df["label"] = labels
        dt_df.to_csv(labels_path, sep=";", index=False)
        print(f"[SAVED] Labels → {labels_path}")

def main():
    # csv_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\preprocessed_csv_files" # change this per run
    # labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\labels_csv" # change this per run
    
    csv_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_same_mismatch_type\objects_removed\occluded\preprocessed_csv_files"
    labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_same_mismatch_type\objects_removed\occluded\labels_csv"
    process_pointcloud_pairs(csv_folder, labels_folder, threshold=0.09)

if __name__ == "__main__":
    main()
