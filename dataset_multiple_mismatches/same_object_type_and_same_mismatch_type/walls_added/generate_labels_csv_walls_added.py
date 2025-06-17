import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def compute_per_point_labels(dt_scan, robot_scan, threshold=0.05):
    dt_tree = cKDTree(dt_scan)
    distances, _ = dt_tree.query(robot_scan)
    labels = np.zeros(len(robot_scan), dtype=np.float32)
    labels[distances > threshold] = 1
    return labels

def process_pointcloud_pairs(csv_folder, labels_folder, threshold=0.05):
    os.makedirs(labels_folder, exist_ok=True)

    files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    for file in files:
        if "robot_scan" in file:
            # This is a mismatched robot scan
            dt_file = file.replace("robot_scan", "dt_scan")
            dt_path = os.path.join(csv_folder, dt_file)
            robot_path = os.path.join(csv_folder, file)
            labels_path = os.path.join(labels_folder, file.replace(".csv", "_labels.csv"))

            # if os.path.exists(labels_path):
            #     print(f"[SKIP] Labels already exist for {file} → {labels_path}")
            #     continue
            if not os.path.exists(dt_path):
                print(f"[SKIP] DT-scan not found for {file} → {dt_path}")
                continue

            print(f"[INFO] Processing changed sample: {file}")
        else:
            # This is an unchanged DT scan
            dt_path = os.path.join(csv_folder, file)
            robot_path = dt_path
            labels_path = os.path.join(labels_folder, file.replace(".csv", "_labels.csv"))

            # if os.path.exists(labels_path):
            #     print(f"[SKIP] Labels already exist for {file} → {labels_path}")
            #     continue

            print(f"[INFO] Processing unchanged sample: {file}")

        dt_df = pd.read_csv(dt_path, sep=';')
        robot_df = pd.read_csv(robot_path, sep=';')

        dt_scan = dt_df[["X", "Y", "Z"]].values
        robot_scan = robot_df[["X", "Y", "Z"]].values

        labels = compute_per_point_labels(dt_scan, robot_scan, threshold)

        if dt_path == robot_path:
            if np.any(labels != 0):
                print(f"[WARNING] Unchanged sample {file} contains non-zero labels!")
            else:
                print(f"[INFO] Verified: unchanged sample {file} has all-zero labels.")
            dt_df["label"] = labels
            dt_df.to_csv(labels_path, sep=';', index=False)
        else:
            robot_df["label"] = labels
            robot_df.to_csv(labels_path, sep=';', index=False)

def main():
    # csv_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\occluded\preprocessed_csv_files" # change this per run
    # labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\occluded\labels_csv" # change this per run

    # csv_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\preprocessed_csv_files" # change this per run
    # labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\labels_csv" # change this per run

    csv_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\walls_added\occluded\preprocessed_csv_files"
    labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\walls_added\occluded\labels_csv"
    process_pointcloud_pairs(csv_folder, labels_folder, threshold=0.09)

if __name__ == "__main__":
    main()
