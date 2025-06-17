import os
import numpy as np
from scipy.spatial import cKDTree

def compute_per_point_labels(dt_scan, robot_scan, threshold=0.05):
    """Generates per-point labels by comparing DT-scan and Robot-scan."""
    dt_tree = cKDTree(dt_scan)  # Build KDTree for fast lookup
    distances, _ = dt_tree.query(robot_scan)  # Find nearest DT-scan point for each Robot-scan point
    
    labels = np.zeros(len(robot_scan), dtype=np.float32)
    labels[distances > threshold] = 1  # Mark mismatched points
    return labels

def process_pointcloud_pairs(npy_folder, labels_folder, threshold=0.05):
    """Processes all point cloud pairs and generates per-point labels."""
    os.makedirs(labels_folder, exist_ok=True)
    
    files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
    
    for file in files:
        if "robot_scan" in file:
            # This is a mismatched robot-scan
            dt_file = file.replace("robot_scan", "dt_scan")
            dt_scan_path = os.path.join(npy_folder, dt_file)
            robot_scan_path = os.path.join(npy_folder, file)
            labels_path = os.path.join(labels_folder, file.replace(".npy", "_labels.npy"))

            if os.path.exists(labels_path):
                print(f"[SKIP] Labels already exist for {file} → {labels_path}")
                continue

            if not os.path.exists(dt_scan_path):
                print(f"[SKIP] DT-scan not found for {file} → {dt_scan_path}")
                continue

            print(f"[INFO] Processing changed sample: {file}")
        else:
            # This is an unchanged DT-scan sample
            dt_scan_path = os.path.join(npy_folder, file)
            robot_scan_path = dt_scan_path
            labels_path = os.path.join(labels_folder, file.replace(".npy", "_labels.npy"))

            if os.path.exists(labels_path):
                print(f"[SKIP] Labels already exist for {file} → {labels_path}")
                continue

            print(f"[INFO] Processing unchanged sample: {file}")
                
        dt_scan = np.load(dt_scan_path)
        robot_scan = np.load(robot_scan_path)
        
        labels = compute_per_point_labels(dt_scan, robot_scan, threshold)

        if dt_scan_path == robot_scan_path:
            # Sanity check for unchanged samples
            if np.any(labels != 0):
                print(f"[WARNING] Unchanged sample {file} contains non-zero labels!")
            else:
                print(f"[INFO] Verified: unchanged sample {file} has all-zero labels.")

        np.save(labels_path, labels)
        print(f"Saved per-point labels for {file} → {labels_path}")

def main():
    npy_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\clean\npy_files" # change for right dataset
    labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\clean\labels" # change for right dataset
    process_pointcloud_pairs(npy_folder, labels_folder, threshold=0.09)

if __name__ == "__main__":
    main()