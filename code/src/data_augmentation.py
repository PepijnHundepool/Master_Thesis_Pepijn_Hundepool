import os
import pandas as pd
import numpy as np
import open3d as o3d
from pathlib import Path
import re

def should_skip_augmentation(base_name, output_folder, suffixes=("_rotated", "_translated", "_rotated_and_translated")):
    """
    Returns True if any augmented file for this base_name already exists.
    """
    output_dir = Path(output_folder)
    for suffix in suffixes:
        if (output_dir / f"{base_name}{suffix}.csv").exists():
            return True
    return False

def apply_random_rotation(df, angle=None, axis=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(df[["X", "Y", "Z"]].values)

    if angle is None:
        angle = np.random.uniform(-np.pi, np.pi)
    if axis is None:
        axis = np.random.choice(["X", "Y", "Z"])

    c, s = np.cos(angle), np.sin(angle)
    if axis == "X":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "Y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # Z-axis
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    rotated = np.asarray(cloud.points) @ R.T
    df[["X", "Y", "Z"]] = rotated
    return df

def apply_random_translation(df, translation=None):
    if translation is None:
        translation = np.random.uniform(-10, 10, size=3)
    df[["X", "Y", "Z"]] += translation
    return df

def save_augmented_pointcloud(df, output_path):
    df.to_csv(f"{output_path}.csv", sep=';', index=False)

def augment_pair(clean_path, added_path, output_folder):
    clean_df = pd.read_csv(clean_path, sep=';')
    added_df = pd.read_csv(added_path, sep=';')

    # Construct output filenames
    base_clean = os.path.splitext(os.path.basename(clean_path))[0]
    base_added = os.path.splitext(os.path.basename(added_path))[0]

    if should_skip_augmentation(base_clean, output_folder) or should_skip_augmentation(base_added, output_folder):
        print(f"[SKIP] Augmentation skipped for: {base_clean} / {base_added}")
        return

    # Generate shared transformation
    angle = np.random.uniform(-np.pi/2, np.pi/2)
    axis = np.random.choice(["X", "Y", "Z"])
    translation = np.random.uniform(-5, 5, size=3)

    # Apply same transforms to both
    clean_rotated = apply_random_rotation(clean_df.copy(), angle, axis)
    added_rotated = apply_random_rotation(added_df.copy(), angle, axis)

    clean_translated = apply_random_translation(clean_df.copy(), translation)
    added_translated = apply_random_translation(added_df.copy(), translation)

    clean_rt = apply_random_translation(clean_rotated.copy(), translation)
    added_rt = apply_random_translation(added_rotated.copy(), translation)

    save_augmented_pointcloud(clean_rotated, os.path.join(output_folder, f"{base_clean}_rotated"))
    save_augmented_pointcloud(added_rotated, os.path.join(output_folder, f"{base_added}_rotated"))
    save_augmented_pointcloud(clean_translated, os.path.join(output_folder, f"{base_clean}_translated"))
    save_augmented_pointcloud(added_translated, os.path.join(output_folder, f"{base_added}_translated"))
    save_augmented_pointcloud(clean_rt, os.path.join(output_folder, f"{base_clean}_rotated_and_translated"))
    save_augmented_pointcloud(added_rt, os.path.join(output_folder, f"{base_added}_rotated_and_translated"))

def process_folder(folder):
    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])

    # Define augmentation suffixes to exclude
    suffixes = ["_rotated", "_translated", "_rotated_and_translated"]

    # Filter to original (non-augmented) files
    dt_files = [
        f for f in all_files
        if "dt_scan" in f and not any(s in f for s in suffixes) and not f.startswith("room_test")
    ]
    robot_files = [
        f for f in all_files
        if "robot_scan" in f and not any(s in f for s in suffixes) and not f.startswith("room_test")
    ]

    for dt_file in dt_files:
        room_id = dt_file.split("_dt_scan")[0]
        robot_file = next((f for f in robot_files if f.startswith(room_id)), None)

        if not robot_file:
            print(f"[WARNING] No matching robot scan for: {dt_file}")
            continue

        dt_path = os.path.join(folder, dt_file)
        robot_path = os.path.join(folder, robot_file)

        augment_pair(dt_path, robot_path, folder)

if __name__ == "__main__":
    folder_path = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_pillar_removed\clean\labels_csv_filtered" # change to desired folder
    process_folder(folder_path)
