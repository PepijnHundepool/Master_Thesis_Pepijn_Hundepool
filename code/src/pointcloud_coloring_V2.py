import pandas as pd
import numpy as np
import open3d as o3d
import itertools

def load_and_color_pointcloud(csv_file, color_palette):
    """
    Loads a point cloud from a CSV file and assigns colors based on categoryID.
    """
    print(f"[INFO] Loading point cloud from {csv_file}")
    df = pd.read_csv(csv_file, delimiter=';')

    # Ensure required columns exist
    required_cols = ["X", "Y", "Z", "categoryID"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain at least {required_cols}")

    xyz = df[["X", "Y", "Z"]].values
    category_ids = df["categoryID"].astype(str).values  # Convert to string for consistent mapping

    # Define a color map
    unique_categories = np.unique(category_ids)
    color_cycle = itertools.cycle(color_palette)
    color_map = {category: next(color_cycle) for category in unique_categories}

    # Assign colors based on categoryID
    colors = np.array([color_map[cid] for cid in category_ids])

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# Define a set of distinguishable colors
COLOR_PALETTE = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 0.5, 0.0],  # Orange
    [0.5, 0.0, 1.0],  # Purple
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.0, 1.0],  # Magenta
    [0.5, 1.0, 0.0],  # Lime
    [0.0, 0.5, 1.0],  # Sky Blue
    [1.0, 0.0, 0.5],  # Pink
    [0.5, 0.5, 0.5],  # Gray
]

# File paths (update accordingly)
# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\preprocessed_csv_files\room_3_occluded_robot_scan_frames_17_to_17_object_filtered_preprocessed_rotated_and_translated.csv" # change this per run
# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\labels_csv\room_3_occluded_dt_scan_frames_17_to_17_object_filtered_preprocessed_labels.csv" # change this per run
# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\labels_csv_filtered\room_3_occluded_dt_scan_frames_17_to_17_object_filtered_preprocessed_labels.csv" # change this per run

# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\csv_files\room_test_1_occluded_robot_scan_wall_removed_frames_1_to_1.csv" # change this per run
# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\object_filtered_csv_files\room_test_1_occluded_robot_scan_wall_removed_frames_1_to_1_object_filtered.csv" # change this per run
# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\preprocessed_csv_files\room_test_1_occluded_robot_scan_wall_removed_frames_1_to_1_object_filtered_preprocessed.csv" # change this per run

# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_different_mismatch_type\pillar_added_and_removed\occluded\csv_files\room_1_occluded_dt_scan_pillar_added_removed_frames_17_to_17.csv"
# clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_different_mismatch_type\pillar_added_and_removed\occluded\object_filtered_csv_files\room_1_occluded_robot_scan_pillar_added_removed_frames_17_to_17_object_filtered.csv"
clean_csv_file = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\preprocessed_csv_files\room_1_occluded_robot_scan_chaos_frames_17_to_17_object_filtered_preprocessed.csv"

# Load and color point clouds
pcd_clean = load_and_color_pointcloud(clean_csv_file, COLOR_PALETTE)

# Display the original colored point cloud
print("[INFO] Displaying colored clean point cloud")
o3d.visualization.draw_geometries([pcd_clean], window_name="Colored Clean Point Cloud")
