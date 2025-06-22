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

# File paths
clean_csv_file = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_pillar_removed\clean\preprocessed_csv_files\room_1_clean_robot_scan_frames_17_to_17_object_filtered_preprocessed.csv" # change this to desired file 

# Load and color point clouds
pcd_clean = load_and_color_pointcloud(clean_csv_file, COLOR_PALETTE)

# Display the original colored point cloud
print("[INFO] Displaying colored clean point cloud")
o3d.visualization.draw_geometries([pcd_clean], window_name="Colored Clean Point Cloud")
