import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import os
from pathlib import Path

def load_point_cloud_from_csv(file_path):
    """
    Loads a CSV file with at least 'x', 'y', 'z' columns.
    Returns the point array and original DataFrame.
    """
    df = pd.read_csv(file_path, sep=';')
    points = df[['X', 'Y', 'Z', "categoryID"]].values
    return points, df

def filter_by_category_zspan(df, z_threshold=1.5, floor_z=-1.5, ceiling_z=1.5, tolerance=0.1): # change per run
    """
    Filters objects based on categoryID and their Z-span,
    but always keeps groups near known Z-values for floor/ceiling.
    """
    if 'categoryID' not in df.columns:
        raise ValueError("Missing 'categoryID' column in input file.")

    keep_indices = []

    for category_id in df['categoryID'].unique():
        object_points = df[df['categoryID'] == category_id]
        z_values = object_points['Z']

        z_median = z_values.median()

        # Always keep floor and ceiling
        if abs(z_median - floor_z) < tolerance or abs(z_median - ceiling_z) < tolerance:
            keep_indices.extend(object_points.index.tolist())
            continue

        # Normal height-based filter
        # z_span = z_values.max() - z_values.min()
        # if z_span >= z_threshold:
        #     keep_indices.extend(object_points.index.tolist())

        # Filter based on max Z relative to floor
        z_top = z_values.max()
        if z_top - floor_z >= z_threshold:
            keep_indices.extend(object_points.index.tolist())

    return df.loc[keep_indices].reset_index(drop=True)

def filter_csv_pointcloud(file_path, output_path, z_threshold=1.5):
    df = pd.read_csv(file_path, sep=';')
    filtered_df = filter_by_category_zspan(df, z_threshold)
    # filtered_df.to_csv(output_path, sep=';', index=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, sep=';', index=False)
    print(f"Filtered point cloud saved to: {output_path}")
    return filtered_df

# Example usage (replace paths with real ones when calling this from your pipeline)
if __name__ == "__main__":
    # input_csv = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_moved\occluded\csv_files\room_test_3_occluded_robot_scan_frames_17_to_17.csv" # change this per run
    # output_csv = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_moved\occluded\object_filtered_csv_files\room_test_3_occluded_robot_scan_frames_17_to_17_object_filtered.csv" # change this per run

    # input_csv = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\csv_files\room_test_1_occluded_robot_scan_wall_removed_frames_1_to_1.csv" # change this per run
    # output_csv = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\object_filtered_csv_files\room_test_1_occluded_robot_scan_wall_removed_frames_1_to_1_object_filtered.csv" # change this per run

    input_csv = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\csv_files\room_1_occluded_dt_scan_chaos_frames_17_to_17.csv"
    output_csv = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\object_filtered_csv_files\room_1_occluded_dt_scan_chaos_frames_17_to_17_object_filtered.csv"
    filter_csv_pointcloud(input_csv, output_csv, z_threshold=2.4)

