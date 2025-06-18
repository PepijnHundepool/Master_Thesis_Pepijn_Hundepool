import os
import numpy as np
import pandas as pd
import open3d as o3d

def load_csv_pointcloud(file_path):
    """Loads a point cloud from a CSV file and keeps only X, Y, Z."""
    df = pd.read_csv(file_path, delimiter=";")
    # print(df.columns)
    required_cols = ["X", "Y", "Z", "categoryID"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain at least {required_cols}")
    # return df[required_cols]
    if "categoryID" in df.columns:
        return df[["X", "Y", "Z", "categoryID"]]
    else:
        return df[["X", "Y", "Z"]]

def voxel_downsample(df, voxel_size=0.1):
    """Applies voxel downsampling while preserving X, Y, Z."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(df[["X", "Y", "Z"]].values)
    downsampled_cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    # return pd.DataFrame(np.asarray(downsampled_cloud.points), columns=["X", "Y", "Z"])

    # Below code is for retaining categoryID
    downsampled_points = np.asarray(downsampled_cloud.points)

    # Find nearest neighbors from downsampled to original
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(df[["X", "Y", "Z"]].values)
    _, indices = nbrs.kneighbors(downsampled_points)

    # Select corresponding rows from original DataFrame
    df_downsampled = df.iloc[indices.flatten()].reset_index(drop=True)

    return df_downsampled

def remove_outliers(df, nb_neighbors=20, std_ratio=2.0):
    """
    Removes statistical outliers from the point cloud while preserving all original columns (e.g., categoryID).
    """
    print(f"[INFO] Applying statistical outlier removal: neighbors={nb_neighbors}, std_ratio={std_ratio}")

    # Convert only XYZ to Open3D format
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(df[["X", "Y", "Z"]].values)

    # Remove outliers
    clean_cloud, inlier_indices = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    # Return original DataFrame with inliers only (preserving categoryID or others)
    df_cleaned = df.iloc[inlier_indices].reset_index(drop=True)

    # Optional: Check if categoryID was preserved
    if "categoryID" in df.columns:
        assert "categoryID" in df_cleaned.columns, "categoryID not preserved in cleaned data"

    return df_cleaned

def preprocess_pointcloud_folder(input_folder, output_folder, voxel_size=0.1, remove_outliers_flag=False):
    """Processes all CSV point clouds in a folder and saves the results, skipping already processed files."""
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".csv", "_preprocessed.csv"))
            
            # Check if file has already been processed
            if os.path.exists(output_path):
                print(f"Skipping {file}, already processed.")
                continue
            
            print(f"Processing {file}...")
            df = load_csv_pointcloud(input_path)
            df = voxel_downsample(df, voxel_size=voxel_size)
            
            if remove_outliers_flag:
                df = remove_outliers(df)
            
            df.to_csv(output_path, sep=";", index=False)
            print(f"Saved processed file to {output_path}")

def main():
    input_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_wall_added\clean\object_filtered_csv_files"
    output_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_wall_added\clean\preprocessed_csv_files"

    # input_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\object_filtered_csv_files" # change this per run
    # output_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\wall_removed\occluded\preprocessed_csv_files" # change this per run

    # input_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\object_filtered_csv_files"
    # output_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\preprocessed_csv_files"
    preprocess_pointcloud_folder(input_folder, output_folder, voxel_size=0.1, remove_outliers_flag=False)

if __name__ == "__main__":
    main()
