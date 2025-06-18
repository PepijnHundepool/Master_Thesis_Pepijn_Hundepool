from pathlib import Path
import pandas as pd
import numpy as np

# Define input and output paths
# csv_folder = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\labels_csv_filtered") # change this per run
# npy_folder_xyz = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\npy_files_filtered") # change this per run
# npy_folder_labels = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\labels_npy_filtered") # change this per run

# csv_folder = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\labels_csv_filtered_improved") # change this per run
# npy_folder_xyz = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\npy_files_filtered") # change this per run
# npy_folder_labels = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\labels_npy_filtered") # change this per run

csv_folder = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\walls_removed\occluded\labels_csv_filtered")
npy_folder_xyz = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\walls_removed\occluded\npy_files_filtered_no_augmented") # change this per run
npy_folder_labels = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\walls_removed\occluded\labels_npy_filtered_no_augmented") # change this per run

npy_folder_xyz.mkdir(parents=True, exist_ok=True)
npy_folder_labels.mkdir(parents=True, exist_ok=True)

# Process each CSV file
for csv_file in csv_folder.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file, sep=";")
        xyz = df[["X", "Y", "Z"]].values.astype(np.float32)
        labels = df["label"].values.astype(np.float32)

        # Save XYZ
        out_xyz_path = npy_folder_xyz / csv_file.name.replace("_labels", "").replace(".csv", ".npy")
        np.save(out_xyz_path, xyz)

        # Save labels
        label_filename = csv_file.name.replace("_labels", "").replace(".csv", "_labels.npy")
        out_label_path = npy_folder_labels / label_filename
        np.save(out_label_path, labels)

    except Exception as e:
        print(f"[ERROR] Failed to process {csv_file.name}: {e}")

"Done splitting and saving XYZ + labels as separate .npy files."
