from pathlib import Path
import pandas as pd
import numpy as np

# === Change these paths per run ===
csv_folder = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv_filtered")

npy_folder_xyz = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\npy_files_filtered_no_augmented")
npy_folder_labels = Path(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_npy_filtered_no_augmented")

npy_folder_xyz.mkdir(parents=True, exist_ok=True)
npy_folder_labels.mkdir(parents=True, exist_ok=True)

# Process each CSV file
for csv_file in csv_folder.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file, sep=";")
        xyz = df[["X", "Y", "Z"]].values.astype(np.float32)
        labels = df["label"].values.astype(np.float32)

        # Save XYZ
        xyz_filename = csv_file.name.replace("_labels", "").replace(".csv", ".npy")
        out_xyz_path = npy_folder_xyz / xyz_filename
        np.save(out_xyz_path, xyz)

        # Save labels
        label_filename = csv_file.name.replace(".csv", ".npy")
        out_label_path = npy_folder_labels / label_filename
        np.save(out_label_path, labels)

        print(f"[DONE] {csv_file.name} → XYZ: {xyz_filename} | Labels: {label_filename}")

    except Exception as e:
        print(f"[ERROR] Failed to process {csv_file.name}: {e}")

print("Done splitting and saving XYZ + labels as separate .npy files.")
