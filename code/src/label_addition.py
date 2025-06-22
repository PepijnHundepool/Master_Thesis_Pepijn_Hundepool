import pandas as pd
from pathlib import Path

def add_labels(file1, file2, output_file):
    """Adds labeled points from two different scans into one complete labeled scan. 
    This is needed for combining filtered pillar labels with filtered wall labels."""
    df1 = pd.read_csv(file1, sep=";")
    df2 = pd.read_csv(file2, sep=";")

    if df1.shape[0] != df2.shape[0]:
        raise ValueError(f"Mismatch in number of points: {file1} has {df1.shape[0]}, {file2} has {df2.shape[0]}")

    # Sanity check: ensure XYZ match
    if not (df1[["X", "Y", "Z"]].round(5).values == df2[["X", "Y", "Z"]].round(5).values).all():
        raise ValueError("XYZ coordinates do not match between the two files.")

    # Add labels
    df1["label"] = df1["label"] + df2["label"]
    df1["label"] = df1["label"].clip(upper=1)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df1.to_csv(output_file, sep=";", index=False)
    print(f"[INFO] Combined labels saved to: {output_file}")

# === MODIFY PER RUN ===
file_pillars = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv_filtered_pillar\room_1_occluded_dt_scan_chaos_frames_17_to_17_object_filtered_preprocessed_labels.csv" # change to desired file
file_walls   = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv_filtered_wall\room_1_occluded_dt_scan_chaos_frames_17_to_17_object_filtered_preprocessed_labels.csv" # change to desired file
output_file  = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_multiple_mismatches\different_object_type_and_different_mismatch_type\objects_added_and_removed\occluded\labels_csv_filtered\room_1_occluded_dt_scan_chaos_frames_17_to_17_object_filtered_preprocessed_labels.csv" # change to desired file

add_labels(file_pillars, file_walls, output_file)
