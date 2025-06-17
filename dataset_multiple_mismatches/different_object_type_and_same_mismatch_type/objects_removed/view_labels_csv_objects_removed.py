import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Replace with your actual path ===
# csv_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\labels_csv_filtered\room_test_1_occluded_dt_scan_pillar_removed_frames_1_to_1_object_filtered_preprocessed_labels.csv"  # Change this per run
# csv_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\labels_csv_filtered_improved\room_1_occluded_dt_scan_frames_17_to_17_object_filtered_preprocessed_labels.csv"
csv_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_same_mismatch_type\objects_removed\occluded\labels_csv_filtered\room_1_occluded_dt_scan_pillar_and_wall_removed_frames_17_to_17_object_filtered_preprocessed_labels.csv"

# Load CSV
df = pd.read_csv(csv_path, sep=';')  # assumes columns are semicolon-separated: X;Y;Z;categoryID;label

# Check required columns
if not {"X", "Y", "Z", "label"}.issubset(df.columns):
    raise ValueError("CSV file must contain columns: X, Y, Z, label")

# Extract points and labels
points = df[["X", "Y", "Z"]].values
labels = df["label"].values

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

mismatch_points = points[labels == 1]
normal_points = points[labels == 0]

ax.scatter(normal_points[:, 0], normal_points[:, 1], normal_points[:, 2], c='gray', s=1, label='Normal')
ax.scatter(mismatch_points[:, 0], mismatch_points[:, 1], mismatch_points[:, 2], c='red', s=5, label='Mismatch')

ax.set_title("CSV Label Visualization")
ax.legend()
plt.tight_layout()
plt.show()
