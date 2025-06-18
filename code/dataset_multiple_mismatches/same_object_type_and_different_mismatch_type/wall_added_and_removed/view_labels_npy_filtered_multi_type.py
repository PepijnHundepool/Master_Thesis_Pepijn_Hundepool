import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# === Replace with your actual paths ===
points_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_different_mismatch_type\pillar_added_and_removed\occluded\npy_files_filtered_no_augmented\room_1_occluded_robot_scan_pillar_added_removed_frames_17_to_17_object_filtered_preprocessed.npy"
labels_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_different_mismatch_type\pillar_added_and_removed\occluded\labels_npy_filtered_no_augmented\room_1_occluded_robot_scan_pillar_added_removed_frames_17_to_17_object_filtered_preprocessed_labels.npy"

# Load data
points = np.load(points_path)
labels = np.load(labels_path)

assert points.shape[0] == labels.shape[0], "Mismatch between points and labels"

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

mismatch_points = points[labels == 1]
normal_points = points[labels == 0]

ax.scatter(normal_points[:, 0], normal_points[:, 1], normal_points[:, 2], c='gray', s=1, label='Normal')
ax.scatter(mismatch_points[:, 0], mismatch_points[:, 1], mismatch_points[:, 2], c='red', s=5, label='Mismatch')

# Dynamic title based on scan type
scan_type = "Robot-scan" if "robot_scan" in os.path.basename(points_path).lower() else "DT-scan"
ax.set_title(f"Filtered NPY Label Visualization ({scan_type})")
ax.legend()
plt.tight_layout()
plt.show()
