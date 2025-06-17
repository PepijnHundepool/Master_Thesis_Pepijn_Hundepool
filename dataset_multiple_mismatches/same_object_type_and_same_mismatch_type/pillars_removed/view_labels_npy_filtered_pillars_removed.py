import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Replace with your actual paths ===
# points_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\npy_files_filtered\room_test_1_occluded_dt_scan_pillar_removed_frames_1_to_1_object_filtered_preprocessed.npy" # change per run
# labels_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_removed\occluded\labels_npy_filtered\room_test_1_occluded_dt_scan_pillar_removed_frames_1_to_1_object_filtered_preprocessed_labels.npy" # change per run

# points_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\npy_files_filtered\room_test_2_occluded_dt_scan_frames_17_to_17_object_filtered_preprocessed.npy" # change per run
# labels_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_removed\occluded\labels_npy_filtered\room_test_2_occluded_dt_scan_frames_17_to_17_object_filtered_preprocessed_labels.npy" # change per run

points_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\pillars_removed\occluded\npy_files_filtered_no_augmented\room_1_occluded_dt_scan_pillars_removed_frames_17_to_17_object_filtered_preprocessed.npy"
labels_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\pillars_removed\occluded\labels_npy_filtered_no_augmented\room_1_occluded_dt_scan_pillars_removed_frames_17_to_17_object_filtered_preprocessed_labels.npy"

# Load data
points = np.load(points_path)  # (N, 3)
labels = np.load(labels_path)  # (N,)

# Validate dimensions
assert points.shape[0] == labels.shape[0], "Mismatch between points and labels"

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

mismatch_points = points[labels == 1]
normal_points = points[labels == 0]

ax.scatter(normal_points[:, 0], normal_points[:, 1], normal_points[:, 2], c='gray', s=1, label='Normal')
ax.scatter(mismatch_points[:, 0], mismatch_points[:, 1], mismatch_points[:, 2], c='red', s=5, label='Mismatch')

ax.set_title("Filtered NPY Label Visualization")
ax.legend()
plt.tight_layout()
plt.show()
