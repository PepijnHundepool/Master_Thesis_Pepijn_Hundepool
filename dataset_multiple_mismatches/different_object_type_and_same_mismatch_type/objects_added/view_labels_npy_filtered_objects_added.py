import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Replace with your actual paths ===
# points_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\occluded\npy_files_filtered\room_1_occluded_robot_scan_frames_17_to_17_object_filtered_preprocessed.npy"
# labels_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\occluded\labels_npy_filtered\room_1_occluded_robot_scan_frames_17_to_17_object_filtered_preprocessed_labels.npy"

# points_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\npy_files_filtered_no_augmented\room_test_1_occluded_robot_scan_pillar_added_frames_1_to_1_object_filtered_preprocessed.npy"
# labels_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\labels_npy_filtered_no_augmented\room_test_1_occluded_robot_scan_pillar_added_frames_1_to_1_object_filtered_preprocessed_labels.npy"

points_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_same_mismatch_type\objects_added\occluded\npy_files_filtered_no_augmented\room_1_occluded_robot_scan_pillar_and_wall_added_frames_17_to_17_object_filtered_preprocessed.npy"
labels_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\different_object_type_and_same_mismatch_type\objects_added\occluded\labels_npy_filtered_no_augmented\room_1_occluded_robot_scan_pillar_and_wall_added_frames_17_to_17_object_filtered_preprocessed_labels.npy"

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
