import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# === Replace with your actual path to either robot or dt labeled scan ===
csv_path = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_pillar_removed\clean\labels_csv\room_1_clean_dt_scan_frames_17_to_17_object_filtered_preprocessed_labels.csv"

# Load CSV
df = pd.read_csv(csv_path, sep=';')

# Check required columns
if not {"X", "Y", "Z", "label"}.issubset(df.columns):
    raise ValueError("CSV file must contain columns: X, Y, Z, label")

# Extract data
points = df[["X", "Y", "Z"]].values
labels = df["label"].values

# Split into mismatch vs normal
mismatch_points = points[labels == 1]
normal_points = points[labels == 0]

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(normal_points[:, 0], normal_points[:, 1], normal_points[:, 2], c='gray', s=1, label='Normal')
ax.scatter(mismatch_points[:, 0], mismatch_points[:, 1], mismatch_points[:, 2], c='red', s=5, label='Mismatch')

scan_type = "Robot-scan" if "robot_scan" in os.path.basename(csv_path) else "DT-scan"
ax.set_title(f"Mismatch Label Visualization ({scan_type})")
ax.legend()
plt.tight_layout()
plt.show()
