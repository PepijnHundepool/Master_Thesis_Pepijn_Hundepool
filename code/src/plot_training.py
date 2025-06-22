import os
import numpy as np
import matplotlib.pyplot as plt

# Set the directory where the .npy files are saved
log_dir = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed" # change per run

# Load the saved arrays
val_losses = np.load(os.path.join(log_dir, "val_losses_clean_and_occluded_3_no_augmented_no_xyz.npy"))
val_precisions = np.load(os.path.join(log_dir, "val_precisions_clean_and_occluded_3_no_augmented_no_xyz.npy"))
val_recalls = np.load(os.path.join(log_dir, "val_recalls_clean_and_occluded_3_no_augmented_no_xyz.npy"))

# Optional: Load train_losses if available
train_loss_path = os.path.join(log_dir, "train_losses_clean_and_occluded_3_no_augmented_no_xyz.npy")
train_losses = np.load(train_loss_path) if os.path.exists(train_loss_path) else None

# Epochs for x-axis
epochs = range(1, len(val_losses) + 1)

# Plot loss curves
plt.figure(figsize=(10, 5))
if train_losses is not None:
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot precision and recall
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_precisions, label='Validation Precision', marker='o')
plt.plot(epochs, val_recalls, label='Validation Recall', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Precision and Recall per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
