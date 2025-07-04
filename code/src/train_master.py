import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import open3d as o3d
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

TRAIN_MODEL = False  # change per run
threshold = 0.6

class PointCloudDataset(Dataset):
    def __init__(self, npy_folder, labels_folder, exclude_files=None):
        self.npy_folder = npy_folder
        self.labels_folder = labels_folder
        self.files = [
            f for f in os.listdir(npy_folder)
            if f.endswith(".npy") and "_frames_" in f and "_labels" not in f and (exclude_files is None or f not in exclude_files)
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        scan_file = self.files[idx]
        dt_scan_file = scan_file.replace("robot_scan", "dt_scan")

        dt_scan = np.load(os.path.join(self.npy_folder, dt_scan_file))
        robot_scan = np.load(os.path.join(self.npy_folder, scan_file))

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dt_scan)
        distances, _ = nbrs.kneighbors(robot_scan)
        features = distances.astype(np.float32) # change to uncommented for only NN-distance as feature
        # features = np.concatenate([robot_scan, distances], axis=1).astype(np.float32)  # Shape: (N, 4) # change to uncommented for XYZ, NN features

        label_file = scan_file.replace(".npy", "_labels.npy")
        label_path = os.path.join(self.labels_folder, label_file)
        labels = np.load(label_path)

        # Sanity check (only print once per label file)
        if not hasattr(self, "checked_labels"):
            self.checked_labels = set()
        if label_path not in self.checked_labels:
            # print(f"[DEBUG] {label_path} → Unique labels: {np.unique(labels)}")
            self.checked_labels.add(label_path)

        return (
            torch.tensor(robot_scan, dtype=torch.float32),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            scan_file
        )

class MismatchDetectionNet(nn.Module):
    def __init__(self, input_dim=4, dropout_rate=0.3):
        super(MismatchDetectionNet, self).__init__()
        # self.conv1 = nn.Conv1d(1, 64, 1) # change to uncommented for single feature
        self.conv1 = nn.Conv1d(input_dim, 64, 1) # change to uncommented for four features
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.seg_conv1 = nn.Conv1d(1024, 256, 1)
        self.seg_conv2 = nn.Conv1d(256, 128, 1)
        self.seg_conv3 = nn.Conv1d(128, 1, 1)

        self.seg_bn1 = nn.BatchNorm1d(256)
        self.seg_bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        """
        x shape: [B, N, 4] → 4 features per point (XYZ + distance)
        """
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)  # [B, 4, N]

        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, N]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 512, N]
        x = self.dropout(x)

        # Global feature
        global_feat = torch.max(x, 2, keepdim=True)[0]  # [B, 512, 1]

        # Concatenate global with local features
        x_concat = torch.cat([x, global_feat.expand(-1, -1, N)], dim=1)  # [B, 1024, N]

        # Segmentation head
        x = F.relu(self.seg_bn1(self.seg_conv1(x_concat)))  # [B, 256, N]
        x = self.dropout(x)
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))         # [B, 128, N]
        x = self.dropout(x)
        x = self.seg_conv3(x)                               # [B, 1, N]

        return x.squeeze(1)  # [B, N]
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

def get_bounding_box(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    aabb = pcd.get_axis_aligned_bounding_box()
    return pcd, aabb

def custom_collate_fn(batch):
    robot_scans, features, labels, filenames = zip(*batch)
    max_size = max(scan.shape[0] for scan in robot_scans)

    def pad(tensor, target_size):
        pad_size = target_size - tensor.shape[0]
        if pad_size > 0:
            padding = torch.zeros((pad_size, tensor.shape[1]), dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)
        return tensor[:target_size]

    robot_scans_padded = torch.stack([pad(r, max_size) for r in robot_scans])
    features_padded = torch.stack([pad(f, max_size) for f in features])
    labels_padded = torch.stack([pad(l.unsqueeze(1), max_size).squeeze(1) for l in labels])

    return robot_scans_padded, features_padded, labels_padded, filenames

def show_colored_pointcloud_with_colorbar(points, probabilities):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Convert probabilities to NumPy array on CPU
    probabilities = probabilities.detach().cpu().numpy()
    colors = plt.get_cmap("hot")(probabilities)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create dummy image for colorbar legend
    fig, ax = plt.subplots(figsize=(1.5, 6))
    cmap = plt.cm.hot
    norm = plt.Normalize(vmin=0, vmax=1)
    fig.subplots_adjust(right=0.5)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cb.set_label("Mismatch Probability")
    plt.show()

    return pcd  # Important: return the actual colored point cloud for visualization

def compute_metrics(preds, labels):
    preds = preds.cpu().numpy().astype(bool)
    labels = labels.cpu().numpy().astype(bool)

    TP = np.logical_and(preds, labels).sum()
    FP = np.logical_and(preds, np.logical_not(labels)).sum()
    FN = np.logical_and(np.logical_not(preds), labels).sum()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    return TP, FP, FN, precision, recall, iou

npy_folder = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_pillar_removed\clean_and_occluded\npy_files_filtered_no_augmented" # change this per run
labels_folder = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_pillar_removed\clean_and_occluded\labels_npy_filtered_no_augmented" # change this per run

# Reserve specific files for testing only
test_files = [f for f in os.listdir(npy_folder) if f.startswith("room_test") and f.endswith(".npy")]

# Build datasets
full_dataset = PointCloudDataset(npy_folder, labels_folder, exclude_files=test_files)
test_dataset = PointCloudDataset(npy_folder, labels_folder, exclude_files=None)
test_dataset.files = test_files  # Only include your manually specified test files

# Split full dataset into train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

# Check label balance in the training set
total_ones = 0
total_zeros = 0
for _, _, labels, _ in train_loader:
    total_ones += (labels == 1).sum().item()
    total_zeros += (labels == 0).sum().item()
    if (labels == 1).sum().item() == 0:
        print("[WARNING] A batch contains no mismatched labels.")
print(f"Total mismatch points (1): {total_ones}, normal points (0): {total_zeros}")
ratio = total_ones / (total_zeros + 1e-8)
print(f"Ratio mismatch/normal: {ratio:.4f}")

# Visualize class distribution
all_labels = []
for _, _, labels, _ in train_loader:
    all_labels.extend(labels.cpu().numpy().flatten())

# plt.hist(all_labels, bins=2, color='orange')
# plt.xticks([0, 1])
# plt.title("Label Distribution")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MismatchDetectionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)

loss_function = FocalLoss(alpha=10.0, gamma=2).to(device) # "Mismatch points are 10x more important than normal ones."

if TRAIN_MODEL:
    train_losses = []
    val_losses = []
    val_precisions = []  # Store validation precision per epoch
    val_recalls = []     # Store validation recall per epoch
    total_start = time.time()
    epochs = 300
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        for robot_scan, features, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            # inputs = torch.cat((robot_scan, features), dim=2).to(device) # We do not need XYZ
            inputs = features.to(device)
            labels = labels.to(device)
            logits = model(inputs).squeeze(-1)  # [B, N]
            probs = torch.sigmoid(logits)       # Apply sigmoid explicitly here

            # Important: ensure labels are also float and same shape
            labels = labels.float()

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_TP, val_FP, val_FN = 0, 0, 0
        with torch.no_grad():
            for robot_scan, features, labels, _ in val_loader:
                # inputs = torch.cat((robot_scan, features), dim=2).to(device) # We do not need XYZ
                inputs = features.to(device)
                labels = labels.to(device)
                predictions = model(inputs).squeeze(-1)
                loss = loss_function(predictions, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(predictions)
                mismatch_mask = (probs > threshold)

                TP, FP, FN, _, _, _ = compute_metrics(mismatch_mask, labels)
                val_TP += TP
                val_FP += FP
                val_FN += FN

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Compute validation precision and recall
        val_precision = val_TP / (val_TP + val_FP + 1e-8) # how many of the predicted mismatches were actually correct
        val_recall = val_TP / (val_TP + val_FN + 1e-8) # how many of the actual mismatches were detected
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
    
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Time: {epoch_time:.2f} sec")
        
        # Save model every 50 epochs
        if (epoch + 1) % 50 == 0:
            model_save_path = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed\mismatch_detection_model_per_point_pillar_removed_clean_and_occluded_1_epoch_{}.pth".format(epoch + 1) # change this per run
            torch.save(model.state_dict(), model_save_path)
            print(f"[CHECKPOINT] Saved model to {model_save_path}")

            # Save logs up to current epoch
            log_dir = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed"
            np.save(os.path.join(log_dir, "train_losses_epoch_{}.npy".format(epoch + 1)), np.array(train_losses))
            np.save(os.path.join(log_dir, "val_losses_epoch_{}.npy".format(epoch + 1)), np.array(val_losses))
            np.save(os.path.join(log_dir, "val_precisions_epoch_{}.npy".format(epoch + 1)), np.array(val_precisions))
            np.save(os.path.join(log_dir, "val_recalls_epoch_{}.npy".format(epoch + 1)), np.array(val_recalls))
            print("[LOG] Saved training logs at epoch {}".format(epoch + 1))

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), val_precisions, label='Validation Precision', marker='o')
    plt.plot(range(1, epochs + 1), val_recalls, label='Validation Recall', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Precision and Recall per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed\mismatch_detection_model_per_point_pillar_removed_clean_and_occluded_1.pth") # change this per run
    np.save(r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed\train_losses_clean_and_occluded_1.npy", np.array(train_losses)) # change this per run
    np.save(r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed\val_losses_clean_and_occluded_1.npy", np.array(val_losses)) # change this per run
    np.save(r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed\val_precisions_clean_and_occluded_1.npy", np.array(val_precisions)) # change this per run
    np.save(r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed\val_recalls_clean_and_occluded_1.npy", np.array(val_recalls)) # change this per run

