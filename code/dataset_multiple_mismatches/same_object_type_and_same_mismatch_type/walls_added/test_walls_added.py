import os
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from pyransac3d import Cylinder
from sklearn.metrics import precision_recall_fscore_support
from train_walls_added_NoXYZ import ( # change this per run
    PointCloudDataset,
    MismatchDetectionNet,
    custom_collate_fn,
    compute_metrics,
)

def infer_object_type(cluster):
    cluster_xy = cluster[:, :2] - np.mean(cluster[:, :2], axis=0)
    pca = PCA(n_components=2)
    pca.fit(cluster_xy)
    eig_vals = pca.explained_variance_
    ratio = eig_vals[0] / eig_vals[1]

    print(f"[DEBUG] → PCA XY eigenvalue ratio: {ratio:.2f}")

    # Pillar shape override: if very few points and large curvature
    if ratio > 5 and len(cluster) < 600:
        print("[DEBUG]→ Ratio high but small and likely curved: forcing type to 'cylinder'")
        return "cylinder"

    if ratio < 2.0:
        return "cylinder"
    elif ratio > 5.0:
        return "wall"
    else:
        return "box"

def fit_shape_primitive(cluster, robot_points, obj_type):
    z_values = robot_points[:, 2]
    floor_z = np.percentile(z_values, 1)
    ceiling_z = np.percentile(z_values, 99)

    if obj_type == "cylinder":
        cyl = Cylinder()
        axis, center, radius, inliers = cyl.fit(cluster, thresh=0.4)
        if axis is not None and center is not None and len(axis) == 3 and len(center) == 3 and not np.isnan(radius):
            height = ceiling_z - floor_z
            # Use the cluster's centroid as cylinder center in XY
            cluster_centroid = np.mean(cluster, axis=0)
            corrected_center = np.array([
                cluster_centroid[0],
                cluster_centroid[1],
                (floor_z + ceiling_z) / 2.0
            ])

            # Visual confirmation
            print("[DEBUG]→ Fitted Cylinder: Center =", center, ", Radius =", radius, ", Height =", height)
            print("[DEBUG]→ Corrected Center:", corrected_center)

            # Create solid cylinder mesh
            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=30)
            mesh_cylinder.compute_vertex_normals()

            # Transform mesh to the correct position
            T = np.eye(4)
            T[:3, 3] = corrected_center
            mesh_cylinder.transform(T)

            # Convert mesh to wireframe (line set)
            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_cylinder)
            line_set.paint_uniform_color([0, 1, 0])  # green wireframe

            # # Create Open3D cylinder
            # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=30)
            # mesh_cylinder.compute_vertex_normals()
            # mesh_cylinder.paint_uniform_color([0, 1, 0])
            # T = np.eye(4)
            # T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0))
            # T[:3, 3] = corrected_center
            # mesh_cylinder.transform(T)

            return {
                "type": "pillar",
                "shape": "cylinder",
                "position": center.tolist(),
                "radius": float(radius),
                "height": float(height),
                "orientation": axis.tolist(),
                # "geometry": mesh_cylinder
                "geometry": line_set  # <-- wireframe instead of solid
            }
    else:
        cluster_xy = cluster[:, :2]
        center_xy = np.mean(cluster_xy, axis=0)
        pca = PCA(n_components=2)
        pca.fit(cluster_xy)
        axes_2d = pca.components_
        projections = (cluster_xy - center_xy) @ axes_2d.T
        min_proj = projections.min(axis=0)
        max_proj = projections.max(axis=0)
        extent_xy = max_proj - min_proj
        # Only enforce square shape for square pillars
        if obj_type == "box":
            max_range = np.max(extent_xy) / 2
            extent_xy = np.array([2 * max_range, 2 * max_range])

        center_world_xy = center_xy

        bbox_center = np.array([center_world_xy[0], center_world_xy[1], (floor_z + ceiling_z) / 2])
        bbox_extent = np.array([extent_xy[0], extent_xy[1], ceiling_z - floor_z])

        axes_3d = np.eye(3)
        axes_3d[0, :2] = axes_2d[0]
        axes_3d[1, :2] = axes_2d[1]
        axes_3d[2, 2] = 1.0

        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = bbox_center
        obb.extent = bbox_extent
        obb.R = axes_3d.T
        # obb.color = (0, 1, 0)

        # Convert to wireframe line set
        obb_wireframe = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        obb_wireframe.paint_uniform_color([0, 1, 0])  # green wireframe

        print("[DEBUG] → Fitted Box: Center =", bbox_center, ", Extent =", bbox_extent)

        return {
            "type": "wall" if obj_type == "wall" else "pillar",
            "shape": "box",
            "center": bbox_center.tolist(),
            "extent": bbox_extent.tolist(),
            "rotation": axes_2d.tolist(),
            "geometry": obb_wireframe
        }

def show_colored_pointcloud_with_colorbar(points, probabilities):
    probabilities = probabilities.astype(np.float32)
    cmap = plt.get_cmap("hot")
    norm = Normalize(vmin=0, vmax=1)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colors = cmap(norm(probabilities))[:, :3]

    # Show colorbar (in matplotlib)
    plt.figure(figsize=(1.5, 6))
    plt.title("Mismatch Probability")
    plt.imshow([[colors[np.argmax(probabilities)]]], aspect='auto', cmap=cmap)
    plt.colorbar(mappable)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# Setup paths and model
# npy_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\clean_and_occluded\npy_files_no_augmented" # change this per run
# labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\clean_and_occluded\labels_no_augmented" # change this per run

# npy_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\npy_files_filtered_no_augmented" # change this per run
# labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_real_world\pillar_added\occluded\labels_npy_filtered_no_augmented" # change this per run

npy_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\walls_added\occluded\npy_files_filtered_no_augmented" # change this per run
labels_folder = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_multiple_mismatches\same_object_type_and_same_mismatch_type\walls_added\occluded\labels_npy_filtered_no_augmented" # change this per run

# test_files = ["room_test_1_occluded_robot_scan_frames_17_to_17_object_filtered_preprocessed.npy"] # change this per run

# test_files = ["room_test_1_occluded_robot_scan_pillar_added_frames_1_to_1_object_filtered_preprocessed.npy"] # change this per run

test_files = ["room_1_occluded_robot_scan_walls_added_frames_17_to_17_object_filtered_preprocessed.npy"] # change this per run

# test_files = [f for f in os.listdir(npy_folder) if f.startswith("room_test") and "robot_scan" in f and f.endswith(".npy")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# threshold = 0.2215
threshold = 0.6

# Load dataset and model
test_dataset = PointCloudDataset(npy_folder, labels_folder)
test_dataset.files = test_files
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model_path = r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\mismatch_detection_model_per_point_pillar_added_clean_and_occluded_1_no_augmented_no_xyz.pth" # change this per run
model = MismatchDetectionNet().to(device)
model.load_state_dict(torch.load(model_path)) 
model.eval()

# Run inference
for robot_scan, features, labels, filenames in test_loader:
    labels = labels.to(device)
    num_gt_mismatches = (labels == 1).sum().item()
    print(f"[INFO] Ground Truth: {num_gt_mismatches} mismatch points in {filenames[0]}")

    inputs = features.to(device)
    with torch.no_grad():
        outputs = model(inputs).squeeze()
        outputs = torch.sigmoid(outputs)

        # Logits histogram BEFORE sigmoid
        plt.hist(outputs.cpu().numpy(), bins=50, color='skyblue')
        plt.title("Logits Before Sigmoid")
        plt.xlabel("Logit Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Probability histogram AFTER sigmoid
        plt.hist(torch.sigmoid(outputs).cpu().numpy(), bins=50, color='purple')
        plt.title("Histogram of Predicted Probabilities")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        predicted_labels = (outputs > threshold).cpu().numpy().astype(np.uint8).squeeze()
        filename = filenames[0].replace(".npy", "_pred.npy")
        save_path = os.path.join(r"D:\Graduation Project\Pointclouds\total\datasets\dataset_pillar_added\clean_and_occluded\predicted_labels", filename) # change this per run
        np.save(save_path, predicted_labels)
        print(f"[INFO] Saved predicted labels to {save_path}")

    robot_points = robot_scan.squeeze().cpu().numpy()
    mismatch_mask = (outputs > threshold)
    TP, FP, FN, precision, recall, iou = compute_metrics(mismatch_mask, labels.squeeze())

    print(f"Evaluation Metrics:\nTP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
    print(f"{filenames[0]} → {mismatch_mask.sum().item()} predicted mismatch points")
    print(f"Min prob: {outputs.min():.4f}, Max prob: {outputs.max():.4f}, Mean prob: {outputs.mean():.4f}")

    # Heatmap visualization with colorbar
    colored_pcd = show_colored_pointcloud_with_colorbar(robot_points, outputs.cpu().numpy())
    vis_objects = [colored_pcd]

    # mismatch_points = robot_points[mismatch_mask.cpu().numpy()]
    gt_labels = labels.squeeze().cpu().numpy()
    predicted_mask = mismatch_mask.cpu().numpy()
    true_positive_mask = np.logical_and(predicted_mask, gt_labels == 1)
    mismatch_points = robot_points[true_positive_mask]

    if len(mismatch_points) > 0:
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(mismatch_points)
        cluster_labels = clustering.labels_
        unique_labels = set(cluster_labels)
        filtered_clusters = 0  # ← Initialize counter before the loop
        total_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # for label in unique_labels:
        #     if label == -1:
        #         continue
        #     cluster = mismatch_points[cluster_labels == label]
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(cluster)
        #     bbox = pcd.get_axis_aligned_bounding_box()
        #     bbox.color = (0, 1, 0)
        #     vis_objects.append(bbox)

        # Estimate floor and ceiling from robot scan (z-axis)
        z_values = robot_points[:, 2]
        floor_z = np.percentile(z_values, 1)   # robust to outliers
        ceiling_z = np.percentile(z_values, 99)

        for label in unique_labels:
            if label == -1:
                continue

            cluster = mismatch_points[cluster_labels == label]
            if len(cluster) < 30:
                continue  # skip very small clusters (likely noise)
            center = np.mean(cluster, axis=0)

            # PCA only in XY-plane
            cluster_xy = cluster[:, :2]  # Extract XY
            print(f"[DEBUG] Processing cluster {label}, {len(cluster)} points")
            obj_type = infer_object_type(cluster)
            print(f"[DEBUG] Inferred object type: {obj_type}")
            shape_result = fit_shape_primitive(cluster, robot_points, obj_type)
            # OPTIONAL: reject shapes with extreme size
            if shape_result["shape"] == "cylinder" and shape_result["radius"] > 5.0:
                continue
            if shape_result["shape"] == "box" and max(shape_result["extent"][:2]) > 20.0:
                continue

            # Passed all filters → keep
            filtered_clusters += 1
            print(f"[INFO] Keeping cluster {label} of type '{obj_type}' with {len(cluster)} points")
            # vis_objects.append(shape_result["geometry"]) # change to commented for not showing bounding box

        print(f"[INFO] Total clusters detected: {total_clusters}")
        print(f"[INFO] Clusters shown after filtering: {filtered_clusters}")
    else:
        print(f"{filenames[0]}: No mismatch predicted.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mismatch Heatmap", width=1280, height=720)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])  # Black background
    for geom in vis_objects:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()

    # ========== NEW: TP/FP/FN/TN visualization ==========
    TP_mask = np.logical_and(predicted_mask == 1, gt_labels == 1)
    FP_mask = np.logical_and(predicted_mask == 1, gt_labels == 0)
    FN_mask = np.logical_and(predicted_mask == 0, gt_labels == 1)
    TN_mask = np.logical_and(predicted_mask == 0, gt_labels == 0)

    colors = np.zeros((len(robot_points), 3))  # default black
    colors[TP_mask] = [0, 1, 0]     # green
    colors[FP_mask] = [1, 0, 0]     # red
    colors[FN_mask] = [0, 0, 1]     # blue
    colors[TN_mask] = [0.5, 0.5, 0.5]  # gray (optional)

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(robot_points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="TP/FP/FN/TN Visualization", width=1280, height=720)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd_colored)
    vis.run()
    vis.destroy_window()

    # Create a legend-like colorbar for TP/FP/FN/TN
    legend_elements = [
        mpatches.Patch(color='green', label='True Positive (TP)'),
        mpatches.Patch(color='red', label='False Positive (FP)'),
        mpatches.Patch(color='blue', label='False Negative (FN)'),
        mpatches.Patch(color='gray', label='True Negative (TN)')
    ]

    plt.figure(figsize=(4, 1))
    plt.legend(handles=legend_elements, loc='center', ncol=4, frameon=False)
    plt.axis('off')
    plt.title("Point Classification Legend")
    plt.tight_layout()
    plt.show()

    # ========== NEW: PR and ROC curve plotting ==========
    # Evaluate across thresholds
    precisions, recalls, thresholds_pr = precision_recall_curve(gt_labels.flatten(), outputs.cpu().numpy())
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # avoid division by zero
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_idx]
    best_f1 = f1_scores[best_idx]
    best_p = precisions[best_idx]
    best_r = recalls[best_idx]

    print(f"[INFO] Best Threshold (Highest F1): {best_threshold:.4f}")
    print(f"[INFO]  → F1 = {best_f1:.4f}, Precision = {best_p:.4f}, Recall = {best_r:.4f}")
    
    fpr, tpr, _ = roc_curve(gt_labels, outputs.cpu().numpy())

    plt.figure()
    plt.plot(recalls, precisions, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc(fpr, tpr):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========== NEW: AUC and FPR@95% Recall ==========

    # Compute AUC
    roc_auc = auc(fpr, tpr)
    print(f"[INFO] ROC AUC Score: {roc_auc:.4f}")

    # precisions, recalls, thresholds_pr = precision_recall_curve(gt_labels, outputs.cpu().numpy())
    # best_threshold = None
    # for i, t in enumerate(thresholds_pr):
    #     p = precisions[i + 1]
    #     r = recalls[i + 1]
    #     if r >= 0.95:
    #         best_threshold = t
    #         break

    # print(f"[INFO] Suggested threshold with Recall≥0.95: {best_threshold:.4f}")

    # FPR at 95% Recall
    target_recall = 0.95
    fpr_at_target = None
    precision_at_target = None
    threshold_at_target = None
    iou_at_target = None

    # Loop backwards to find the highest threshold with Recall ≥ 0.95
    for idx in reversed(range(1, len(recalls))):
        if recalls[idx] >= target_recall:
            threshold_at_target = thresholds_pr[min(idx, len(thresholds_pr) - 1)]
            predicted_at_target = (outputs > threshold_at_target).cpu().numpy().astype(np.uint8)

            TP = np.logical_and(predicted_at_target == 1, gt_labels == 1).sum()
            FP = np.logical_and(predicted_at_target == 1, gt_labels == 0).sum()
            FN = np.logical_and(predicted_at_target == 0, gt_labels == 1).sum()

            precision_at_target = TP / (TP + FP + 1e-8)
            iou_at_target = TP / (TP + FP + FN + 1e-8)
            fpr_at_target = FP / (FP + (gt_labels == 0).sum() + 1e-8)
            break

    if fpr_at_target is not None:
        print(f"[INFO] FPR at 95% Recall: {fpr_at_target:.4f}")
        print(f"[INFO] Threshold @ 95% Recall: {threshold_at_target:.4f}")
        print(f"[INFO] Precision @ 95% Recall: {precision_at_target:.4f}")
        print(f"[INFO] IoU @ 95% Recall: {iou_at_target:.4f}")
    else:
        print("[WARN] Could not find recall ≥ 0.95 in evaluation.")

    # ========== NEW: 3D PR-Threshold Visualization with F1 Coloring ==========
    thresholds_full = np.append(thresholds_pr, thresholds_pr[-1])

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    norm = Normalize(vmin=np.min(f1_scores), vmax=np.max(f1_scores))
    colors = plt.cm.viridis(norm(f1_scores))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(thresholds_full) - 1):
        ax.plot(
            thresholds_full[i:i+2],
            precisions[i:i+2],
            recalls[i:i+2],
            color=colors[i],
            linewidth=2
        )

    mappable = ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array(f1_scores)
    fig.colorbar(mappable, ax=ax, label='F1 Score')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax.set_zlabel('Recall')
    ax.set_title('3D Precision-Recall-Threshold Curve Colored by F1 Score')
    plt.tight_layout()
    plt.show()
    