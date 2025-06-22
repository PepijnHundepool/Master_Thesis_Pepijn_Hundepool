import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from bounding_box_helpers import (
    fit_pillar_bounding_box,
    is_valid_pillar_bbox,
    fit_wall_bounding_box,
    fit_wall_bounding_box_pca_split,
    is_valid_wall_bbox,
    infer_object_type,
    compute_bbox_iou,
    merge_wall_boxes
)
from train_master import MismatchDetectionNet
import open3d as o3d
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neighbors import NearestNeighbors
from matplotlib import MatplotlibDeprecationWarning
import json
import time
from MD_performance_helpers import evaluate_model_md_metrics
from config import *

# Suppress specific categories of warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

MODEL_PATHS = [
    r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_added\mismatch_detection_model_per_point_pillar_added_clean_and_occluded_1_no_augmented_no_xyz.pth",
    r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\pillar_removed\mismatch_detection_model_per_point_pillar_removed_clean_and_occluded_3_no_augmented_no_xyz.pth",
    r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\wall_added\mismatch_detection_model_per_point_wall_added_clean_and_occluded_1_no_augmented_no_xyz.pth",
    r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\trained_models\wall_removed\mismatch_detection_model_per_point_wall_removed_clean_and_occluded_2_no_augmented_no_xyz.pth",
]

TEST_SCAN_PAIRS = [
    (r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_real_world\pillar_added\clean\npy_files_filtered_no_augmented\room_test_1_clean_dt_scan_pillar_added_frames_1_to_1_object_filtered_preprocessed", # change per run
     r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_real_world\pillar_added\clean\npy_files_filtered_no_augmented\room_test_1_clean_robot_scan_pillar_added_frames_1_to_1_object_filtered_preprocessed") # change per run
]

def load_ground_truth_bbox(test_filename, json_folder):
    base_name = test_filename.replace("_frames_1_to_1_object_filtered_preprocessed.npy", "") # change this per run, "1_to_1" for real-world test cases, "17_to_17" for all other
    json_path = os.path.join(json_folder, base_name + ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] Ground truth JSON not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def extract_model_metadata(path):
    name = os.path.basename(path)
    for mismatch in ["added", "removed"]:
        for obj in ["pillar", "wall"]:
            if obj in name and mismatch in name:
                return mismatch, obj
    return None, None

def run_model(model, features):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        return torch.sigmoid(outputs).numpy()

def compute_f1(predictions, ground_truth):
    preds = (predictions > THRESHOLD).astype(np.uint8)
    return f1_score(ground_truth, preds, zero_division=0)

def generate_bounding_box_without_gt(points, predictions, object_type, threshold=0.5):
    # Apply global Z-trimming before clustering. This is a configurable setting. 
    if ENABLE_Z_TRIMMING:
        z_vals = points[:, 2]
        z_min = np.percentile(z_vals, 1)
        z_max = np.percentile(z_vals, 99)
        z_trim = Z_TRIM_MARGIN  # Configurable floor/ceiling exclusion
        z_mask = (points[:, 2] > z_min + z_trim) & (points[:, 2] < z_max - z_trim)

        # Apply this mask to both points and predictions
        points_trimmed = points[z_mask]
        predictions_trimmed = predictions[z_mask]
    else:
        points_trimmed = points
        predictions_trimmed = predictions
        print("[DEBUG] Z-trimming disabled")

    mismatch_points = points_trimmed[predictions_trimmed > threshold]
    if len(mismatch_points) == 0:
        print("[INFO] No mismatch points above threshold after Z-trimming.")
        return []

    if object_type == "pillar":
        return fit_pillar_bounding_box(mismatch_points, points)
    elif object_type == "wall":
        if USE_WALL_PCA_SPLIT: # Configurable setting
            return fit_wall_bounding_box_pca_split(
                mismatch_points,
                points,
                split_eccentricity_threshold=WALL_SPLIT_ECC_THRESHOLD,
                min_cluster_size=MIN_CLUSTER_SIZE
            )
        else:
            return fit_wall_bounding_box(mismatch_points, points)
    else:
        raise ValueError(f"Unknown object type: {object_type}")

def visualize_outputs(points, predictions, ground_truth, threshold=0.6):
    if len(points) != len(ground_truth):
        print(f"[WARNING] Mismatched lengths: points={len(points)}, labels={len(ground_truth)} — skipping visualization")
        return

    # mismatch_mask = predictions > THRESHOLD
    mismatch_mask = predictions > threshold
    TP_mask = np.logical_and(mismatch_mask == 1, ground_truth == 1)
    FP_mask = np.logical_and(mismatch_mask == 1, ground_truth == 0)
    FN_mask = np.logical_and(mismatch_mask == 0, ground_truth == 1)
    TN_mask = np.logical_and(mismatch_mask == 0, ground_truth == 0)

    colors = np.zeros((len(points), 3))
    colors[TP_mask] = [0, 1, 0]
    colors[FP_mask] = [1, 0, 0]
    colors[FN_mask] = [0, 0, 1]
    colors[TN_mask] = [0.5, 0.5, 0.5]

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd_colored])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_colored)
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.run()
    vis.destroy_window()

    legend = [
        plt.Line2D([0], [0], color='green', lw=4, label='TP'),
        plt.Line2D([0], [0], color='red', lw=4, label='FP'),
        plt.Line2D([0], [0], color='blue', lw=4, label='FN'),
        plt.Line2D([0], [0], color='gray', lw=4, label='TN'),
    ]
    # plt.figure()
    # plt.legend(handles=legend, loc='center', ncol=4)
    # plt.axis('off')
    # plt.title("Classification Legend")
    # plt.tight_layout()
    # plt.show() 

def show_heatmap_pointcloud(points, predictions):
    norm = Normalize(vmin=0, vmax=1)
    # cmap = plt.cm.get_cmap("hot")
    cmap = plt.colormaps["hot"]
    colors = cmap(norm(predictions))[:, :3]

    # Plot the colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    # plt.figure(figsize=(1.5, 6))
    # plt.title("Mismatch Probability")
    # plt.colorbar(mappable)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_colored)
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.run()
    vis.destroy_window()

def plot_pr_roc(outputs, ground_truth):
    precisions, recalls, thresholds_pr = precision_recall_curve(ground_truth, outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    print(f"[INFO] Best threshold: {thresholds_pr[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

    fpr, tpr, _ = roc_curve(ground_truth, outputs)
    plt.figure()
    plt.plot(recalls, precisions, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc(fpr,tpr):.2f})')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()

    thresholds_full = np.append(thresholds_pr, thresholds_pr[-1])
    norm = Normalize(vmin=np.min(f1_scores), vmax=np.max(f1_scores))
    colors = plt.cm.viridis(norm(f1_scores))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(thresholds_full) - 1):
        ax.plot(
            thresholds_full[i:i+2],
            precisions[i:i+2],
            recalls[i:i+2],
            color=colors[i], linewidth=2
        )
    mappable = ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array(f1_scores)
    fig.colorbar(mappable, ax=ax, label='F1 Score')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax.set_zlabel('Recall')
    ax.set_title('PR-Threshold Curve (Colored by F1)')
    plt.tight_layout()
    # plt.show()

def visualize_gt_bounding_boxes_json(json_path, points, labels, window_name="GT Viewer", color=[1, 1, 1]):
    """
    Visualizes a point cloud with heatmap coloring and GT bounding boxes from a JSON file.
    - json_path: path to GT .json file
    - points: np.ndarray of shape (N, 3)
    - labels: np.ndarray of per-point scores (for coloring)
    - window_name: title of the Open3D window
    - color: RGB triplet to color the bounding box edges
    """
    with open(json_path, "r") as f:
        gt_json = json.load(f)

    geometries = []

    # Add point cloud with heatmap coloring
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.colormaps["hot"]
    colors = cmap(norm(labels))[:, :3]

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd_colored)

    # Add GT bounding boxes
    for gt_box in gt_json["objects"]:
        shape = gt_box.get("shape", "rectangular")
        center = gt_box["center"]
        dims = gt_box["dimensions"]
        rotation_deg = gt_box.get("rotation", 0)
        angle_rad = np.deg2rad(rotation_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = center
        obb.extent = dims
        obb.R = R

        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        lineset.paint_uniform_color(color)
        geometries.append(lineset)

    # Show everything
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for g in geometries:
        vis.add_geometry(g)
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().point_size = 2.0
    vis.run()
    vis.destroy_window()

def test_master():
    start_total_time = time.time()
    results = []
    for model_path in MODEL_PATHS:
        start_model_time = time.time()
        input_dim = 1 if "no_xyz" in model_path else 4
        model = MismatchDetectionNet(input_dim=input_dim)
        # model = MismatchDetectionNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        mismatch_type, object_type = extract_model_metadata(model_path)
        print(f"[INFO] Testing model: {mismatch_type} {object_type}")

        for dt_file, robot_file in TEST_SCAN_PAIRS:
            dt_scan = np.load(dt_file)
            robot_scan = np.load(robot_file)
            label_dir = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_real_world\pillar_added\clean\labels_npy_filtered_no_augmented" # change this per run
            dt_label_file = os.path.join(label_dir, os.path.basename(dt_file).replace(".npy", "_labels.npy"))
            robot_label_file = os.path.join(label_dir, os.path.basename(robot_file).replace(".npy", "_labels.npy"))

            labels_dt = np.load(dt_label_file)
            labels_robot = np.load(robot_label_file)

            # Compute features based on distance to opposite scan (to match training setup)
            nbrs_robot = NearestNeighbors(n_neighbors=1).fit(dt_scan)
            robot_distances, _ = nbrs_robot.kneighbors(robot_scan)
            # robot_features = np.concatenate([robot_scan, robot_distances], axis=1)
            if "no_xyz" in model_path:
                robot_features = robot_distances  # shape: (N, 1)
            else:
                robot_features = np.concatenate([robot_scan, robot_distances], axis=1)  # shape: (N, 4)

            nbrs_dt = NearestNeighbors(n_neighbors=1).fit(robot_scan)
            dt_distances, _ = nbrs_dt.kneighbors(dt_scan)
            # dt_features = np.concatenate([dt_scan, dt_distances], axis=1)
            if "no_xyz" in model_path:
                dt_features = dt_distances  # shape: (N, 1)
            else:
                dt_features = np.concatenate([dt_scan, dt_distances], axis=1)  # shape: (N, 4)

            # Determine which scan type to evaluate
            is_robot_trained = mismatch_type == "added"
            is_dt_trained = mismatch_type == "removed"

            if is_dt_trained:
                preds_dt = run_model(model, dt_features)
                f1_dt = compute_f1(preds_dt, labels_dt)
                preds_robot = np.zeros_like(labels_robot, dtype=np.float32)  # skip robot
                f1_robot = 0.0
            elif is_robot_trained:
                preds_robot = run_model(model, robot_features)
                # print(f"[DEBUG] Robot predictions: min={preds_robot.min():.4f}, max={preds_robot.max():.4f}, mean={preds_robot.mean():.4f}")
                f1_robot = compute_f1(preds_robot, labels_robot)
                preds_dt = np.zeros_like(labels_dt, dtype=np.float32)  # skip dt
                f1_dt = 0.0
            else:
                preds_dt = np.zeros_like(labels_dt, dtype=np.float32)
                preds_robot = np.zeros_like(labels_robot, dtype=np.float32)
                f1_dt = f1_robot = 0.0

            # === Mismatch Detection Performance Metrics ===
            if is_robot_trained:
                print(f"[METRICS] Mismatch Detection Evaluation for {mismatch_type.upper()} - {object_type}")
                # evaluate_model_md_metrics(
                metrics = evaluate_model_md_metrics(
                    model=model,
                    points=robot_scan,
                    features=robot_features,
                    labels=labels_robot,
                    mismatch_type=mismatch_type,
                    predictions=preds_robot
                )
                threshold_dynamic = metrics.get("threshold_at_95", 0.5)  # fallback if not found
                print(f"[DEBUG] → Dynamic threshold: {threshold_dynamic:.4f}")
                print(f"[DEBUG] → Prediction stats: min={preds_robot.min():.4f}, max={preds_robot.max():.4f}, mean={preds_robot.mean():.4f}")
            elif is_dt_trained:
                print(f"[METRICS] Mismatch Detection Evaluation for {mismatch_type.upper()} - {object_type}")
                # evaluate_model_md_metrics(
                metrics = evaluate_model_md_metrics(
                    model=model,
                    points=dt_scan,
                    features=dt_features,
                    labels=labels_dt,
                    mismatch_type=mismatch_type,
                    predictions=preds_dt
                )
                threshold_dynamic = metrics.get("threshold_at_95", 0.5)  # fallback if not found
                print(f"[DEBUG] → Dynamic threshold: {threshold_dynamic:.4f}")
                print(f"[DEBUG] → Prediction stats: min={preds_dt.min():.4f}, max={preds_dt.max():.4f}, mean={preds_dt.mean():.4f}")

            # === Optional: Per-scan Visualization ===
            if is_robot_trained:
                print("[VISUALIZE] Robot-scan predictions")
                # visualize_outputs(robot_scan, preds_robot, labels_robot, threshold=threshold_dynamic) 
                visualize_outputs(robot_scan, preds_robot, labels_robot, threshold=THRESHOLD)
            elif is_dt_trained:
                print("[VISUALIZE] DT-scan predictions")
                # visualize_outputs(dt_scan, preds_dt, labels_dt, threshold=threshold_dynamic)
                visualize_outputs(dt_scan, preds_dt, labels_dt, threshold=THRESHOLD)

            # # Plot pr and roc curves
            # if is_robot_trained:
            #     plot_pr_roc(preds_robot, labels_robot)
            # elif is_dt_trained:
            #     plot_pr_roc(preds_dt, labels_dt)

            # Compute bounding box score only on correct scan
            if is_robot_trained:
                predicted_bboxes = generate_bounding_box_without_gt(
                    points=robot_scan,
                    predictions=preds_robot,
                    object_type=object_type,
                    threshold=THRESHOLD
                    # threshold=threshold_dynamic # This line is cheating
                )
                if object_type == "pillar":
                    valid_bboxes = [box for box in predicted_bboxes if is_valid_pillar_bbox(box)]
                elif object_type == "wall":
                    valid_bboxes = [box for box in predicted_bboxes if is_valid_wall_bbox(box)]
                    if MERGE_WALL_BOXES:
                        valid_bboxes = merge_wall_boxes(valid_bboxes) # Configurable setting
                else:
                    print(f"[DEBUG] → Skipped: failed bbox validation.")
                    valid_bboxes = []

                bbox_score = 1.0 if len(valid_bboxes) > 0 else 0.0
                final_bbox = valid_bboxes
            elif is_dt_trained:
                predicted_bboxes = generate_bounding_box_without_gt(
                    points=dt_scan,
                    predictions=preds_dt,
                    object_type=object_type,
                    threshold=THRESHOLD
                    # threshold=threshold_dynamic # This line is cheating
                )
                if object_type == "pillar":
                    valid_bboxes = [box for box in predicted_bboxes if is_valid_pillar_bbox(box)]
                elif object_type == "wall":
                    valid_bboxes = [box for box in predicted_bboxes if is_valid_wall_bbox(box)]
                    if MERGE_WALL_BOXES:
                        valid_bboxes = merge_wall_boxes(valid_bboxes) # Configurable setting
                else:
                    print(f"[DEBUG] → Skipped: failed bbox validation.")
                    valid_bboxes = []

                bbox_score = 1.0 if len(valid_bboxes) > 0 else 0.0
                final_bbox = valid_bboxes
            else:
                bbox_score = 0.0
                final_bbox = None

            results.append((model_path, f1_dt, f1_robot, bbox_score,
                preds_dt, preds_robot, dt_scan, robot_scan,
                labels_dt, labels_robot, final_bbox, mismatch_type, object_type, None))
            end_model_time = time.time()
            print(f"[TIME] Inference for model {os.path.basename(model_path)} took {end_model_time - start_model_time:.2f} seconds")
            
    print("\n[INFO] Per-model scores:")
    for res in results:
        print(f"{res[0]} | F1_DT={res[1]:.4f} | F1_Robot={res[2]:.4f} | Predicted Box Score={res[3]:.4f}")

    selected_results = [r for r in results if r[3] > 0.5]  # bbox_score
    
    print("\n========================")
    print("\n[INFO] Selected models (score ≥ threshold):")
    final_boxes = []
    added_centers = []

    for res in selected_results:
        model_name = os.path.basename(res[0])
        bboxes = res[10] if isinstance(res[10], list) else [res[10]]
        for bbox in bboxes:
            if bbox is None or "center" not in bbox:
                continue
            center = np.array(bbox["center"])
            # final_boxes.append((model_name, bbox)) # change this line to commented and uncomment portion below to enable duplicate object checking
            is_duplicate = any(np.linalg.norm(center - c) < DUPLICATE_CENTER_DISTANCE_THRESHOLD for c in added_centers)
            if not is_duplicate:
                final_boxes.append((model_name, bbox))
                added_centers.append(center)
            else:
                print(f"[DEBUG] Skipping duplicate bbox at {np.round(center, 2)}")

    print(f"\n[SUMMARY] Models selected: {len(selected_results)}")
    print(f"[SUMMARY] Bounding boxes visualized: {len(final_boxes)}")

    print("\n[INFO] Showing predicted bounding boxes in two separate Open3D windows...")

    added_geometries = []
    removed_geometries = []

    for model_name, bbox in final_boxes:
        if bbox is None or "center" not in bbox or "bbox" not in bbox:
            continue

        match = [r for r in results if os.path.basename(r[0]) == model_name]
        if not match:
            continue
        res = match[0]
        preds_dt = res[4]
        preds_robot = res[5]
        dt_points = res[6]
        robot_points = res[7]

        mismatch_type, object_type = extract_model_metadata(model_name)

        # print(f"[FINAL_OUTPUT] A {bbox['shape']} {bbox['type']} has been {mismatch_type} at {np.round(bbox['center'], 2)} with dimensions {np.round(bbox['dimensions'], 2)}")
        print(f"[FINAL_OUTPUT] A {bbox['type']} has been {mismatch_type} at {np.round(bbox['center'], 2)} with dimensions {np.round(bbox['dimensions'], 2)}")
        
        if mismatch_type == "added":
            scan_points = robot_points
            scan_preds = preds_robot
            target_geometries = added_geometries
            color = [0, 1, 0]  # green
        elif mismatch_type == "removed":
            scan_points = dt_points
            scan_preds = preds_dt
            target_geometries = removed_geometries
            color = [0, 0.5, 1]  # blue
        else:
            continue

        # Heatmap point cloud
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.colormaps["hot"]
        colors = cmap(norm(scan_preds))[:, :3]

        pcd_colored = o3d.geometry.PointCloud()
        pcd_colored.points = o3d.utility.Vector3dVector(scan_points)
        pcd_colored.colors = o3d.utility.Vector3dVector(colors)
        target_geometries.append(pcd_colored)

        # Bounding box lines
        obb_geom = bbox["bbox"]
        if isinstance(obb_geom, o3d.geometry.OrientedBoundingBox):
            lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb_geom)
        else:
            lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(obb_geom)

        lineset.paint_uniform_color(color)
        target_geometries.append(lineset)

    # Show "added" predictions
    if added_geometries:
        vis_added = o3d.visualization.Visualizer()
        vis_added.create_window(window_name="Predicted Bounding Boxes (Added)")
        for g in added_geometries:
            vis_added.add_geometry(g)
        render = vis_added.get_render_option()
        render.background_color = np.array([0, 0, 0])
        render.point_size = 2.0
        vis_added.run()
        vis_added.destroy_window()

    # Show "removed" predictions
    if removed_geometries:
        vis_removed = o3d.visualization.Visualizer()
        vis_removed.create_window(window_name="Predicted Bounding Boxes (Removed)")
        for g in removed_geometries:
            vis_removed.add_geometry(g)
        render = vis_removed.get_render_option()
        render.background_color = np.array([0, 0, 0])
        render.point_size = 2.0
        vis_removed.run()
        vis_removed.destroy_window()

    # === Viewer for GT bounding boxes only ===
    if not final_boxes:
        print("[INFO] Skipping GT bounding box viewer: no predicted boxes available.")
        return  # prevents undefined scan_preds/scan_points error

    bounding_box_dir = r"D:\GitHub\Master_Thesis_Pepijn_Hundepool\datasets\dataset_real_world\pillar_added\clean\bounding_boxes" # change per run
    
    dt_gt_filename = os.path.basename(TEST_SCAN_PAIRS[0][0]).replace("_frames_1_to_1_object_filtered_preprocessed.npy", "") + ".json" # change this per run, "1_to_1" for real-world test cases, "17_to_17" for all other
    robot_gt_filename = os.path.basename(TEST_SCAN_PAIRS[0][1]).replace("_frames_1_to_1_object_filtered_preprocessed.npy", "") + ".json" # change this per run, "1_to_1" for real-world test cases, "17_to_17" for all other

    dt_gt_path = os.path.join(bounding_box_dir, dt_gt_filename)
    robot_gt_path = os.path.join(bounding_box_dir, robot_gt_filename)

    for res in selected_results:
        mismatch_type, _ = extract_model_metadata(res[0])
        dt_points = res[6]
        robot_points = res[7]
        preds_dt = res[4]
        preds_robot = res[5]

        if mismatch_type == "removed" and os.path.exists(dt_gt_path):
            print(f"[INFO] Showing DT-scan ground truth bounding boxes from {dt_gt_filename}")
            visualize_gt_bounding_boxes_json(dt_gt_path, dt_points, preds_dt, window_name="GT DT-scan")

        if mismatch_type == "added" and os.path.exists(robot_gt_path):
            print(f"[INFO] Showing Robot-scan ground truth bounding boxes from {robot_gt_filename}")
            visualize_gt_bounding_boxes_json(robot_gt_path, robot_points, preds_robot, window_name="GT Robot-scan")

    if not os.path.exists(dt_gt_path) and not os.path.exists(robot_gt_path):
        print("[INFO] Skipping GT bounding box viewer: no ground truth JSONs found.")

    end_total_time = time.time()
    total_runtime = end_total_time - start_total_time
    print(f"\n[INFO] Total runtime for test_master(): {total_runtime:.2f} seconds")

if __name__ == "__main__":
    test_master()
