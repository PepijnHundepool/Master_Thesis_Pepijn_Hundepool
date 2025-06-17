import numpy as np
import open3d as o3d
from pyransac3d import Cylinder, Plane
from sklearn.decomposition import PCA
from shapely.geometry import box
from scipy.spatial import ConvexHull

def infer_object_type(cluster):
    cluster_xy = cluster[:, :2] - np.mean(cluster[:, :2], axis=0)
    pca = PCA(n_components=2)
    pca.fit(cluster_xy)
    eig_vals = pca.explained_variance_
    ratio = eig_vals[0] / eig_vals[1]

    rotation_angle = np.degrees(np.arctan2(pca.components_[0][1], pca.components_[0][0]))
    rotation_offset = rotation_angle % 90
    alignment_error = min(rotation_offset, 90 - rotation_offset)

    # Project to PCA axes
    proj_x = cluster_xy @ pca.components_[0]
    proj_y = cluster_xy @ pca.components_[1]
    length = np.ptp(proj_x)
    width = np.ptp(proj_y)

    # Ensure width is always the shorter axis
    shorter, longer = sorted([width, length])
    radial_distances = np.linalg.norm(cluster_xy, axis=1)
    radial_std = np.std(radial_distances)

    # Log info
    print(f"[DEBUG] → PCA XY eigenvalue ratio: {ratio:.2f}")
    print(f"[DEBUG] → Width: {shorter:.2f} m, Length: {longer:.2f} m")
    print(f"[DEBUG] → Rotation_angle: {rotation_angle} degrees")
    print(f"[DEBUG] → Radial std: {radial_std:.3f} m")

    is_wall_thin = shorter < 0.25
    is_wall_ratio = ratio > 10.0
    is_axis_aligned = alignment_error < 5.0  # tolerance in degrees

    if is_wall_thin and is_wall_ratio and is_axis_aligned:
        print(f"[DEBUG] → Inferred object type is wall_object")
        return "wall_object"

    elif ratio < 10:
        if 0.1 < radial_std < 0.2:
            print(f"[DEBUG] → Inferred object type is cylinder_pillar (low radial std)")
            return "cylinder_pillar"
        else:
            print(f"[DEBUG] → Inferred object type is rectangular_pillar (high radial std)")
            return "rectangular_pillar"

    print(f"[DEBUG] → Ambiguous: defaulting to rectangular_pillar")
    return "rectangular_pillar"

    # is_square = abs(length - width) < 0.2
    # is_wall_thickness = shorter < 0.25
    # is_wall_ratio = ratio > 10.0
    
    # if is_wall_thickness and is_wall_ratio and not is_square:
    #     print(f"[DEBUG] → Inferred object type is wall_object")
    #     return "wall_object"
    # elif is_square:
    #     print(f"[DEBUG] → Inferred object type is cylinder_pillar or rectangular_pillar")
    #     return "cylinder_pillar" if ratio < 2.0 else "rectangular_pillar"
    # else:
    #     print(f"[DEBUG] → Inferred object type is cylinder_pillar or rectangular_pillar")
    #     return "cylinder_pillar" if ratio < 2.0 else "rectangular_pillar"

def fit_pillar_bounding_box(mismatch_points, points):
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    import open3d as o3d
    import numpy as np

    clustering = DBSCAN(eps=0.2, min_samples=10).fit(mismatch_points)
    labels = clustering.labels_
    unique_labels = set(labels)
    z_vals = points[:, 2]
    floor_z = np.percentile(z_vals, 1)
    ceil_z = np.percentile(z_vals, 99)

    fitted_boxes = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster = mismatch_points[labels == label]

        # # Remove floor/ceiling outliers entirely. This is a configurable setting. It must be disabled when global trimming is enabled. 
        # z_valid_mask = np.logical_and(cluster[:, 2] > floor_z + 0.02, cluster[:, 2] < ceil_z - 0.02)
        # cluster = cluster[z_valid_mask]

        # Reject clusters with too many points on floor or ceiling (these are shadow artifacts). This is a configurable setting. 
        z_cluster = cluster[:, 2]
        z_outliers = np.logical_or(z_cluster <= floor_z, z_cluster >= ceil_z)
        if len(z_outliers) != 0:
            outlier_ratio = np.mean(z_outliers)
        else:
            outlier_ratio = 0
        if outlier_ratio > 0.01:  # more than 5% of points are floor/ceiling artifacts
            print(f"[DEBUG] → Skipping cluster {label}: {outlier_ratio:.1%} points on floor/ceiling")
            continue

        if len(cluster) < 30:
            print(f"[DEBUG] → Skipping cluster {label}: Too few points ({len(cluster)})")
            continue

        vertical_extent = np.ptp(cluster[:, 2])
        if vertical_extent < 0.2:
            print(f"[DEBUG] → Skipping cluster {label}: Vertical extent too small ({vertical_extent:.2f} m)")
            continue

        print(f"[PILLAR_DEBUG] → Cluster {label} with {len(cluster)} points")

        pca = PCA(n_components=2)
        xy = cluster[:, :2]
        pca.fit(xy)
        width = np.ptp(xy @ pca.components_[0])
        length = np.ptp(xy @ pca.components_[1])
        size = max(width, length)

        # Build point cloud and AABB
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster))
        aabb = pcd.get_axis_aligned_bounding_box()
        min_bound = list(aabb.get_min_bound())
        max_bound = list(aabb.get_max_bound())

        # Force full Z span
        min_bound[2] = floor_z
        max_bound[2] = ceil_z

        # Force square footprint
        center_x = (min_bound[0] + max_bound[0]) / 2
        center_y = (min_bound[1] + max_bound[1]) / 2
        # center_z = (min_bound[2] + max_bound[2]) / 2
        center_z = (floor_z + ceil_z) / 2

        # corrected_center_xy = estimate_true_pillar_center(xy)
        # center_x, center_y = corrected_center_xy
        # center_z = (floor_z + ceil_z) / 2

        half_size = size / 2

        min_bound = [center_x - half_size, center_y - half_size, floor_z]
        max_bound = [center_x + half_size, center_y + half_size, ceil_z]

        aabb_final = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        aabb_final.color = (0, 1, 0)

        rotation_angle = np.degrees(np.arctan2(pca.components_[0][1], pca.components_[0][0]))
        
        # Enforce square before inference
        square_cluster = cluster.copy()
        square_cluster[:, 0] = center_x + (square_cluster[:, 0] - center_x) * (half_size / (width / 2))
        square_cluster[:, 1] = center_y + (square_cluster[:, 1] - center_y) * (half_size / (length / 2))
        # object_shape = infer_object_type(square_cluster)
        object_shape = infer_object_type(cluster)

        print(f"[PILLAR_DEBUG] → Inferred shape: {object_shape}")
        print(f"[PILLAR_DEBUG] → Cluster center XY: {center_x:.2f}, {center_y:.2f}, Size: {size:.2f}, Rotation: {rotation_angle:.1f}°")

        if object_shape == "cylinder_pillar":
            shape = "cylinder"
        elif object_shape == "rectangular_pillar":
            shape = "rectangular"
        else:
            print(f"[PILLAR_DEBUG] → Skipped: shape {object_shape} not allowed")
            continue  # skip this cluster, it's not a pillar

        bbox = {
            "type": "pillar",
            "shape": shape,
            "center": [float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1])), float((floor_z + ceil_z) / 2)],
            # "center": [float(center_x), float(center_y), float(center_z)],
            "dimensions": [size, size, ceil_z - floor_z],
            # "dimensions": [size, size, max_bound[2] - min_bound[2]], # uncomment this line for unstretched vertical dimension
            "rotation": float(rotation_angle),
            "bbox": aabb_final  # optional for Open3D vis
        }
        fitted_boxes.append(bbox)

    return fitted_boxes

def estimate_true_pillar_center(cluster_xy):
    center = np.mean(cluster_xy, axis=0)
    shifted = cluster_xy - center

    pca = PCA(n_components=2)
    pca.fit(shifted)
    forward_axis = pca.components_[0]  # major axis of visible half

    projections = shifted @ forward_axis
    mirrored_proj = -projections
    mirrored = np.outer(mirrored_proj, forward_axis)

    full_cluster = np.vstack([shifted, mirrored])
    corrected_center = center + np.mean(full_cluster, axis=0)
    return corrected_center

def split_cluster_along_pca(cluster, n_components=2, axis=0, eps=0.2, min_samples=10):
    """Split a cluster along its PCA long axis if elongated."""
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    cluster_xy = cluster[:, :2]
    pca = PCA(n_components=n_components).fit(cluster_xy)
    proj = cluster_xy @ pca.components_[axis]  # project onto long axis

    # Cluster 1D projection
    proj = proj.reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(proj)
    labels = clustering.labels_

    # Return new clusters
    subclusters = []
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        subclusters.append(cluster[mask])
    return subclusters

def is_valid_pillar_bbox(box):
    dx, dy, dz = box["dimensions"]
    return (
        # These are configurable settings.
        0.1 < dx < 3.0 and
        0.1 < dy < 3.0 and
        2.0 < dz < 10.0
    )

def fit_wall_bounding_box(mismatch_points, points):
    from sklearn.cluster import DBSCAN
    import open3d as o3d
    import numpy as np

    clustering = DBSCAN(eps=0.2, min_samples=10).fit(mismatch_points)
    labels = clustering.labels_
    z_vals = points[:, 2]
    floor_z = np.percentile(z_vals, 1)
    ceil_z = np.percentile(z_vals, 99)

    fitted_boxes = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = mismatch_points[labels == label]

        # # Remove floor/ceiling outliers entirely. This is a configurable setting. It must be disabled when global trimming is enabled. 
        # z_valid_mask = np.logical_and(cluster[:, 2] > floor_z + 0.02, cluster[:, 2] < ceil_z - 0.02)
        # cluster = cluster[z_valid_mask]

        # Reject clusters with too many points on floor or ceiling (these are shadow artifacts)
        z_cluster = cluster[:, 2]
        z_outliers = np.logical_or(z_cluster <= floor_z, z_cluster >= ceil_z)
        if len(z_outliers) != 0:
            outlier_ratio = np.mean(z_outliers)
        else:
            outlier_ratio = 0
        if outlier_ratio > 0.05:  # more than 5% of points are floor/ceiling artifacts
            print(f"[DEBUG] → Skipping cluster {label}: {outlier_ratio:.1%} points on floor/ceiling")
            continue

        if len(cluster) < 30:
            print(f"[DEBUG] → Skipping cluster {label}: Too few points ({len(cluster)})")
            continue

        vertical_extent = np.ptp(cluster[:, 2])
        if vertical_extent < 0.1:
            print(f"[DEBUG] → Skipping cluster {label}: Vertical extent too small ({vertical_extent:.2f} m)")
            continue

        print(f"[WALL_DEBUG] → Cluster {label} with {len(cluster)} points")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster))
        # obb = pcd.get_oriented_bounding_box()
        # aabb = obb.get_axis_aligned_bounding_box()
        aabb = pcd.get_axis_aligned_bounding_box()

        dims = np.array(aabb.get_extent())
        thickness_dim = np.argmin(dims)
        mid = (aabb.get_min_bound()[thickness_dim] + aabb.get_max_bound()[thickness_dim]) / 2
        min_bound = list(aabb.get_min_bound())
        max_bound = list(aabb.get_max_bound())
        min_bound[thickness_dim] = mid - 0.075
        max_bound[thickness_dim] = mid + 0.075

        # rotation_angle = np.degrees(np.arctan2(obb.R[1, 0], obb.R[0, 0]))  # Z-axis rotation
        rotation_angle = 0.0

        object_shape = infer_object_type(cluster)

        if object_shape != "wall_object":
            print(f"[DEBUG] → Skipping cluster: Not a wall")
            continue  # skip non-wall clusters

        shape = "rectangular"  # wall shape always rectangular

        # Force full room height. This is a configurable setting. 
        min_bound[2] = floor_z
        max_bound[2] = ceil_z

        final_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        print(f"[WALL_DEBUG] → Inferred shape: {object_shape}")
        if 'final_box' in locals():
            print(f"[WALL_DEBUG] → Final box center: {final_box.get_center()}, dimensions: {final_box.get_extent()}")
        else:
            print("[WALL_DEBUG] → Skipped cluster, no box fitted.")

        fitted_boxes.append({
            "type": "wall",
            "shape": shape,
            "center": final_box.get_center().tolist(),
            "dimensions": final_box.get_extent().tolist(),
            "rotation": float(rotation_angle),
            "bbox": final_box
        })

    return fitted_boxes

def is_valid_wall_bbox(box):
    dx, dy, dz = box["dimensions"]
    thin_dim = min(dx, dy)
    long_dim = max(dx, dy)
    return (
        # These are configurable settings.
        0.05 < thin_dim < 0.25 and  # wall thickness
        0.5 < long_dim < 50.0 and   # wall length
        2.0 < dz < 10.0              # height
    )

def merge_wall_boxes(wall_boxes, angle_tolerance=5.0, gap_tolerance=0.15):
    """
    Merges wall boxes if their XY bounding regions overlap (in the long axis) and are similarly aligned.
    - angle_tolerance: max difference in rotation (degrees)
    - gap_tolerance: how far apart the thin axis can be before still considering them mergeable
    """
    merged_boxes = []
    used = [False] * len(wall_boxes)

    for i, box_i in enumerate(wall_boxes):
        if used[i]:
            continue

        center_i = np.array(box_i["center"])
        dims_i = np.array(box_i["dimensions"])
        rot_i = box_i.get("rotation", 0.0)
        min_i = center_i - dims_i / 2
        max_i = center_i + dims_i / 2
        dir_i = "x" if dims_i[0] > dims_i[1] else "y"

        bounds_i = [min_i.copy(), max_i.copy()]
        used[i] = True

        for j, box_j in enumerate(wall_boxes):
            if i == j or used[j]:
                continue

            center_j = np.array(box_j["center"])
            dims_j = np.array(box_j["dimensions"])
            rot_j = box_j.get("rotation", 0.0)
            min_j = center_j - dims_j / 2
            max_j = center_j + dims_j / 2
            dir_j = "x" if dims_j[0] > dims_j[1] else "y"

            # 1. Rotation check
            angle_diff = abs(rot_i - rot_j)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff > angle_tolerance:
                continue

            # 2. Direction check
            if dir_i != dir_j:
                continue

            # 3. Overlap check in long axis and small gap in short axis
            if dir_i == "x":
                overlap_long = not (max_i[0] < min_j[0] or max_j[0] < min_i[0])
                gap_short = abs(min_i[1] - max_j[1]) < gap_tolerance or abs(min_j[1] - max_i[1]) < gap_tolerance
            else:  # dir == "y"
                overlap_long = not (max_i[1] < min_j[1] or max_j[1] < min_i[1])
                gap_short = abs(min_i[0] - max_j[0]) < gap_tolerance or abs(min_j[0] - max_i[0]) < gap_tolerance

            if overlap_long and gap_short:
                bounds_i[0] = np.minimum(bounds_i[0], min_j)
                bounds_i[1] = np.maximum(bounds_i[1], max_j)
                used[j] = True

        new_center = (bounds_i[0] + bounds_i[1]) / 2
        new_dims = bounds_i[1] - bounds_i[0]
        new_bbox = o3d.geometry.AxisAlignedBoundingBox(bounds_i[0], bounds_i[1])

        merged_boxes.append({
            "center": new_center.tolist(),
            "dimensions": new_dims.tolist(),
            "shape": "rectangular",
            "type": "wall",
            "rotation": rot_i,
            "bbox": new_bbox
        })

    return merged_boxes

def fit_wall_bounding_box_pca_split(mismatch_points, points, split_eccentricity_threshold=5.0, min_cluster_size=30):
    """
    Similar to fit_wall_bounding_box, but includes PCA-based splitting of elongated clusters.
    """
    from sklearn.cluster import DBSCAN
    import open3d as o3d

    clustering = DBSCAN(eps=0.2, min_samples=10).fit(mismatch_points)
    labels = clustering.labels_
    z_vals = points[:, 2]
    floor_z = np.percentile(z_vals, 1)
    ceil_z = np.percentile(z_vals, 99)

    fitted_boxes = []

    for label in set(labels):
        if label == -1:
            continue

        cluster = mismatch_points[labels == label]

        # Remove floor/ceiling outliers
        z_valid_mask = np.logical_and(cluster[:, 2] > floor_z + 0.02, cluster[:, 2] < ceil_z - 0.02)
        cluster = cluster[z_valid_mask]
        if len(cluster) < min_cluster_size:
            print(f"[DEBUG] → Skipping cluster {label}: too few points after Z trim.")
            continue

        z_cluster = cluster[:, 2]
        z_outliers = np.logical_or(z_cluster <= floor_z, z_cluster >= ceil_z)
        outlier_ratio = np.mean(z_outliers) if len(z_outliers) != 0 else 0
        if outlier_ratio > 0.05:
            print(f"[DEBUG] → Skipping cluster {label}: {outlier_ratio:.1%} floor/ceiling points")
            continue

        vertical_extent = np.ptp(z_cluster)
        if vertical_extent < 0.1:
            print(f"[DEBUG] → Skipping cluster {label}: vertical extent too small")
            continue

        print(f"[WALL_DEBUG] → Cluster {label} with {len(cluster)} points")

        # PCA splitting if eccentricity is very high
        cluster_xy = cluster[:, :2]
        pca = PCA(n_components=2).fit(cluster_xy)
        ratio = pca.explained_variance_[0] / pca.explained_variance_[1]
        print(f"[WALL_DEBUG] → PCA ratio for cluster {label}: {ratio:.2f}")

        if ratio > split_eccentricity_threshold:
            projections = cluster_xy @ pca.components_[0]
            median_proj = np.median(projections)
            left_mask = projections < median_proj
            right_mask = ~left_mask
            subclusters = [cluster[left_mask], cluster[right_mask]]
            print(f"[WALL_DEBUG] → Split cluster {label} into {len(subclusters)} subclusters due to high eccentricity")

        else:
            subclusters = [cluster]

        for sub_id, subcluster in enumerate(subclusters):
            if len(subcluster) < min_cluster_size:
                print(f"[DEBUG] → Skipping subcluster {sub_id} of cluster {label}: too small")
                continue

            object_shape = infer_object_type(subcluster)
            if object_shape != "wall_object":
                print(f"[DEBUG] → Skipping subcluster {sub_id}: not inferred as wall")
                continue

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(subcluster))
            aabb = pcd.get_axis_aligned_bounding_box()

            dims = np.array(aabb.get_extent())
            thickness_dim = np.argmin(dims)
            mid = (aabb.get_min_bound()[thickness_dim] + aabb.get_max_bound()[thickness_dim]) / 2
            min_bound = list(aabb.get_min_bound())
            max_bound = list(aabb.get_max_bound())
            min_bound[thickness_dim] = mid - 0.075
            max_bound[thickness_dim] = mid + 0.075

            min_bound[2] = floor_z
            max_bound[2] = ceil_z
            final_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

            print(f"[WALL_DEBUG] → Subcluster {sub_id} box center: {final_box.get_center()}, dims: {final_box.get_extent()}")

            fitted_boxes.append({
                "type": "wall",
                "shape": "rectangular",
                "center": final_box.get_center().tolist(),
                "dimensions": final_box.get_extent().tolist(),
                "rotation": 0.0,
                "bbox": final_box
            })

    return fitted_boxes

def compute_bbox_iou(pred, gt):
    # Handle rectangular boxes (walls or square pillars)
    if pred["shape"] == "rectangular" and gt["shape"] == "rectangular":
        pred_center = np.array(pred.get("center", pred.get("position", [0, 0, 0])))
        gt_center = np.array(gt["center"])
        pred_size = np.array(pred["dimensions"])
        gt_size = np.array(gt["dimensions"])
        
        pred_min = pred_center - pred_size / 2
        pred_max = pred_center + pred_size / 2
        gt_min = gt_center - gt_size / 2
        gt_max = gt_center + gt_size / 2

        inter_min = np.maximum(pred_min, gt_min)
        inter_max = np.minimum(pred_max, gt_max)
        inter_size = np.maximum(0.0, inter_max - inter_min)

        inter_vol = np.prod(inter_size)
        pred_vol = np.prod(pred_size)
        gt_vol = np.prod(gt_size)
        union_vol = pred_vol + gt_vol - inter_vol
        return inter_vol / union_vol if union_vol > 0 else 0.0

    elif pred["shape"] == "cylinder" and gt["shape"] == "cylinder":
        from math import pi
        pred_r, gt_r = pred["dimensions"][0] / 2, gt["dimensions"][0] / 2
        pred_h, gt_h = pred["dimensions"][2], gt["dimensions"][2]
        pred_center = np.array(pred["center"])
        gt_center = np.array(gt["center"])
        d_xy = np.linalg.norm(pred_center[:2] - gt_center[:2])
        inter_area = 0
        if d_xy < pred_r + gt_r:
            r1, r2 = pred_r, gt_r
            h = min(pred_h, gt_h)
            inter_area = pi * min(r1, r2)**2
            vol_inter = inter_area * h
            vol_union = pi * r1**2 * pred_h + pi * r2**2 * gt_h - vol_inter
            return vol_inter / vol_union if vol_union > 0 else 0.0
        return 0.0
    else:
        return 0.0  # Incompatible shapes
