import numpy as np
import pyransac3d as pyrsc
from sklearn.decomposition import PCA
import transformations as tf

class POSE:
    def __init__(self):
        self.position = np.zeros(3)
        self.quaternion = np.zeros(4)
        self.euler = np.zeros(3)
        self.matrix = np.zeros((3,3))

def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    

    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T

    return (pc, rgb)

def extract_point_clouds(depth, K, segmap=None, rgb=None, z_range=[0,3], segmap_id=0, skip_border_objects=False, margin_px=5):
    """
    Converts depth map + intrinsics to point cloud. 
    If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

    Arguments:
        depth {np.ndarray} -- HxW depth map in m
        K {np.ndarray} -- 3x3 camera Matrix

    Keyword Arguments:
        segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
        rgb {np.ndarray} -- HxW rgb image (default: {None})
        z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
        segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
        skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
        margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

    Returns:
        [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
    """

    if K is None:
        raise ValueError('K is required either as argument --K or from the input numpy file')
        
    # Convert to pc 
    pc_full, pc_colors = depth2pc(depth, K, rgb)

    # Threshold distance
    if pc_colors is not None:
        pc_colors = pc_colors[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])] 

    pc_full = pc_full[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])]

    # Extract instance point clouds from segmap and depth map
    pc_segments = {}
    if segmap is not None:
        pc_segments = {}
        pc_segments_colors = {}
        # obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap>0])
        for i in range(1, np.max(segmap)+1):
            if skip_border_objects and not i==segmap_id:
                obj_i_y, obj_i_x = np.where(segmap==i)
                if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1]-margin_px) or np.any(obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0]-margin_px):
                    print('object {} not entirely in image bounds, skipping'.format(i))
                    continue
            inst_mask = segmap==i
            pc_segment,_ = depth2pc(depth*inst_mask, K)
            pc_segments[i-1] = pc_segment[(pc_segment[:,2] < z_range[1]) & (pc_segment[:,2] > z_range[0])] #regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

    return pc_full, pc_segments, pc_colors

def plane_from_points(points, plane_threshold=0.005, min_points=100):
    """
    Find object surface planes from point clouds using RANSAC

    Arguments:
        points {list} -- List of point clouds of different objects
        plane_threshold {float} -- Threshold error for plane fitting
        min_points {int} -- Minimum number of points that fit the plane
    """

    pyplane = pyrsc.Plane()
    if len(points) < min_points:
        return None, None
    plane, inliers = pyplane.fit(points, thresh=plane_threshold, minPoints=min_points)
    plane = np.array(plane)
    if plane[2] > 0:
        plane = -plane
    centroid = np.mean(points[inliers], axis=0)

    return centroid, plane

def get_pose(plane, centroid, dominant_axis):
    pose = POSE()
    pose.position = centroid
    
    # We get the orientation of the plane
    z = plane[0:3]
    z = z / np.linalg.norm(z)
    x = dominant_axis
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    pose.matrix = np.array([x, y, z])
    rotation = np.eye(4)
    rotation[0:3, 0:3] = pose.matrix
    pose.euler = tf.euler_from_matrix(rotation)
    pose.quaternion = tf.quaternion_from_matrix(rotation)
    
    return pose

def estimate_pose(depth, mask=None, K=None, pointcloud=None):

    # We get the caps segmented point clouds
    if pointcloud is None:
        pointcloud, _ = depth2pc(depth*mask, K)

    # We get the cap planes and centroids
    centroid, plane = plane_from_points(pointcloud, plane_threshold=0.005, min_points=40)

    # We get the dominant axis
    pca = PCA(2,svd_solver='full')
    pca.fit(pointcloud) # find principal components
    dominant_axis = pca.components_[0] / np.linalg.norm(pca.components_[0])

    # We get the poses
    pose = get_pose(plane, centroid, dominant_axis)

    return pose