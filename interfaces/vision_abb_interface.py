from vision.client import MLDetector
from vision.pose_estimator import POSE, estimate_pose
from interfaces.base_world_interface import BaseWorldInterface
from interfaces.abb_world_interface import WorldInterface as AbbWorldInterface
from vision.kinect_camera import KinectCamera
import open3d as o3d
import torch
import vision.perception_utils as utils
import vision.k4a as k4a
import numpy as np
import cv2
import os
import time
from reflect.main.utils import get_pcd_dist, is_inside

IMAGE_DIR = "BETR-XP-LLM/detections/"
# =========  Parameters for spatial relation heuristics ============
IN_CONTACT_DISTANCE = 0.01
CLOSE_DISTANCE = 0.02
INSIDE_THRESH = 0.5 # increqasing it makes on and decreasing it makes inside
ON_TOP_OF_THRESH = 0.4
NORM_THRESH_FRONT_BACK = 0.9
NORM_THRESH_UP_DOWN = 0.9
NORM_THRESH_LEFT_RIGHT = 0.8
OCCLUDE_RATIO_THRESH = 0.5
DEPTH_THRESH = 0.9
BULKY_OBJECTS = ["green box"]
# ==================================================================



def gen_node(obj, pose, mask, pcd):
    name = obj
    # total_points = torch.tensor(np.array([]))
        
    # downsample point cloud
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(pcd)
    voxel_down_pcd = obj_pcd.voxel_down_sample(voxel_size=0.01)

    # denoise point cloud
    pcd_obj = torch.tensor(np.array(voxel_down_pcd.points))

    #==============================================================

    total_points = pcd_obj

    boxes3d_pts = o3d.utility.Vector3dVector(pcd_obj)
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)

    node = Node(name=name,
                pose=pose,
                mask=mask,
                pos3d=box.get_center(), 
                corner_pts=np.array(box.get_box_points()), 
                pcd=total_points)
    return node

class Edge(object):
    def __init__(self, start_node, end_node, edge_type="none"):
        self.start = start_node
        self.end = end_node
        self.edge_type = edge_type
    
    def __hash__(self):
        return hash((self.start, self.end, self.edge_type))

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end and self.edge_type == other.edge_type:
            return True
        else:
            return False

    def __str__(self):
        return str(self.start) + "->" + self.edge_type + "->" + str(self.end)

class Node(object):
    def __init__(self, name, object_id=None, pos3d=None, corner_pts=None, bbox2d=None, pcd=None, mask=None, pose=None, global_node=False):
        self.name = name
        self.object_id = object_id # object_id
        self.bbox2d = bbox2d # 2d bounding box (4x1)
        self.pose = pose # object pose
        self.pos3d = pose.position if pos3d is None else pos3d # object position
        self.orientation = pose.quaternion if pose is not None else None # object orientation
        self.corner_pts = corner_pts # corner points of 3d bbox (8x3)
        self.pcd = pcd # point cloud (px3)
        self.mask = mask
        self.name_w_state = None
        self.global_node = global_node

    def set_state(self, state):
        self.name_w_state = state

    def __str__(self):
        return self.get_name()

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return True if self.get_name() == other.get_name() else False

    def get_name(self):
        if self.name_w_state is not None:
            return self.name_w_state
        else:
            return self.name

class SceneGraph(object):
    """
    Create a spatial scene graph
    """
    def __init__(self):
        self.nodes = []
        self.total_nodes = []
        self.edges = {}
        
    def add_node_wo_edge(self, node):
        self.total_nodes.append(node)

    def add_node(self, new_node):
        for node in self.total_nodes:
            if node is None:
                continue
            if node.name != new_node.name:
                self.add_edge(node, new_node)
                self.add_edge(new_node, node)
                self.nodes.append(new_node)
        return new_node

    # def add_edge(self, node, new_node):
    #     target_object = new_node.name
    #     relative_object = node.name
    #     relation = None

    #     if relative_object not in self.graspable_objects:
    #         return
    #     if self.object_position_known[target_object] and self.object_position_known[relative_object]:
    #         if abs(self.object_positions[target_object][0] - self.object_positions[relative_object][0]) < 0.01 and \
    #             abs(self.object_positions[target_object][1] - self.object_positions[relative_object][1]) < 0.01 and \
    #             abs(self.object_positions[target_object][2] - \
    #                 BaseWorldInterface.CUBE_SIZE - self.object_positions[relative_object][2]) < 0.01:
    #             relation = 'on'

    #         elif np.sum(new_node.mask) > np.sum(node.mask):
    #             if abs(self.object_positions[target_object][0] - self.object_positions[relative_object][0]) < 0.01 and \
    #             abs(self.object_positions[target_object][1] - self.object_positions[relative_object][1]) < 0.01 and \
    #             abs(self.object_positions[target_object][2] - self.object_positions[relative_object][2]) < 0.03:
    #                 relation = 'inside'

    #     elif isinstance(relative_object, np.ndarray):
    #         if self.object_position_known[target_object]:
    #             if self.calc_distance(target_object, relative_object) < 0.01:
    #                relation = 'at'
                   
    #     if relation is not None:
    #         self.edges[(target_object, relative_object)] = Edge(target_object, relative_object, relation)

    def add_edge(self, node, new_node):

        box_A, box_B = np.array(node.corner_pts), np.array(new_node.corner_pts)
        if len(node.pcd) == 0 or len(new_node.pcd) == 0:
            return
        else:
            dist = get_pcd_dist(node.pcd, new_node.pcd)
        
        box_A_pts, box_B_pts = np.array(node.pcd), np.array(new_node.pcd)


        # IN CONTACT
        if dist < IN_CONTACT_DISTANCE:
            if new_node.name not in BULKY_OBJECTS:
                # print("Long expression: ", len(np.where((box_B_pts[:, 0] < box_A[4, 0]) & (box_B_pts[:, 0] > box_A[0, 0]) & 
                #         (box_B_pts[:, 2] < box_A[4, 2]) & (box_B_pts[:, 2] > box_A[0, 2]))[0]))
                # print("Compared against: ", len(box_B_pts) * ON_TOP_OF_THRESH)

                if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):
                    print("Distance: ", np.linalg.norm(np.array(new_node.pose.position) - np.array(node.pose.position)))
                    if np.linalg.norm(np.array(new_node.pose.position) - np.array(node.pose.position)) < CLOSE_DISTANCE:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")

                elif len(np.where((box_B_pts[:, 0] < box_A[4, 0]) & (box_B_pts[:, 0] > box_A[0, 0]) & 
                        (box_B_pts[:, 2] < box_A[4, 2]) & (box_B_pts[:, 2] > box_A[0, 2]))[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:
                    # print("\n Passed First Condition \n")
                    # print("Long expression: ", len(np.where(box_B_pts[:, 1] > box_A[4, 1])[0]))
                    # print("Compared against: ", len(box_B_pts) * ON_TOP_OF_THRESH)
                    # if len(np.where(box_B_pts[:, 1] > box_A[4, 1])[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on")

                    # elif len(np.where(box_A_pts[:, 1] > box_B[4, 1])[0]) > len(box_A_pts) * ON_TOP_OF_THRESH:
                    #     if node.name not in BULKY_OBJECTS:
                    #         self.edges[(node.name, new_node.name)] = Edge(node, new_node, "on")

    def __eq__(self, other):
        if (set(self.nodes) == set(other.nodes)) and (set(self.edges.values()) == set(other.edges.values())):
            return True
        else:
            return False

    def __str__(self):
        visited = []
        res = "[Nodes]:\n"
        for node in set(self.nodes):
            res += node.get_name()
            res += "\n"
        res += "\n"
        res += "[Edges]:\n"
        for edge_key, edge in self.edges.items():
            name_1, name_2 = edge_key
            edge_key_reversed = (name_2, name_1)
            if (edge_key not in visited and edge_key_reversed not in visited) or edge.edge_type in ['on', 'inside', 'occluding']:
                res += str(edge)
                res += "\n"
            visited.append(edge_key)
        return res

class WorldInterface(AbbWorldInterface):
    """
    Class for handling the simple planning world
    """
    def __init__(self, cfree_interface, rws, movable_objects, graspable_objects=None, table_offset=0, use_vision=True, known_objects=[], root_folder_path=''):
        self.gripper_position = 0
        self.known_objects = known_objects
        self.root_folder_path = root_folder_path
        video_path = os.path.join(root_folder_path, 'video.avi')
        self.video_color = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 4, (960, 960))
        self.image_index = 6
        self.hole_pose = None

        self.use_vision = use_vision
        if use_vision:
            self.detector = MLDetector("localhost:50051")
            self.detector.check_connection()
            self.camera = KinectCamera()
            self.cropping = [1250, 2500, 1000, 3000]
            pos = np.array([0.117, -0.038, 0.647])
            quat = np.array([-0.645, 0.653, -0.273, 0.288])
            self.T_camera_in_robot = utils.homogeneous_matrix(pos, quat) #pylint:disable=invalid-name

        self.graspable_objects = graspable_objects
        self.movable_objects = movable_objects
        self.scene_graph = SceneGraph()
        # self.scene_graph_nodes = [node.name for node in self.scene_graph.total_nodes]
        self.scene_graph_nodes = []
        self.scene_graph_file = 'scene_graph.txt'
        self.text_graph = ''
        self.hierarchical_summary_file = 'hierarchical_summary.txt'
        BaseWorldInterface.__init__(self, cfree_interface, rws, movable_objects, graspable_objects, table_offset)

        self.scene_changes = []

        self.failure_added = False
        self.scene_graph.object_position_known = self.object_position_known
        self.scene_graph.object_positions = self.object_positions
        self.scene_graph.graspable_objects = self.graspable_objects

        self.get_feedback()
    
    def update_scene_graph_file(self, file_path=None):
        """
        Reads the scene graph from a file and returns its content as text.
        """
        self.text_graph = ""
        for edge in self.scene_graph.edges.keys():
            edge_text = f"{self.scene_graph.edges[edge]}"
            self.text_graph += edge_text + "\n"
            
        object_text = "\nKnown Object Locations: ["
        for object in self.object_position_known.keys():
            if self.object_position_known[object]:
                object_text += " " + object.split("|")[0] + ","
                
        self.text_graph += object_text + "]"

        if file_path is None:
            file_path=self.root_folder_path
        try:
            file_path = os.path.join(file_path, self.scene_graph_file)        
            with open(file_path, 'w') as f:
                f.write(self.text_graph)
            return self.scene_graph_file
        except FileNotFoundError:
            print(f"[ERROR] Scene graph file '{self.scene_graph_file}' not found.")
            return None
        
    def update_hierarchical_summary_file(self, file_path=None):
        """
        Reads the hierarchical summary from a file and returns its content as text.
        """
        if self.scene_changes == []:
            return

        Timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        observation = ""
        for edge in self.scene_changes:
            observation += f" {edge},"
        self.text_summary = f"Timestamp: {Timestamp} | Observation:{observation}\n"
        self.scene_changes = []
        
        if file_path is None:
            file_path=self.root_folder_path
        try:
            file_path = os.path.join(file_path, self.hierarchical_summary_file)        
            with open(file_path, 'a') as f:
                f.write(self.text_summary)
            return self.hierarchical_summary_file
        except FileNotFoundError:
            print(f"[ERROR] Hierarchical summary file '{self.hierarchical_summary_file}' not found.")
            return None

    def get_feedback(self):
        """ Get feedback from sensors to update world state """
        self.gripper_position = float(self.rws.ios_get_signal_value("hand_ActualPosition_R")) / 10000
        if self.gripper_position < 0.005:
            #This means we lost the object which allowed the fingers to close fully
            if self.grasped_object is not None:
                self.object_position_known[self.grasped_object] = False
                if (self.grasped_object,"robot_gripper") in self.scene_graph.edges.keys():
                    self.scene_graph.edges.pop((self.grasped_object,"robot_gripper"))
                self.update_scene_graph_file()
                self.grasped_object = None
                self.stop() # Almost definitely we should stop here

        # run pose estimation pipeline
        if self.use_vision:
            rgb_img, depth_img, _  = self.get_updated_image()
            # depth_img = cv2.imread(os.path.join(IMAGE_DIR, "depth.png"), cv2.IMREAD_UNCHANGED)

            for obj in self.known_objects:
                # if "green" in obj:
                #     prompt = "circle shaped hole in the center green box"
                # if "square hole" in obj:
                #     prompt = "square shaped hole in the center green box"
                # elif "circle hole" in obj:
                #     prompt = "circle shaped hole in the center of green box"
            #     elif "black cube" in obj:
            #         prompt = "black black cube"
            #     else:
            #         prompt = obj

            #     objects += prompt + " . "

            # masks, boxes, scores, labels = self.detector.detect(rgb_img, prompt)
            
            # color = np.array([200, 30, 230])
            # if "black cube" in obj:
            #     i = np.argmin([np.sum(mask) for mask in masks])                    
            # else:
            #     i = np.argmax(scores)

                mask = None
                while mask is None:
                    mask = cv2.imread(os.path.join(IMAGE_DIR, obj+'.png'), cv2.IMREAD_GRAYSCALE)
                points = utils.get_points_3D(mask, depth_img, self.camera, self.T_camera_in_robot, self.cropping)
                pointcloud = np.array(points)
                pose = estimate_pose(mask, pointcloud=pointcloud)
                pose.position[2] = pose.position[2] + 0.03

            # visualize detection results
            # overlay = np.zeros_like(rgb_img)
            # overlay[mask > 0] = color
            # annotated = cv2.addWeighted(rgb_img, 1, overlay, 0.5, 0)
            # center = np.mean(pointcloud, axis=0)
            # center = (center[0], center[1], center[2])
            # pose2d = self.camera._transformation.point_3d_to_pixel_2d(center, k4a.ECalibrationType.COLOR, k4a.ECalibrationType.COLOR)

            # Draw center of object
            # annotated = cv2.circle(annotated, (int(pose2d[0]), int(pose2d[1])), 5, (0, 255, 0), -1)
            # cv2.imwrite('rgb_img.png', annotated)


                # hole_mask = cv2.imread(os.path.join(IMAGE_DIR, 'hole.png'), cv2.IMREAD_GRAYSCALE)
                # hole_points = utils.get_points_3D(hole_mask, depth_img, self.camera, self.T_camera_in_robot, self.cropping)
                # hole_pointcloud = np.array(hole_points)
                # self.hole_pose = estimate_pose(mask, pointcloud=hole_pointcloud)

                self.update_scene_graph(label=obj, mask=mask, pose=pose, pointcloud=pointcloud)
             
        self.update_scene_graph_file()
        self.update_hierarchical_summary_file()
        
    def update_scene_graph(self, label, mask, pose, pointcloud):
        self.object_position_known[label] = True
        # new_node = Node(name=label, pose=pose, mask=mask, pcd=pointcloud)
        new_node = gen_node(obj=label, pose=pose, mask=mask, pcd=pointcloud)
        if new_node.name not in self.scene_graph_nodes:
            # if new_node.name == "green box":
            #     self.object_positions[new_node.name] = self.hole_pose.position
            # else:
            self.object_positions[new_node.name] = new_node.pose.position
            self.scene_graph.add_node_wo_edge(new_node)
            self.scene_graph.add_node(new_node)
            self.scene_graph_nodes.append(new_node.name)
        else:
            if new_node.name in self.movable_objects:
                if abs(self.calc_distance(new_node.name, new_node.pose.position)):
                    self.object_positions[new_node.name] = new_node.pose.position
                    remove_list = []
                    for edge in self.scene_graph.edges.keys():
                        if new_node.name in edge:
                            remove_list.append(edge)
                    for edge in remove_list:
                        self.scene_graph.edges.pop(edge)

                    for node in self.scene_graph.total_nodes:
                        if node.name == new_node.name:
                            self.scene_graph.total_nodes.remove(node)
                    # for node in self.scene_graph.nodes:
                    #     if node.name == new_node.name:
                            # self.scene_graph.nodes.remove(node)

                    self.scene_graph.add_node_wo_edge(new_node)
                    self.scene_graph.add_node(new_node)

    def get_updated_image(self, file_path=None):
        """ Returns the current image of the last event"""
        if file_path is None:
            file_path = os.path.join(IMAGE_DIR, 'rgb.jpg')
        # rgb_img, depth_img, _  = self.camera.get_image(self.cropping)
        # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.imread(os.path.join(IMAGE_DIR, "rgb.jpg"), cv2.IMREAD_COLOR)
        counter = 0
        while counter < 3:
            try:
                depth_img = np.load(os.path.join(IMAGE_DIR, "depth.npy"))
                break
            except:
                time.sleep(1)
                counter += 1

        # cv2.imwrite(file_path, rgb_img)
        self.video_color.write(rgb_img)
        return rgb_img, depth_img, [file_path]
    
    def set_grasped_object(self, target_object):
        """ Set grasped object"""
        self.grasped_object = target_object
        self.object_position_known[target_object] = False
        self.scene_graph.edges[(target_object, "robot_gripper")] = Edge(target_object, "robot_gripper", edge_type="inside")

    def object_at(self, target_object, relation, relative_object):
        """ Check if object is at a specific location """
        if (target_object, relative_object) in self.scene_graph.edges.keys():
            return self.scene_graph.edges[(target_object, relative_object)].edge_type == relation
        return False