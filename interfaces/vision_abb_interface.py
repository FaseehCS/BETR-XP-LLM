from vision.client import MLDetector
from interfaces.base_world_interface import BaseWorldInterface
from abb_world_interface import WorldInterface as AbbWorldInterface
from vision.kinect_camera import KinectCamera
import vision.perception_utils as utils
from reflect.main.scene_graph import Edge as GraphEdge
import numpy as np
import cv2
import os
import time

class Node(object):
    def __init__(self, name, object_id=None, pos3d=None, corner_pts=None, bbox2d=None, pcd=None, mask=None, pose=None, global_node=False):
        self.name = name
        self.object_id = object_id # object_id
        self.bbox2d = bbox2d # 2d bounding box (4x1)
        self.pose = pose # object pose
        self.pos3d = pose.position if pos3d is None else pos3d # object position
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

    def add_edge(self, node, new_node):
        target_object = new_node.name
        relative_object = node.name
        relation = None

        if self.object_position_known[target_object] and self.object_position_known[relative_object]:
            if abs(self.object_positions[target_object][0] - self.object_positions[relative_object][0]) < 0.01 and \
                abs(self.object_positions[target_object][1] - self.object_positions[relative_object][1]) < 0.01 and \
                abs(self.object_positions[target_object][2] - \
                    BaseWorldInterface.CUBE_SIZE - self.object_positions[relative_object][2]) < 0.01:
                relation = 'on'

            elif abs(self.object_positions[target_object][0] - self.object_positions[relative_object][0]) < 0.01 and \
                abs(self.object_positions[target_object][1] - self.object_positions[relative_object][1]) < 0.01 and \
                abs(self.object_positions[target_object][2] - self.object_positions[relative_object][2]) < 0.03:
                relation = 'in'

        elif isinstance(relative_object, np.ndarray):
            if self.object_position_known[target_object]:
                if self.calc_distance(target_object, relative_object) < 0.01:
                   relation = 'at'
                   
        if relation is not None:
            self.edges[(target_object, relative_object)] = GraphEdge(target_object, relative_object, relation)

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
        video_path = os.path.join(root_folder_path, 'video.avi')
        self.video_color = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 4, (960, 960))

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
        self.scene_graph = SceneGraph(event=self.controller.last_event, task=None)
        # self.scene_graph_nodes = [node.name for node in self.scene_graph.total_nodes]
        self.scene_graph_nodes = []
        self.scene_graph_file = 'scene_graph.txt'
        self.text_graph = ''
        self.hierarchical_summary_file = 'hierarchical_summary.txt'

        self.scene_changes = []

        self.failure_added = False
        self.scene_graph.object_position_known = self.object_position_known
        self.scene_graph.object_positions = self.object_positions        


        BaseWorldInterface.__init__(self, cfree_interface, rws, movable_objects, graspable_objects, table_offset)

    def get_updated_image(self , file_path=None):
        """ Returns the current image of the last event"""
        if file_path is None:
            file_path=self.root_folder_path
        image = self.controller.last_event.cv2img
        file_path = os.path.join(file_path, 'updated_image.png')
        cv2.imwrite(file_path, image)
        return [file_path]
    
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
        self.gripper_position = int(self.rws.ios_get_signal_value("hand_ActualPosition_R")) / 10000
        if self.gripper_position < 0.005:
            #This means we lost the object which allowed the fingers to close fully
            if self.grasped_object is not None:
                self.object_position_known[self.grasped_object] = False
                self.grasped_object = None
                self.stop() # Almost definitely we should stop here

        # run pose estimation pipeline
        if self.use_vision:
            rgb_img, _, depth_img  = self.camera.get_image(self.cropping)
            intrinsics = self.camera._calibration.extrinsics
            objects = f"{self.known_objects.replace("[", "").replace("]", "").replace("'", "")}"
            # masks, boxes, scores, labels = self.detector.detect(rgb_img, objects)
            poses, masks, boxes, scores, labels = self.detector.detect_pose(rgb_img, depth_img, intrinsics, prompt=objects, box_threshold=0.3)
            for label, box, mask, pose, score in zip(labels, boxes, masks, poses, scores):
                self.update_scene_graph(label, box, mask, pose, score)
             
        self.update_scene_graph_file()
        self.update_hierarchical_summary_file()
        
    def update_scene_graph(self, label, box, mask, pose, score):
        self.object_position_known[label] = True
        new_node = Node(name=label, pose=pose, bbox2d=box, mask=mask)
        if new_node.name not in self.scene_graph_nodes:
            self.object_positions[new_node.name] = new_node.pos3d
            self.scene_graph.add_node_wo_edge(new_node)
            self.scene_graph.add_node(new_node)
            self.scene_graph_nodes.append(new_node.name)
        else:
            if new_node.name in self.movable_objects:
                if abs(self.calc_distance(new_node.pos3d, self.object_positions[new_node.name])):
                    self.object_positions[new_node.name] = new_node.pos3d
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
                            self.scene_graph.nodes.remove(node)

                    self.scene_graph.add_node_wo_edge(new_node)
                    self.scene_graph.add_node(new_node)