"""Interface to Ai2thor simulator."""

from interfaces.base_world_interface import BaseWorldInterface
from ai2thor.controller import Controller
import numpy as np
from reflect.main.scene_graph import SceneGraph as BaseSceneGraph
from reflect.main.scene_graph import Node, Edge

import torch
torch.set_grad_enabled(False)
torch.manual_seed(0)

from reflect.main.get_local_sg import get_2d_bbox_from_3d_pcd
from reflect.main.point_cloud_utils import *
import open3d as o3d
from PIL import Image
from reflect.main.utils import *

DIRECTIONS = {
    'w' : "MoveAhead",
    'a' : "MoveLeft",
    's' : "MoveBack",
    'd' : "MoveRight"
}
ROTATIONS = {
    'r' : 'RotateRight',
    'l' : 'RotateLeft'
}


def get_node(name, event, obj_held_prev=False):
    object_id = event.metadata['objects'].index(name)
    height, width, channel = event.frame.shape
    camera_space_xyz = depth_frame_to_camera_space_xyz(
            depth_frame=torch.as_tensor(event.depth_frame.copy()), mask=None, fov=event.metadata['fov'])
    x = event.metadata['agent']['position']['x']
    y = event.metadata['agent']['position']['y']
    z = event.metadata['agent']['position']['z']

    if not event.metadata['agent']['isStanding']:
        y = y - 0.22

    world_points = camera_space_xyz_to_world_xyz(
        camera_space_xyzs=camera_space_xyz,
        camera_world_xyz=torch.as_tensor([x, y, z]),
        rotation=event.metadata['agent']['rotation']['y'],
        horizon=event.metadata['agent']['cameraHorizon'],
    ).reshape(channel, height, width).permute(1, 2, 0)

    sinkbasin_pts = None

    if object_id.split("|")[0] in ["Window", "Floor", "Wall", "Ceiling", "Cabinet"]:
        return None
    label = object_id

    # register this box in 3D
    mask = event.instance_masks[object_id].reshape(height, width)
    obj_points = torch.as_tensor(world_points[mask])
    
    if len(obj_points) < 700:
        return None
    
    depth_obj = event.depth_frame[mask]
    # obj_colors = torch.as_tensor(world_colors[mask])

    # downsample point cloud
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    voxel_down_pcd = obj_pcd.voxel_down_sample(voxel_size=0.01)

    # denoise point cloud
    if "Pan" == label.split("|")[0] or "EggCracked" == label.split("|")[0] or "Bowl" == label.split("|")[0] or "Pot" == label.split("|")[0]:
        _, ind = voxel_down_pcd.remove_radius_outlier(nb_points=30, radius=0.03)
        inlier = voxel_down_pcd.select_by_index(ind)
        pcd_obj = torch.tensor(np.array(inlier.points))
    elif "CounterTop" == label.split("|")[0]:
        _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
        inlier = voxel_down_pcd.select_by_index(ind)
        pcd_obj = torch.tensor(np.array(inlier.points))
    elif "SinkBasin" in label:
        sinkbasin_pts = torch.tensor(np.array(voxel_down_pcd.points))
    else:
        pcd_obj = torch.tensor(np.array(voxel_down_pcd.points))
        # assert points[label].shape == colors[label].shape
    #==============================================================

    total_points = pcd_obj

    if is_receptacle(label, event):
        if is_moving(label, event) or is_picked_up(label, event) or obj_held_prev == label:
            total_points = pcd_obj
        else:
            total_points = torch.unique(torch.cat((total_points, pcd_obj), 0), dim=0)
    else:
        total_points = pcd_obj

    if label.split("|")[0] == "Sink" and sinkbasin_pts is not None:
        total_points = torch.unique(torch.cat((total_points, sinkbasin_pts), 0), dim=0)

    boxes3d_pts = o3d.utility.Vector3dVector(total_points)
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)

    # Generate local scene graph
    total_points_dict = {}
    total_points_dict[label] = total_points
    bbox = get_2d_bbox_from_3d_pcd(event, label, total_points_dict)
    if name is not None and bbox is not None:
        node = Node(name, 
                    object_id=label, 
                    pos3d=box.get_center(), 
                    corner_pts=np.array(box.get_box_points()), 
                    bbox2d=bbox, 
                    pcd=total_points,
                    depth=depth_obj)
        return node
    return None

class SceneGraph(BaseSceneGraph):
        
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
            self.edges[(target_object, relative_object)] = Edge(target_object, relative_object, relation)


class WorldInterface(BaseWorldInterface):
    
    def __init__(self, scene='FloorPlan1', movable_objects=[], graspable_objects=[]):
        self.controller = Controller(agentMode="default", visibilityDistance=1.0, scene=scene, gridSize=0.25, rotateStepDegrees=45.0)
        self.graspable_objects = graspable_objects
        self.movable_objects = movable_objects
        self.scene_graph = SceneGraph(event=self.controller.last_event, task=None)
        self.scene_graph_nodes = [node.name for node in self.scene_graph.total_nodes]
        
        self.grasped_object = None
        self.manipulation_target = None
        self.object_positions = {}
        self.object_position_known = {}
        self.object_upright = {}
        self.object_opened = {}
        self.object_unlocked = {}
        self.held_prev = []

        self.error_message = ''
        self.failed_behavior = ''
        
    def get_feedback(self):
        event = self.controller.last_event
        self.robot_position = self.controller.last_event.metadata['agent']['position']
        self.robot_orientation = self.controller.last_event.metadata['agent']['rotation']

        self.color_frame = self.controller.last_event.cv2img
        self.depth_frame = self.controller.last_event.depth_frame
        
        for obj in self.controller.last_event.metadata['objects']:   
            if obj['pickupable']:
                self.graspable_objects.append(obj['object_id'])
                if obj['isPickedUp']:
                    self.grasped_object = obj['object_id']
                    self.held_prev.append(obj['object_id'])
            if obj['movable']:
                self.movable_objects.append(obj['object_id'])                
            if obj['toggleable']:
                self.object_unlocked[obj['object_id']] = obj['isToggled']
            if obj['openable']:
                self.object_opened[obj['object_id']] = obj['isOpened']

            if obj['object_id'] not in self.scene_graph_nodes:
                if obj['visible']:
                    # node = get_node(obj['object_id'], event, obj['object_id'] in self.held_prev)
                    node = Node(obj['name'], object_id=obj['objectId'])
                    self.scene_graph.add_node_wo_edge(node)
                    if node is not None:
                        self.scene_graph.add_node(node)
                    self.object_positions[obj[obj['object_id']]] = self.dict_to_pos(obj['position'])
                    self.object_position_known[obj[obj['object_id']]] = True
                else:
                    self.object_position_known[obj[obj['object_id']]] = False
    
    def get_position(self, target_object):
        target_id = self.controller.last_event.metadata['objects'].index(target_object)
        return self.controller.last_event.metadata['objects'][target_id]['position']
    
    def is_near_robot(self, target_object, distance=0.6):
        """ Checks if object is within reach """
        if self.object_position_known[target_object] and \
            self.calc_distance(target_object, self.dict_to_pos(self.robot_position)) < distance:
            return True
        return False
    
    def object_at(self, target_object, relation, relative_object):
        return relation == self.scene_graph.edges[(target_object, relative_object)].edge_type
    
    def move(self, direction, magnitude=0.25):
        """ Move one step in the specified direction """
        return self.controller.step(action=DIRECTIONS[direction], moveMagnitude=magnitude)
    
    def rotate(self, rotation, degrees=90):
        """ Rotate the robot in the specified direction """
        return self.controller.step(action=ROTATIONS[rotation], degrees=degrees)
    
    def navigate_to(self, target_object):
        """ Navigate to a specific object """
        raise NotImplementedError
    
    def pick(self, target_object):
        """ Pick up an object """
        return self.controller.step(action='PickupObject', objectId=target_object)
    
    def place(self, target_object, position):
        """ Place an object at a specific location """
        return self.controller.step(action='PutObject', objectId=target_object, position=position)
    
    def drop(self):
        """ Drop the object held by the robot """
        return self.controller.step(action='DropHandObject')
    
    def put_on(self, target_object, receptacle):
        """ Put an object on another object """
        placing_position = self.controller.step(action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle)
        return self.controller.step(action='PutObject', objectId=target_object, position=placing_position)
    
    def pos_to_dict(self, pos):
        return {'x': pos[0], 'y': pos[1], 'z': pos[2]}
    
    def dict_to_pos(self, pos):
        return np.array([pos['x'], pos['y'], pos['z']])