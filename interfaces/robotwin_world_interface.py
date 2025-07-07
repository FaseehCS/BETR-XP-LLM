from interfaces.base_world_interface import BaseWorldInterface
from robotwin.envs._base_task import Base_Task
from collections import defaultdict
import numpy as np

class RobotwinWorldInterface(BaseWorldInterface):
    def __init__(self, movable_objects=None, graspable_objects=None, table_offset=0, config_kwargs=None):
        super().__init__(
            cfree_interface=None,
            movable_objects=movable_objects,
            graspable_objects=graspable_objects,
            table_offset=table_offset,
        )
        self.task = Base_Task()
        self.task._init_task_env_(**(config_kwargs or {}))
        self._build_name_to_actor()
        self.grasped_object = None
        self.manipulation_target = None

    # Utility to build a mapping from object names to actors
    def _build_name_to_actor(self):
        self._name_to_actor = defaultdict(list)
        for actor in self.task.scene.get_all_actors():  # Sapien API to get all actors
            name = actor.get_name()
            if name and name not in ['table', 'wall', 'ground']:
                self._name_to_actor[name].append(actor)

    # Utility to get an actor by name
    def _get_actor(self, object_name, all=False):
        actors = self._name_to_actor.get(object_name, [])
        if not actors:
            return None
        return actors if all else actors[0]
    
    # === 1. ENVIRONMENT CONTROL ===
    def reset(self, config_kwargs=None):
        """
        Reset the world/simulation for a new episode.
        """
        # Re-initialize the task/environment
        if config_kwargs is None:
            config_kwargs = {}
        self.task._init_task_env_(**config_kwargs)
        self.grasped_object = None
        self.manipulation_target = None

    def close_env(self, clear_cache=False):
        """
        Cleanup and close the environment.
        """
        self.task.close_env(clear_cache=clear_cache)

    def delay(self, time_steps):
        """
        Hold current state for some steps (simulate waiting).
        """
        self.task.delay(time_steps)

    # === 2. STATE FEEDBACK & OBSERVATION ===

    def get_feedback(self):
        """Update state from sensors/scene and refresh object locations."""
        self.task.get_obs()

    def get_obs(self):
        """Get the current observation (images, pointclouds, joint states, end-effector pose)."""
        return self.task.get_obs()

    def get_scene_graph(self):
        """Return a scene graph or logical structure of the current world state."""
        # _base_task.py does not explicitly have a scene graph, but we can use now_obs as a base
        # for now, return the current observation dictionary. 
        return getattr(self.task, 'now_obs', {})

    def get_scene_contact(self):
        """Debug: print or return contacts in the scene."""
        return self.task.get_scene_contact()

     # === 3. OBJECT & GRIPPER STATE QUERIES ===
    def get_object_pose(self, object_name):
        """Return the pose (position and orientation) of an object in the scene."""
        if hasattr(self.task, 'get_pose'):
            return self.task.get_pose(object_name)
        return None

    def get_robot_pose(self, arm='left'):
        """Return the pose (position and orientation) of the specified robot arm's end effector."""
        obs = self.get_obs()
        if "endpose" not in obs:
            return None
        endpose = obs["endpose"]
        if arm == "left":
            return endpose[:7]  # [x, y, z, roll, pitch, yaw, gripper]
        elif arm == "right":
            return endpose[7:14]
        else:
            return None

    def get_arm_pose(self, arm='left'):
        """Alias for get_robot_pose."""
        return self.get_robot_pose(arm)

    def is_object_on(self, object_a, object_b):
        """Check if object_a is placed on object_b."""
        # Use contact info and possibly height/pose relations from obs
        # Placeholder: use check_actors_contact plus Z-difference
        contact = self.check_actors_contact(object_a, object_b)
        if not contact:
            return False
        pose_a = self.get_object_pose(object_a)
        pose_b = self.get_object_pose(object_b)
        if pose_a is None or pose_b is None:
            return False
        # Simple Z-axis check: is A above B (threshold can be tuned)
        return pose_a[2] > pose_b[2] + 0.01

    def is_object_in(self, object_a, object_b):
        """Check if object_a is inside object_b."""
        # Placeholder: use bounding boxes or proximity in (x, y, z)
        pose_a = self.get_object_pose(object_a)
        pose_b = self.get_object_pose(object_b)
        if pose_a is None or pose_b is None:
            return False
        # Assume object_b is a container, check proximity
        dist = np.linalg.norm(np.array(pose_a[:3]) - np.array(pose_b[:3]))
        return dist < 0.1  # Threshold for "inside" (tune as needed)

    def is_gripper_open(self, arm='left'):
        """Return True if the specified arm's gripper is open."""
        obs = self.get_obs()
        gripper_key = f"{arm}_gripper_state" if f"{arm}_gripper_state" in obs else "gripper_state"
        # Convention: open = 1, closed = 0
        return obs.get(gripper_key, 1) > 0.5

    def is_gripper_closed(self, arm='left'):
        """Return True if the specified arm's gripper is closed."""
        return not self.is_gripper_open(arm)

    def get_grasped_object(self, arm='left'):
        """Return the object currently grasped by the specified arm (if any)."""
        return self.grasped_object

    def check_actors_contact(self, object_a, object_b):
        """Check if two objects/actors are in physical contact."""
        return self.task.check_actors_contact(object_a, object_b)

    def get_gripper_contact_positions(self, object_name):
        """Return all contact positions between the gripper and the specified object."""
        return self.task.get_gripper_actor_contact_position(object_name)

    # === 4. GRASP & PLACEMENT UTILITIES ===
    
    def choose_grasp_pose(self, object_name, arm='left', pre_dis=0.1, contact_point_id=None):
        """Choose the best grasp pose for an object (optionally for a contact point)."""
        actor = self._get_actor(object_name)
        if actor is None:
            return None, None
        return self.task.choose_grasp_pose(actor, arm, pre_dis=pre_dis, contact_point_id=contact_point_id)

    def get_place_pose(self, object_name, arm='left', target_pose=None, functional_point_id=None):
        """Compute the ideal placement pose for the object."""
        actor = self._get_actor(object_name)
        if actor is None:
            return None
        return self.task.get_place_pose(actor, arm, target_pose=target_pose, functional_point_id=functional_point_id)

    def grasp_actor(self, object_name, arm='left', **kwargs):
        """Perform a grasp action using contact point information."""
        actor = self._get_actor(object_name)
        if actor is None:
            return None, []
        return self.task.grasp_actor(actor, arm, **kwargs)

    def place_actor(self, object_name, arm='left', target_pose=None, **kwargs):
        """Perform a place action using contact point and placement pose."""
        actor = self._get_actor(object_name)
        if actor is None:
            return None, []
        return self.task.place_actor(actor, arm, target_pose=target_pose, **kwargs)

    # === 5. MOTION & MANIPULATION PRIMITIVES ===
    def move_to_pose(self, arm, pose):
        """Move the specified arm's end effector to a desired pose (no collision avoidance)."""
        return self.task.move_to_pose(arm, pose)

    def move_both_arms_to_pose(self, left_pose, right_pose):
        """Move both arms to the specified poses simultaneously (dual-arm)."""
        # Call together_move_to_pose and return actions for both arms
        return self.task.together_move_to_pose(left_pose, right_pose)

    def move_by_displacement(self, arm, dx=0, dy=0, dz=0, quat=None, axis="world"):
        """Move the specified arm by a relative displacement."""
        return self.task.move_by_displacement(arm, dx, dy, dz, quat, axis)

    def open_gripper(self, arm='left', pos=1.0):
        """Open the specified gripper."""
        return self.task.open_gripper(arm, pos)

    def close_gripper(self, arm='left', pos=0.0):
        """Close the specified gripper."""
        return self.task.close_gripper(arm, pos)

    def grasp(self, object_name, arm='left'):
        """Grasp the specified object with the specified arm (plan + close gripper)."""
        actor = self._get_actor(object_name)
        if actor is None:
            return None, []
        return self.task.grasp_actor(actor, arm)

    def place(self, object_name, target_pose, arm='left'):
        """Place a grasped object at the desired target pose."""
        actor = self._get_actor(object_name)
        if actor is None:
            return None, []
        return self.task.place_actor(actor, arm, target_pose)

    def back_to_origin(self, arm='left'):
        """Return the specified arm to its origin (home) pose."""
        return self.task.back_to_origin(arm)

    # === 6. PLANNING & TRAJECTORIES ===
    def plan_joint_trajectory(self, arm, target_pose):
        """
        Plan a joint trajectory for one or both arms.
        - arm: "left", "right", or "together"
        - target_pose: pose or (left_pose, right_pose) if together
        Returns: trajectory or (left_trajectory, right_trajectory)
        """
        if arm == "left":
            return self.task.left_move_to_pose(target_pose)
        elif arm == "right":
            return self.task.right_move_to_pose(target_pose)
        elif arm == "together":
            left_pose, right_pose = target_pose
            left_traj = self.task.left_move_to_pose(left_pose)
            right_traj = self.task.right_move_to_pose(right_pose)
            return left_traj, right_traj
        else:
            raise ValueError(f"Unknown arm argument: {arm}")

    def execute_joint_trajectory(self, arm, trajectory):
        """
        Execute a joint trajectory for the specified arm(s).
        - arm: "left", "right", or "together"
        - trajectory: trajectory dict or (left_traj, right_traj) if together
        """
        control_seq = {
            "left_arm": None,
            "left_gripper": None,
            "right_arm": None,
            "right_gripper": None,
        }
        if arm == "left":
            control_seq["left_arm"] = trajectory
        elif arm == "right":
            control_seq["right_arm"] = trajectory
        elif arm == "together":
            left_traj, right_traj = trajectory
            control_seq["left_arm"] = left_traj
            control_seq["right_arm"] = right_traj
        else:
            raise ValueError(f"Unknown arm argument: {arm}")
        return self.task.take_dense_action(control_seq)

    def set_path_lst(self, args):
        """
        Set path list for joint planning.
        """
        self.task.set_path_lst(args)


    def take_action(self, action, action_type='qpos'):
        """
        Take low-level action (joint or end-effector).
        """
        return self.task.take_action(action, action_type=action_type)


    def take_dense_action(self, control_seq, save_freq=-1):
        """
        Take a dense sequence of actions (arms and/or grippers).
        """
        return self.task.take_dense_action(control_seq, save_freq=save_freq)

    # === 7. CLUTTER, SCENE & CONSTRAINTS ===
    def get_cluttered_table(self):
        """Place random objects on the table (if supported)."""
        return self.task.get_cluttered_table()

    def add_prohibited_area(self, area_spec):
        """Add a prohibited area (for collision avoidance or planning constraints)."""
        return self.task.add_prohibit_area(area_spec)

    # === 8. CAMERA & DATA COLLECTION ===
    def save_camera_rgb(self, file_path, camera_name='head_camera'):
        """Save an RGB image from a specified camera to a file."""
        return self.task.save_camera_rgb(file_path, camera_name=camera_name)

    def save_camera_images(self, task_name, step_name, id_num, save_dir="./camera_images"):
        """Save images from all or key cameras for a task step."""
        return self.task.save_camera_images(task_name, step_name, id_num, save_dir=save_dir)

    # === 9. TRAJECTORY DATA/LOGGING UTILITIES ===

    def save_traj_data(self, idx):
        """Save the joint trajectory for current episode."""
        # Delegates to Base_Task.save_traj_data
        self.task.save_traj_data(idx)

    def load_tran_data(self, idx):
        """Load a previously saved joint trajectory."""
        # Delegates to Base_Task.load_tran_data (typo in 'tran' in base)
        return self.task.load_tran_data(idx)

    def merge_pkl_to_hdf5_video(self):
        """Merge per-step PKL data to a single HDF5+video file (for dataset)."""
        self.task.merge_pkl_to_hdf5_video()

    def remove_data_cache(self):
        """Remove cached episode data (cleanup)."""
        self.task.remove_data_cache()

    # === 10. INSTRUCTION & EPISODE CONTROL ===

    def set_instruction(self, instruction):
        """Set a high-level instruction or task description."""
        self.task.set_instruction(instruction)

    def get_instruction(self):
        """Get the current instruction or task description."""
        return self.task.get_instruction()

    def check_success(self):
        """Check if the current task/episode is successful (goal reached)."""
        return self.task.check_success()

    def play_once(self):
        """Execute one episode of a task (if available as a demo function)."""
        return self.task.play_once()
