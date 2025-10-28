from interfaces.base_world_interface import BaseWorldInterface
from collections import defaultdict
import numpy as np

# robotwin imports
from envs._base_task import Base_Task
from envs import *
from script.collect_data import class_decorator, get_camera_config
from envs.utils.action import ArmTag, Action
import yaml
import os
import traceback


TASK_NAME = "pick_obj"   # This is known. Basically we can get from runtime file

class WorldInterface(BaseWorldInterface): # Methods with same name use the Base_Task implementation
    def __init__(self, task_name=TASK_NAME, task_config="demo_clean", seed=0, gripper_bias=0.16, movable_objects=None, graspable_objects=None, table_offset=0):
        super().__init__(
            cfree_interface=None,
            movable_objects=movable_objects,
            graspable_objects=graspable_objects,
            table_offset=table_offset,
        )

        self.demo = class_decorator(task_name)
        self.grasped_object = None
        self.manipulation_target = None

        # Create robotwin environment
        self.args = self.create_args(task_name, task_config, seed, gripper_bias)
        for i in range(seed, seed + 3):
            try:
                self.args["seed"] = seed
                self.demo.setup_demo(**self.args)
                break
            except Exception as e:
                if i == seed + 2:
                    continue
                print(f"Failed to create demo for task {task_name}: {e}")
                traceback.print_exc()
                try:
                    if self.args["render_freq"]:
                        self.demo.close_env()
                        self.demo.viewer.close()
                except:
                    pass
                continue
            
        # self.run_demo()
        self.actors = self.demo.scene.get_all_actors()
        self._build_name_to_actor()
        self.beat_count = defaultdict(int)
        self.toggled = defaultdict(bool)

        for actor in self.actors:
            self.toggled[actor.get_name()] = False

    def create_args(self, task_name, task_config="demo_clean", seed=0, gripper_bias=0.16):
        
        # task = class_decorator(task_name)
        config_path = f"./task_config/{task_config}.yml"

        with open(config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        args['task_name'] = task_name

        embodiment_type = args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise "missing embodiment files"
            return robot_file

        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise "number of embodiment config parameters should be 1 or 3"

        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"], gripper_bias=gripper_bias)
        args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"], gripper_bias=gripper_bias)

        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

        # show config
        print("============= Config =============\n")
        print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
        print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
        if args["domain_randomization"]["random_background"]:
            print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
        print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
        if args["domain_randomization"]["random_light"]:
            print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
        print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
        print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

        print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
            str(args["camera"]["collect_head_camera"]))
        print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
            str(args["camera"]["collect_wrist_camera"]))
        print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
        print("\n==================================")

        args["embodiment_name"] = embodiment_name
        args['task_config'] = task_config
        args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
        args["seed"] = seed
        return args    

    def run_demo(self):
        """
        Run one episode of the task demo and return success status.
        """

        print(f"Running task: {self.task_name}")
        success = False
        self.demo.play_once() # if you want to run the task
        if self.demo.plan_success and self.demo.check_success():
            print(f"simulate {self.task_name} success!")
            success = True
        else:
            print(f"simulate data episode fail!")

        # if self.args["render_freq"]:
        #     while not self.viewer.closed:
        #         self.scene.step()
        #         self.scene.update_render()
        #         self.viewer.render()

        #     self.close_env()
        #     self.viewer.close()
        
        return success

    # Utility to build a mapping from object names to actors
    def _build_name_to_actor(self):
        self._name_to_actor = defaultdict(list)
        for actor in self.scene.get_all_actors():  # Sapien API to get all actors
            name = actor.get_name()
            if name and name not in ['table', 'wall', 'ground']:
                self._name_to_actor[name].append(actor)

    # Utility to get an actor by name
    def _get_actor(self, object_name):
        for actor in self.actors:
            if actor.get_name() == object_name:
                return actor
        return None
    
    # === 1. ENVIRONMENT CONTROL ===
    def reset(self, config_kwargs=None):
        """
        Reset the world/simulation for a new episode.
        """
        # Re-initialize the task/environment
        if config_kwargs is None:
            config_kwargs = {}
        self._init_task_env_(**config_kwargs)
        self.grasped_object = None
        self.manipulation_target = None

    # def close_env(self, clear_cache=False):
    #     """
    #     Cleanup and close the environment.
    #     """
    #     self.close_env(clear_cache=clear_cache)

    # def delay(self, time_steps):
    #     """
    #     Hold current state for some steps (simulate waiting).
    #     """
    #     self.delay(time_steps)

    # === 2. STATE FEEDBACK & OBSERVATION ===

    def get_feedback(self):
        """Update state from sensors/scene and refresh object locations."""
        self.get_obs()

    # def get_obs(self):
    #     """Get the current observation (images, pointclouds, joint states, end-effector pose)."""
    #     return self.get_obs()

    def get_scene_graph(self):
        """Return a scene graph or logical structure of the current world state."""
        # _base_task.py does not explicitly have a scene graph, but we can use now_obs as a base
        # for now, return the current observation dictionary.
        return getattr(self, 'now_obs', {})

    # def get_scene_contact(self):
    #     """Debug: print or return contacts in the scene."""
    #     return self.get_scene_contact()

     # === 3. OBJECT & GRIPPER STATE QUERIES ===
    def get_object_pose(self, object_name):
        """Return the pose (position and orientation) of an object in the scene."""
        for actor in self.actors:
            if actor.get_name() == object_name:
                pose = actor.get_pose()
                return pose
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

    # def get_arm_pose(self, arm='left'):
    #     """Alias for get_robot_pose."""
    #     return self.get_robot_pose(arm)

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

    def is_grasped(self, arm_tag="any", object_name='"any object"'):
        """Check if the specified object is currently grasped."""
        object_pose = self.get_object_pose(object_name).p
        contact = self.get_gripper_actor_contact_position(object_name)
        return (object_pose[2] > 0.8 and len(contact) > 0)
    
    def object_at(self, target_object, relation, relative_object):
        """ Check if object is at a specific location """
        if target_object == '"any object"':
            for obj in self.get_all_objects():
                if self.object_at(obj.get_name(), relation, relative_object):
                    return True
            return False

        target_object_pose = self.get_object_pose(target_object)
        relative_object_pose = self.get_object_pose(relative_object)

        if relation == "on":
            object_pose = target_object_pose.p
            scale_pose = relative_object_pose.p
            distance_threshold = 0.035
            distance = np.linalg.norm(np.array(scale_pose[:2]) - np.array(object_pose[:2]))
            return (distance < distance_threshold and object_pose[2] > (scale_pose[2] - 0.01))

        elif relation == "inside":
            return np.sum(np.sqrt((target_object_pose.p - relative_object_pose.p)**2)) < 0.15

        elif relation == "to_left_of" or relation == "to_right_of":
            relative_pose = relative_object_pose.p.tolist()
            relative_pose[0] -= 0.13 if relation == "to_left_of" else 0.13
            distance = np.sqrt(np.sum((target_object_pose[:2] - relative_pose[:2])**2))
            return np.all(distance < 0.2 and distance > 0.08 and target_object_pose[0] < relative_pose[0]
                        and abs(target_object_pose[1] - relative_pose[1]) < 0.05)
        
    def get_all_objects(self):
        return self.scene.get_all_actors()
    
    def get_relation(self, target_object, relative_object):
        target_object_pose = self.get_object_pose(target_object)
        relative_object_pose = self.get_object_pose(relative_object)
        target_object_pose_p = target_object_pose.p
        relative_object_pose_p = relative_object_pose.p

        distance_threshold = 0.035
        distance = np.linalg.norm(np.array(relative_object_pose_p[:2]) - np.array(target_object_pose_p[:2]))
        if (distance < distance_threshold and target_object_pose_p[2] > (relative_object_pose_p[2] - 0.01)):
            return "on"

        if np.sum(np.sqrt((target_object_pose_p - relative_object_pose_p)**2)) < 0.15:
            return "inside"

        relative_pose = relative_object_pose.p.tolist()

        distance = np.sqrt(np.sum((target_object_pose[:2] - relative_pose[:2])**2))
        if np.all(distance < 0.2 and distance > 0.08 and target_object_pose[0] < 0.13
                    and distance < 0.2 and distance > 0.08 and target_object_pose[0] > 0
                    and abs(target_object_pose[1] - relative_pose[1]) < 0.05):
            return "to_left_of"
        if np.all(distance < 0.2 and distance > 0.08 and target_object_pose[0] > -0.13
                    and distance < 0.2 and distance > 0.08 and target_object_pose[0] < 0
                    and abs(target_object_pose[1] - relative_pose[1]) < 0.05):
            return "to_right_of"
        
        return None
    
    def is_toggled(self, object_name):
        return self.toggled[object_name]

    def toggle_check(self, object_name):
        """Check if the specified object is toggled (active)."""
        actor = self._get_actor(object_name)

        if object_name == "056_switch":
            limit = self.actor.get_qlimits()[0]
            return self.actor.get_qpos()[0] >= limit[1] - 0.05

        actor_pose = actor.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position(object_name)
        eps = [0.028, 0.028]
        for position in positions:
            if (np.all(np.abs(position[:2] - actor_pose[:2]) < eps) and abs(position[2] - actor_pose[2]) < 0.03):
                self.stage_success_tag = True
                return True
        return False
    
    def is_beaten(self, object_name, tool, count=1):
        """Check if the specified object has been struck by the tool a certain number of times."""
        if (object_name, tool) not in self.beat_count.keys():
            return False
        return self.beat_count[(object_name, tool)] >= count
    
    def beat_check(self, object_name, tool):
        """Check if the specified object has been struck by the tool."""
        actor = self._get_actor(object_name)
        tool_actor = self._get_actor(tool)
        if hasattr(tool_actor, 'get_functional_point'):
            tool_target_pose = tool_actor.get_functional_point(0, "pose").p
        else:
            tool_target_pose = tool_actor.get_pose().p
        if hasattr(actor, 'get_functional_point'):
            block_pose = actor.get_functional_point(1, "pose").p
        else:
            block_pose = actor.get_pose().p
        eps = np.array([0.02, 0.02])
        return np.all(abs(tool_target_pose[:2] - block_pose[:2]) < eps) and self.check_actors_contact(
            tool_actor.get_name(), actor.get_name())
    
    def pick(self, target_object, arm_tag='any'):
        """ Pick up the target object using the specified arm. """
        # Determine which arm to use based on object's x position
        target_object = self._get_actor(self.parameters["object"])
        if arm_tag == "any":
            arm_tag = ArmTag("right" if target_object.get_pose().p[0] > 0 else "left")
        else:
            arm_tag = ArmTag(arm_tag)

        # Grasp the object with specified arm
        self.move(self.grasp_actor(target_object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        # Lift the object upward by 0.1 meters along z-axis using arm movement
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

    def place(self, target_object, relation, relative_object, arm_tag='any'):
        """ Place the target object at the specified relation to the relative object using the specified arm. """
        target_object = self._get_actor(target_object)
        relative_object = self._get_actor(relative_object)

        if arm_tag == "any":
            arm_tag = ArmTag("left" if self.is_grasped("left", target_object) else "right")
        else:
            arm_tag = ArmTag(arm_tag)

        if relation == "away":
            # Move the object horizontally (right if right arm, left if left arm)
            self.move(self.move_by_displacement(arm_tag, x=0.3 if arm_tag == "right" else -0.3))
            # Open gripper to release the object
            self.move(self.open_gripper(arm_tag))
        
        elif relation in ["to_left_of", "to_right_of"]:
            target_pose = relative_object.get_pose().p.tolist()
            target_pose[0] += 0.13 if relation == "to_right_of" else -0.13

            # Place the object at the adjusted target position
            self.move(self.place_actor(target_object, arm_tag=arm_tag, target_pose=target_pose))

        elif relation == "on":
            # Get the target pose from display stand's functional point
            if hasattr(relative_object, 'get_functional_point'):
                target_pose = relative_object.get_functional_point(0)
            else:
                target_pose = relative_object.get_pose()

            # Place the object onto the display stand with free constraint
            self.move(
                self.place_actor(
                    target_object,
                    arm_tag=arm_tag,
                    target_pose=target_pose,
                    constrain="free",
                    pre_dis=0.07,
                ))
            
        elif relation == "inside":
            # Get functional points of basket for placing
            f0 = np.array(relative_object.get_functional_point(0))
            f1 = np.array(relative_object.get_functional_point(1))
            place_pose = (f0 if np.linalg.norm(f0[:2] - target_object.get_pose().p[:2])
                        < np.linalg.norm(f1[:2] - target_object.get_pose().p[:2]) else f1)
            place_pose[:2] = f0[:2] if place_pose is f0 else f1[:2]
            place_pose[3:] = (-1, 0, 0, 0) if arm_tag == "left" else (0.05, 0, 0, 0.99)

            # Place the toy car in the basket
            self.move(self.place_actor(
                target_object,
                arm_tag=arm_tag,
                target_pose=place_pose,
                dis=0.02,
                is_open=False,
            ))

            if not self.plan_success:
                self.plan_success = True  # Try new way
                # Move up and away (recovery motion when plan fails)
                place_pose[0] += -0.15 if arm_tag == "left" else 0.15
                place_pose[2] += 0.15
                self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=place_pose))

                # Lower down (recovery motion when plan fails)
                place_pose[2] -= 0.05
                self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=place_pose))

            # Open gripper to release object
            self.move(self.open_gripper(arm_tag=arm_tag))

    def toggle(self, object_name, arm_tag='any'):
        """ Toggle the specified object using the specified arm. """
        target_object = self._get_actor(object_name)
        if target_object is None:
            return

        # Move the gripper above the top center of the alarm clock and close the gripper to simulate a click
        # Note: although the code structure resembles a grasp, it is used here to simulate a touch/click action
        # You can adjust API parameters to move above the top button and close the gripper (similar to grasp_actor)
        self.move((
            ArmTag(arm_tag),
            [
                Action(
                    arm_tag,
                    "move",
                    self.get_grasp_pose(target_object, pre_dis=0.1, contact_point_id=0, arm_tag=arm_tag)[:3] +
                    [0.5, -0.5, 0.5, 0.5],
                ),
                Action(arm_tag, "close", target_gripper_pos=0.0),
            ],
        ))
    
        # Move the gripper downward to press the top button of the alarm clock
        self.move(self.move_by_displacement(arm_tag, z=-0.065))

        if self.toggle_check(object_name):
            self.toggled[object_name] = not self.toggled[object_name]

            # Move the gripper back to the original height (not lifting the alarm clock)
            self.move(self.move_by_displacement(arm_tag, z=0.065))

    def beat(self, tool, object_name, arm_tag='any'):
        """ Strike the specified object with the tool using the specified arm. """
        target_object = self._get_actor(object_name)
        target_pose = target_object.get_functional_point(1, "pose") if hasattr(target_object, 'get_functional_point') else target_object.get_pose()
        tool_actor = self._get_actor(tool)
        if target_object is None or tool_actor is None:
            return

        if arm_tag == "any":
            arm_tag = ArmTag("right" if self.is_grasped("right", tool_actor) else "left")
        else:
            arm_tag = ArmTag(arm_tag)

        # Move the tool above the target object
        self.move(
            self.place_actor(
                tool_actor,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
            ))
        
        if self.beat_check(object_name, tool):
            if (object_name, tool) not in self.beat_count.keys():
                self.beat_count[(object_name, tool)] = 1
            else:
                self.beat_count[(object_name, tool)] += 1

            # Lift the tool upward after striking
            self.move(self.move_by_displacement(arm_tag, z=0.065))

    # def check_actors_contact(self, object_a, object_b):
    #     """Check if two objects/actors are in physical contact."""
    #     return self.check_actors_contact(object_a, object_b)

    # def get_gripper_contact_positions(self, object_name):
    #     """Return all contact positions between the gripper and the specified object."""
    #     return self.get_gripper_actor_contact_position(object_name)

    # === 4. GRASP & PLACEMENT UTILITIES ===
    
    # def choose_grasp_pose(self, object_name, arm='left', pre_dis=0.1, contact_point_id=None):
    #     """Choose the best grasp pose for an object (optionally for a contact point)."""
    #     actor = self._get_actor(object_name)
    #     if actor is None:
    #         return None, None
    #     return self.choose_grasp_pose(actor, arm, pre_dis=pre_dis, contact_point_id=contact_point_id)

    # def get_place_pose(self, object_name, arm='left', target_pose=None, functional_point_id=None):
    #     """Compute the ideal placement pose for the object."""
    #     actor = self._get_actor(object_name)
    #     if actor is None:
    #         return None
    #     return self.get_place_pose(actor, arm, target_pose=target_pose, functional_point_id=functional_point_id)

    # def grasp_actor(self, object_name, arm='left', **kwargs):
    #     """Perform a grasp action using contact point information."""
    #     actor = self._get_actor(object_name)
    #     if actor is None:
    #         return None, []
    #     return self.grasp_actor(actor, arm, **kwargs)

    # def place_actor(self, object_name, arm='left', target_pose=None, **kwargs):
    #     """Perform a place action using contact point and placement pose."""
    #     actor = self._get_actor(object_name)
    #     if actor is None:
    #         return None, []
    #     return self.place_actor(actor, arm, target_pose=target_pose, **kwargs)

    # === 5. MOTION & MANIPULATION PRIMITIVES ===
    # def move_to_pose(self, arm, pose):
    #     """Move the specified arm's end effector to a desired pose (no collision avoidance)."""
    #     return self.move_to_pose(arm, pose)

    # def move_both_arms_to_pose(self, left_pose, right_pose):
    #     """Move both arms to the specified poses simultaneously (dual-arm)."""
    #     # Call together_move_to_pose and return actions for both arms
    #     return self.together_move_to_pose(left_pose, right_pose)

    # def move_by_displacement(self, arm, dx=0, dy=0, dz=0, quat=None, axis="world"):
    #     """Move the specified arm by a relative displacement."""
    #     return self.move_by_displacement(arm, dx, dy, dz, quat, axis)

    # def open_gripper(self, arm='left', pos=1.0):
    #     """Open the specified gripper."""
    #     return self.open_gripper(arm, pos)

    # def close_gripper(self, arm='left', pos=0.0):
    #     """Close the specified gripper."""
    #     return self.close_gripper(arm, pos)

    # def grasp(self, object_name, arm='left'):
    #     """Grasp the specified object with the specified arm (plan + close gripper)."""
    #     actor = self._get_actor(object_name)
    #     if actor is None:
    #         return None, []
    #     return self.grasp_actor(actor, arm)

    # def place(self, object_name, target_pose, arm='left'):
    #     """Place a grasped object at the desired target pose."""
    #     actor = self._get_actor(object_name)
    #     if actor is None:
    #         return None, []
    #     return self.place_actor(actor, arm, target_pose)

    # def back_to_origin(self, arm='left'):
    #     """Return the specified arm to its origin (home) pose."""
    #     return self.back_to_origin(arm)

    # === 6. PLANNING & TRAJECTORIES ===
    def plan_joint_trajectory(self, arm, target_pose):
        """
        Plan a joint trajectory for one or both arms.
        - arm: "left", "right", or "together"
        - target_pose: pose or (left_pose, right_pose) if together
        Returns: trajectory or (left_trajectory, right_trajectory)
        """
        if arm == "left":
            return self.left_move_to_pose(target_pose)
        elif arm == "right":
            return self.right_move_to_pose(target_pose)
        elif arm == "together":
            left_pose, right_pose = target_pose
            left_traj = self.left_move_to_pose(left_pose)
            right_traj = self.right_move_to_pose(right_pose)
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
        return self.take_dense_action(control_seq)

    # def set_path_lst(self, args):
    #     """
    #     Set path list for joint planning.
    #     """
    #     self.set_path_lst(args)


    # def take_action(self, action, action_type='qpos'):
    #     """
    #     Take low-level action (joint or end-effector).
    #     """
    #     return self.take_action(action, action_type=action_type)


    # def take_dense_action(self, control_seq, save_freq=-1):
    #     """
    #     Take a dense sequence of actions (arms and/or grippers).
    #     """
    #     return self.take_dense_action(control_seq, save_freq=save_freq)

    # === 7. CLUTTER, SCENE & CONSTRAINTS ===
    # def get_cluttered_table(self):
    #     """Place random objects on the table (if supported)."""
    #     return self.get_cluttered_table()

    # def add_prohibited_area(self, area_spec):
    #     """Add a prohibited area (for collision avoidance or planning constraints)."""
    #     return self.add_prohibit_area(area_spec)

    # # === 8. CAMERA & DATA COLLECTION ===
    # def save_camera_rgb(self, file_path, camera_name='head_camera'):
    #     """Save an RGB image from a specified camera to a file."""
    #     return self.save_camera_rgb(file_path, camera_name=camera_name)

    # def save_camera_images(self, task_name, step_name, id_num, save_dir="./camera_images"):
    #     """Save images from all or key cameras for a task step."""
    #     return self.save_camera_images(task_name, step_name, id_num, save_dir=save_dir)

    # === 9. TRAJECTORY DATA/LOGGING UTILITIES ===

    # def save_traj_data(self, idx):
    #     """Save the joint trajectory for current episode."""
    #     # Delegates to Base_Task.save_traj_data
    #     self.save_traj_data(idx)

    # def load_tran_data(self, idx):
    #     """Load a previously saved joint trajectory."""
    #     # Delegates to Base_Task.load_tran_data (typo in 'tran' in base)
    #     return self.load_tran_data(idx)

    # def merge_pkl_to_hdf5_video(self):
    #     """Merge per-step PKL data to a single HDF5+video file (for dataset)."""
    #     self.merge_pkl_to_hdf5_video()

    # def remove_data_cache(self):
    #     """Remove cached episode data (cleanup)."""
    #     self.remove_data_cache()

    # === 10. INSTRUCTION & EPISODE CONTROL ===

    # def set_instruction(self, instruction):
    #     """Set a high-level instruction or task description."""
    #     self.set_instruction(instruction)

    # def get_instruction(self):
    #     """Get the current instruction or task description."""
    #     return self.get_instruction()

    # def check_success(self):
    #     """Check if the current task/episode is successful (goal reached)."""
    #     return self.check_success()

    # def play_once(self):
    #     """Execute one episode of a task (if available as a demo function)."""
    #     return self.play_once()
