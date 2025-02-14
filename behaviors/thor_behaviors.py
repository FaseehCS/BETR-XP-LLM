""" A module containing all the behaviors the robot can use inside ai2thor environment. """
from enum import IntEnum
import numpy as np
from behaviors.common_behaviors import Behavior, ActionBehavior, VLMPrompter
import behaviors.common_behaviors
import py_trees as pt

from interfaces.thor_world_interface import WorldInterface

def get_node(node_descriptor, world_interface, verbose = False):
    """ Returns a node object given the descriptor string """
    return behaviors.common_behaviors.get_node(node_descriptor, world_interface, verbose=verbose)

def compatible(_condition1, _condition2):
    """ TODO this is just temp to get it to run, needs to be fixed if we want priorities to work"""
    return True

def extract_name(object_id):
    """ Extract the name of the object from the object id """
    if object_id is None:
        return "None"
    return object_id.split("|")[0]

class AtPos(Behavior):
    """
    Check if object is at position
    """
    skill_name = "At_Pos"
    description = "Check if object is at position e.g. At_Pos(Cup, on, countertop), At_Pos(Cup, inside, microwave), etc. - Relation: 'on' indicates placement on a surface. - Relation: 'inside' indicates containment within an object. For containers, if AtPos(target, relation='inside', relative_object) exists, the container is occupied."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = AtPos.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    @staticmethod
    def check_string_match(node_string, parameters):
        """ Check if the string description of the node matches this node """
        for relation in parameters["relation"].list_of_values:
            if " " + relation + " " in node_string:
                return True
        return False

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["target_object"])
        node_string += " " + parameters["relation"]
        node_string += " " + extract_name(parameters["relative_object"])
        node_string += "?"
        return Behavior.common_string_rules(node_string, parameters)

    @staticmethod
    def parse_parameters(node_descriptor):
        """ Parse behavior parameters from string """
        parameters = {}
        n_marks = 4
        marks = [0] * n_marks
        marks[0] = node_descriptor.find('"')
        for i in range(1, n_marks):
            if marks[i - 1] >= 0:
                marks[i] = node_descriptor.find('"', marks[i - 1] + 1)
            else:
                print("Error, parameter parsing failed")
                return None

        parameters["target_object"] = node_descriptor[marks[0]: marks[1] + 1]
        parameters["relation"] = node_descriptor[marks[1] + 2: marks[2] - 1]
        parameters["relative_object"] = node_descriptor[marks[2]: marks[3] + 1]
        parameters["not"] = node_descriptor[0] == "~"

        return parameters

    def __eq__(self, other) -> bool:
        if not isinstance(other, AtPos):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    def update(self):
        target_object = self.parameters["target_object"]
        relation = self.parameters["relation"]
        relative_object = self.parameters["relative_object"]
        if target_object == '"any object"':
            object_at = False
            for movable_object in self.world_interface.movable_objects:
                if self.world_interface.object_at(movable_object, relation, relative_object):
                    object_at = True
                    break
            return self.check_negated(object_at)
        else:
            return self.check_negated(self.world_interface.object_at(target_object, relation, relative_object))

class Grasped(Behavior):
    """
    Check if object is grasped
    """
    skill_name = "grasped"
    description = "Check if object is in robot's gripper. If the gripper is invisible the object will be in the bottom of the image. The object will be floating and zoomed in. E.g. grasped(Cup),  grasped(Apple), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Grasped.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grasped):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = 'grasped ' + extract_name(parameters["target_object"])
        node_string += "?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        target_object = self.parameters["target_object"]
        if target_object == '"any object"':
            return self.check_negated(self.world_interface.get_grasped_object() is not None)
        else:
            return self.check_negated(self.world_interface.get_grasped_object() == target_object)

class LocationKnown(Behavior):
    """
    Check if object location is known
    """
    skill_name = "Location_Known"
    description = "Check if object location is known. This includes the objects that are clearly visible to the robot and the objects that are not visible but we know their locations and navigable. E.g. Location_Known(Cup), Location_Known(Apple), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = LocationKnown.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, LocationKnown):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["target_object"]) + " location known?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.object_position_known[self.parameters["target_object"]])

class Upright(Behavior):
    """
    Check if object is standing upright
    """
    skill_name = "Upright"
    description = "Check if object is standing upright"

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Upright.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Upright):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = parameters["target_object"] + " upright?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_object_upright(self.parameters["target_object"]))

class NearRobot(Behavior):
    """
    Check if object is within reach
    """
    skill_name = "Near_Robot"
    description = "Check if object is within reach. This includes the objects that are directly infront of the robot and are visible to the robot. E.g. Near_Robot(Cup), Near_Robot(Apple), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = NearRobot.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, NearRobot):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["destination"]) + " near robot?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_near_robot(self.parameters["destination"]))

class Opened(Behavior):
    """
    Check if object is open
    """
    skill_name = "Opened"
    description = "Check if object is opened e.g. Opened(MicrowaveDoor), Opened(FridgeDoor), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Opened.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Opened):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["interact_object"]) + " opened?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_opened(self.parameters["interact_object"]))

class Toggled(Behavior):
    """
    Check if object is open
    """
    skill_name = "Toggled"
    description = "Check if object is turned on/off e.g. Toggled(Microwave), Toggled(CoffeeMachine), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Toggled.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Toggled):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["interact_object"]) + " toggled on?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_toggled(self.parameters["interact_object"]))

class Unlocked(Behavior):
    """
    Check if object is unlocked
    """
    skill_name = "Unlocked"
    description = "Check if object is unlocked e.g. Unlocked(Safe), Unlocked(Box), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Unlocked.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Unlocked):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["interact_object"]) + " unlocked?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_unlocked(self.parameters["interact_object"]))
    
class Sliced(Behavior):
    """
    Check if object is sliced
    """
    skill_name = "Sliced"
    description = "Check if object is sliced e.g. Sliced(Apple), Sliced(Carrot), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Sliced.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Sliced):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["interact_object"]) + " sliced?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_sliced(self.parameters["interact_object"]))
    
class Cracked(Behavior):
    """
    Check if object is cracked
    """
    skill_name = "Cracked"
    description = "Check if object is cracked e.g. Cracked(Egg), Cracked(Nut), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Cracked.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cracked):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["interact_object"]) + " cracked?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_cracked(self.parameters["interact_object"]))

class Cleaned(Behavior):
    """
    Check if object is Clean
    """
    skill_name = "Cleaned"
    description = "Check if object is clean e.g. clean(Mug), clean(Bowl), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Cleaned.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cleaned):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["interact_object"]) + " Clean?"
        return Behavior.common_string_rules(node_string, parameters)

    def update(self):
        return self.check_negated(self.world_interface.is_clean(self.parameters["interact_object"]))

class Filled(Behavior):
    """
    Check if object is filled
    """
    skill_name = "Filled"
    description = "Check if object is filled with liquid e.g. Filled(Cup), Filled(Bowl), etc."

    def __init__(self, name, parameters, world_interface, _verbose=False):
        name = Filled.to_string(parameters)
        super().__init__(name, parameters, world_interface)
        
    def __eq__(self, other) -> bool:
        if not isinstance(other, Filled):
            # don't attempt to compare against unrelated types
            return False
        return super().__eq__(other)
    
    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = extract_name(parameters["interact_object"]) + " filled?"
        return Behavior.common_string_rules(node_string, parameters)
    
    def update(self):
        return self.check_negated(self.world_interface.is_filled(self.parameters["interact_object"], self.parameters["liquid"]))

class Grasp(ActionBehavior):
    """
    Grasp an object
    """
    skill_name = "Pick"
    description = "Picks up a specified object."

    class GraspStates(IntEnum):
        """Define the internal states during execution."""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4

    def __init__(self, name, parameters, world_interface: WorldInterface, vlm, verbose=False):
        self.world_interface = world_interface
        name = Grasp.to_string(parameters)
        self.target_object = None
        self.grasp_position = None
        self.approach_position = None
        self.orientation = None
        self.internal_state = self.GraspStates.INIT
        self.full_grasping_program = ''
        preconditions = [Grasped('', {"not": True, "target_object": '"any object"'}, world_interface),
                         NearRobot('', {"destination": parameters["target_object"]}, world_interface)]
        if world_interface.is_graspable(parameters["target_object"]) or parameters["target_object"] == '"any object"':
            postconditions = [Grasped('', {"target_object": parameters["target_object"]}, world_interface)]
            for edge in world_interface.scene_graph.edges.keys():
                if self.target_object in edge and not "robot_gripper" in edge:
                    relation = world_interface.scene_graph.edges[edge].edge_type
                    relative_object = edge[1] if edge[0] == self.target_object else edge[0]
                    postconditions += [AtPos('', {"not": True,
                                                "target_object": parameters["target_object"],
                                                "relation": relation,
                                                "relative_object": relative_object}, world_interface)]
        # else:
        #     postconditions = [Grasped('', {"target_object": parameters["target_object"]}, world_interface)]
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "grasp " + extract_name(parameters["target_object"])
        # relation = parameters.get("relation")
        # relative_object = extract_name(parameters.get("relative_object"))
        # if relation is not None and relative_object is not None:
        #     node_string += " from " + relation
        #     node_string += " " + relative_object
        node_string += "!"
        return node_string

    @staticmethod
    def check_string_match(node_string, _parameters):
        """ Check if the string description of the node matches this node """
        if "grasp " in node_string:
            return True
        return False

    def initialise(self):
        self.internal_state = self.GraspStates.INIT
        self.target_object = self.find_target_object()
        ActionBehavior.initialise(self)

    @staticmethod
    def parse_parameters(node_descriptor):
        """ Parse behavior parameters from string """
        parameters = {}
        n_marks = 4
        marks = []
        marks.append(node_descriptor.find('"'))
        for i in range(1, n_marks):
            if marks[i-1] >= 0:
                marks.append(node_descriptor.find('"', marks[i-1] + 1))
            else:
                break

        if len(marks) < 2:
            print("Error, parameter parsing failed")
            return None
        
        parameters["target_object"] = node_descriptor[marks[0]: marks[1] + 1]
        if len(marks) >= 4:
            parameters["relation"] = node_descriptor[marks[1] + 7: marks[2] - 1]
            parameters["relative_object"] = node_descriptor[marks[2]: marks[3] + 1]

        return parameters

    def find_target_object(self):
        """ Finds first target object from possible objects in world """
        if self.parameters["target_object"] == '"any object"':
            if "relation" in self.parameters and "relative_object" in self.parameters:
                for target_object in self.world_interface.movable_objects:
                    if self.world_interface.object_at(target_object, self.parameters["relation"], self.parameters["relative_object"]):
                        return target_object
            return None
        else:
            return self.parameters["target_object"]

    def check_for_success(self):
        """Check if object is grasped."""
        # self.world_interface.get_feedback()
        if self.world_interface.get_grasped_object() == self.target_object:
            self.success()

    def check_for_failure(self):
        """Fail if some other object is grasped."""
        grasped_object = self.world_interface.get_grasped_object()
        return grasped_object not in (self.target_object , None)

    def execute(self):
        self.calc_grasp_position()
        self.calc_approach_position()
        # self.world_interface.move_armbase()
        # self.world_interface.move_linear(self.world_interface.pos_to_dict(self.approach_position))
        self.world_interface.pick_up(self.target_object)

    def calc_grasp_position(self):
        """Gets grasp position of object"""
        self.grasp_position = self.world_interface.object_positions[self.target_object]

    def calc_approach_position(self):
        """Gets approach position of object"""
        self.approach_position = self.grasp_position + np.array([0.0, 0.05, 0.0]) #TODO move numbers to world_interface
        if "cap" in self.target_object:
            self.approach_position[1] += 0.02

class Place(ActionBehavior):
    """
    Place object on position
    """
    skill_name = "Place"
    description = "Place an object with specified relation (on, inside) (e.g. Place Cup inside microwave, place apple On countertop). - For 'on' relations: The target surface must be clear. - For 'inside' relations: The target container must not have another object already inside, as indicated by an existing AtPos condition with relation 'inside'."

    class PlaceStates(IntEnum):
        """Define the internal states during execution."""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4

    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.release_position = None
        self.approach_position = None
        self.orientation = None
        self.internal_state = self.PlaceStates.INIT
        self.full_placing_program = ''
        self.target_object = parameters["target_object"]
        preconditions = []
        postconditions = []
        if world_interface.is_graspable(self.target_object):
            preconditions = [Grasped('', {"target_object": self.target_object}, world_interface),
                         NearRobot('', {"destination": parameters["relative_object"]}, world_interface)]
            postconditions = [AtPos('', {"target_object": self.target_object,
                                         "relation": parameters["relation"],
                                         "relative_object": parameters["relative_object"]},
                                    world_interface)]
        elif not "relation" in parameters and not "relative_object" in parameters:
            parameters = parameters.copy() # Make sure not to change incoming
            parameters["target_object"] = '"grasped object"'
            parameters["relation"] = "on"
            parameters["relative_object"] = world_interface.get_id("CounterTop")
            postconditions = [Grasped('', {"not": True, "target_object": '"any object"'}, world_interface)]
        elif self.target_object == '"grasped object"':
            postconditions = [Grasped('', {"not": True, "target_object": '"any object"'}, world_interface)]
        name = Place.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "place " + extract_name(parameters["target_object"])
        node_string += " " + parameters["relation"]
        node_string += " " + extract_name(parameters["relative_object"])
        node_string += "!"
        return node_string

    def initialise(self):
        self.internal_state = self.PlaceStates.INIT
        ActionBehavior.initialise(self)
        if self.parameters["target_object"] == '"grasped object"':
            self.target_object = self.world_interface.get_grasped_object()
            if self.target_object is None:
                self.success()

    @staticmethod
    def parse_parameters(node_descriptor):
        """ Parse behavior parameters from string """
        parameters = {}
        n_marks = 4
        marks = [0] * n_marks
        marks[0] = node_descriptor.find('"')
        for i in range(1, n_marks):
            if marks[i-1] >= 0:
                marks[i] = node_descriptor.find('"', marks[i-1] + 1)
            else:
                print("Error, parameter parsing failed")
                return None

        parameters["target_object"] = node_descriptor[marks[0]: marks[1] + 1]
        parameters["relation"] = node_descriptor[marks[1] + 2: marks[2] - 1]
        parameters["relative_object"] = node_descriptor[marks[2]: marks[3] + 1]

        return parameters

    def check_for_success(self):
        """Check if object is at target position."""
        if self.state != pt.common.Status.SUCCESS:
            if self.world_interface.object_at(self.target_object, self.parameters["relation"], self.parameters["relative_object"]) and \
                self.world_interface.get_grasped_object() != self.target_object:
                self.success()

    def check_for_failure(self):
        """Fail if object is not grasped."""
        return self.world_interface.get_grasped_object() != self.target_object

    def execute(self):
            # self.precondition_verifier_check()
            # self.precondition_suggestor_check()
            # if self.parameters["relation"] == "in":
            #     self.world_interface.put_in(self.target_object, self.parameters["relative_object"])
            # else:
            self.world_interface.place_obj(self.target_object, self.parameters["relative_object"], self.parameters["relation"])

    def calc_release_position(self):
        """Gets release position of object"""
        if self.parameters["relation"] == "on":
            relative_object_position = self.world_interface.get_position(self.parameters["relative_object"])
            if self.parameters["relative_object"] == '"table"':
                self.release_position = relative_object_position + np.array([0.0, 0.0, self.world_interface.CUBE_SIZE / 2 + 0.003])
            else:
                self.release_position = relative_object_position + np.array([0.0, 0.0, self.world_interface.CUBE_SIZE + 0.003])#TODO move numbers to world_interface
        elif self.parameters["relation"] == "in":
            if self.parameters["relative_object"] == '"centrifuge"':
                self.release_position = [0.5, 0.0876, 0.19]
                self.orientation = np.array([0.1227878, -0.6963642, 0.6963642, 0.1227878])
            else:
                relative_object_position = self.world_interface.get_position(self.parameters["relative_object"])
                if relative_object_position is not None:
                    self.release_position = relative_object_position + np.array([0.0, 0.0, self.world_interface.CUP_HEIGHT + 0.02])#TODO move numbers to world_interface
        elif self.parameters["relation"] == "at" and isinstance(self.parameters["relative_object"], np.ndarray):
            self.release_position = self.parameters["relative_object"]

    def calc_place_approach_position(self):
        """Gets place approach position of object"""
        if self.parameters["relation"] == "in" and self.parameters["relative_object"] == '"centrifuge"':
            self.approach_position = self.release_position + np.array([0.0, -0.036, 0.1])
        else:
            self.approach_position = self.release_position + np.array([0.0, 0.0, 0.05])#TODO move numbers to world_interface

class Navigate(ActionBehavior):
    """
    Navigate to a specific location in the environment.
    """
    skill_name = "navigate"
    description = "Navigate to a specific location in the environment."

    class NavigateStates(IntEnum):
        """Define the internal states during execution."""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4

    def __init__(self, name, parameters, world_interface: WorldInterface, vlm, _verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.NavigateStates.INIT
        self.target_object = parameters["destination"]
        
        preconditions = [LocationKnown('', {"target_object": self.target_object}, world_interface)]
        postconditions = [NearRobot('', {"destination": self.target_object}, world_interface)]
        
        name = Navigate.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=False)
    
    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "navigate to " + extract_name(parameters["destination"])
        node_string += "!"
        return node_string

    def initialise(self):
        self.internal_state = self.NavigateStates.INIT
        ActionBehavior.initialise(self)
        if self.world_interface.is_near_robot(self.parameters["destination"]):
            self.success()
    
    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.is_near_robot(self.target_object):
            self.success()

    def execute(self):
        # self.world_interface.move_armbase()
        self.world_interface.navigate_to_obj(self.target_object)
    
class Open(ActionBehavior):
    """
    Open an object (e.g. open microwave door or fridge door)
    """
    skill_name = "Open"
    description = "Open an object (e.g. open microwave door or fridge door)"

    class OpenStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.OpenStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [Grasped('', {"not": True, "target_object": '"any object"'}, world_interface),
                         NearRobot('', {"destination": self.target_object}, world_interface)]
        postconditions = [Opened('', {"interact_object": self.target_object}, world_interface)]
        
        name = Open.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)
        
    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "open " + extract_name(parameters["interact_object"])
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.OpenStates.INIT
        ActionBehavior.initialise(self)
        if self.target_object in self.world_interface.object_opened.keys():
            if self.world_interface.object_opened[self.target_object]:
                self.success()
                
    def check_for_success(self):
        """Check if object is opened."""
        if self.world_interface.object_opened[self.target_object]:
            self.success()
            
    def execute(self):
        self.world_interface.open_obj(self.target_object)

class Close(ActionBehavior):
    """
    Close an object (e.g. close microwave door or fridge door)
    """
    skill_name = "Close"
    description = "Close an object (e.g. close microwave door or fridge door)"

    class CloseStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.CloseStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [Grasped('', {"not": True, "target_object": '"any object"'}, world_interface),
                         NearRobot('', {"destination": self.target_object}, world_interface)]
        postconditions = [Opened('', {"not": True, "interact_object": self.target_object}, world_interface)]
        
        name = Close.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)
        
    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Close " + extract_name(parameters["interact_object"])
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.CloseStates.INIT
        ActionBehavior.initialise(self)
        if not self.target_object in self.world_interface.object_opened.keys():
            if self.world_interface.object_opened[self.target_object]:
                self.success()
                
    def check_for_success(self):
        """Check if object is closed."""
        if not self.world_interface.object_opened[self.target_object]:
            self.success()
            
    def execute(self):
        self.world_interface.close_obj(self.target_object)

class ToggleOn(ActionBehavior):
    """
    Toggle on an object (e.g. microwave or coffee machine)
    """
    skill_name = "Toggle_On"
    description = "Turn on an object (e.g. microwave or coffee machine)"

    class ToggleOnStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.ToggleOnStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [NearRobot('', {"destination": self.target_object}, world_interface)]
        postconditions = [Toggled('', {"interact_object": self.target_object}, world_interface)]
        
        name = ToggleOn.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)
        
    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Toggle on " + extract_name(parameters["interact_object"])
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.ToggleOnStates.INIT
        ActionBehavior.initialise(self)
        if self.world_interface.is_toggled(self.target_object):
            self.success()
                
    def check_for_success(self):
        """Check if object is on."""
        if self.world_interface.is_toggled(self.target_object):
            self.success()
            
    def execute(self):
        self.world_interface.toggle_on(self.target_object)
    
class ToggleOff(ActionBehavior):
    """
    Toggle off an object (e.g. microwave or coffee machine)
    """
    skill_name = "Toggle_Off"
    description = "Turn off an object (e.g. microwave or coffee machine)"

    class ToggleOffStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.ToggleOffStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [NearRobot('', {"destination": self.target_object}, world_interface)]
        postconditions = [Toggled('', {"not": True, "interact_object": self.target_object}, world_interface)]
        
        name = ToggleOff.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)
        
    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Toggle off " + extract_name(parameters["interact_object"])
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.ToggleOffStates.INIT
        ActionBehavior.initialise(self)
        if not self.world_interface.is_toggled(self.target_object):
            self.success()
                
    def check_for_success(self):
        """Check if object is off."""
        if not self.world_interface.is_toggled(self.target_object):
            self.success()
            
    def execute(self):
        self.world_interface.toggle_off(self.target_object)

class Slice(ActionBehavior):
    """
    Toggle on an object (e.g. microwave or coffee machine)
    """
    skill_name = "Slice"
    description = "Slices an object like a fruit or a vegetable (e.g. Slice Apple, Slice Carrot, etc.)"

    class SliceStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.SliceStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [NearRobot('', {"destination": self.target_object}, world_interface)]
        postconditions = [Sliced('', {"interact_object": self.target_object}, world_interface)]
        
        name = Slice.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Slice " + extract_name(parameters["interact_object"])
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.SliceStates.INIT
        ActionBehavior.initialise(self)
        if self.world_interface.is_sliced(self.target_object):
            self.success()
                
    def check_for_success(self):
        """Check if object is on."""
        if self.world_interface.is_sliced(self.target_object):
            self.success()
            
    def execute(self):
        self.world_interface.slice_obj(self.target_object)

class Crack(ActionBehavior):
    """
    Crack an object (e.g. Crack an egg)
    """
    skill_name = "Crack"
    description = "Crack an object (e.g. Crack an egg, Crack a nut, etc.)"

    class CrackStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.CrackStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [NearRobot('', {"destination": self.target_object}, world_interface)]
        postconditions = [Cracked('', {"interact_object": self.target_object}, world_interface)]
        
        name = Crack.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Crack " + extract_name(parameters["interact_object"])
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.CrackStates.INIT
        ActionBehavior.initialise(self)
        if self.world_interface.is_cracked(self.target_object):
            self.success()
                
    def check_for_success(self):
        """Check if object is on."""
        if self.world_interface.is_cracked(self.target_object):
            self.success()
            
    def execute(self):
        self.world_interface.crack_obj(self.target_object)

class FillWithWater(ActionBehavior):
    """
    Fill an object with liquid
    """
    skill_name = "Fill_With_Water"
    description = "Fill an object with Water from the faucet (e.g. Fill Cup with water, Fill Pot with water, etc.)"

    @staticmethod
    class FillWithWaterStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.FillWithWaterStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [AtPos('', {"target_object": self.target_object,
                                    "relation": "inside",
                                    "relative_object": world_interface.object_dict["Sink"]}, world_interface),
                        NearRobot('', {"destination": world_interface.object_dict["Sink"]}, world_interface),
                        Filled('', {"not": True, "interact_object": self.target_object, "liquid": "any liquid"}, world_interface),]
        postconditions = [Toggled('',{"interact_object": world_interface.object_dict["Faucet"]}, world_interface),
                        Filled('', {"interact_object": self.target_object, "liquid": "water"}, world_interface),
                        Cleaned('', {"interact_object": self.target_object}, world_interface)]
        
        name = FillWithWater.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Fill " + extract_name(parameters["interact_object"]) + "with water"
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.FillWithWaterStates.INIT
        ActionBehavior.initialise(self)
        if self.world_interface.is_filled(self.target_object, "water"):
            self.success()
                
    def check_for_success(self):
        """Check if object is on."""
        if self.world_interface.is_filled(self.target_object, "water"):
            self.success()
            
    def execute(self):
        self.world_interface.toggle_on("Faucet")

class FillWithCoffee(ActionBehavior):
    """
    Fill an object with liquid
    """
    skill_name = "Fill_With_Coffee"
    description = "Fill a specified Mug with Coffee from the Coffee Machine"

    @staticmethod
    class FillWithCoffeeStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.FillWithCoffeeStates.INIT
        self.target_object = parameters["interact_object"]
        preconditions = [Cleaned('', {"interact_object": self.target_object}, world_interface),
                         Filled('', {"not": True,"interact_object": self.target_object, "liquid": "any liquid"}, world_interface),
                         AtPos('', {"target_object": self.target_object,
                                    "relation": "inside",
                                    "relative_object": world_interface.object_dict["CoffeeMachine"]}, world_interface),
                         NearRobot('', {"destination": world_interface.object_dict["CoffeeMachine"]}, world_interface),
                         ]
        postconditions = [Filled('', {"interact_object": self.target_object, "liquid": "coffee"}, world_interface),
                          Toggled('',{"interact_object": world_interface.object_dict["CoffeeMachine"]}, world_interface)]
        
        name = FillWithCoffee.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Fill " + extract_name(parameters["interact_object"]) + "with coffee"
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.FillWithCoffeeStates.INIT
        ActionBehavior.initialise(self)
        if self.world_interface.is_filled(self.target_object, "coffee"):
            self.success()
                
    def check_for_success(self):
        """Check if object is on."""
        if self.world_interface.is_filled(self.target_object, "coffee"):
            self.success()
            
    def execute(self):
        self.world_interface.toggle_on("CoffeeMachine")

class Pour(ActionBehavior):
    """
    Pour liquid out of object
    """
    skill_name = "Pour"
    description = "Empty an object from liquid (e.g. Pour water from cup, Pour coffee from mug, etc.)"

    @staticmethod
    class PourStates(IntEnum):
        """Define the internal states"""
        INIT = 1
        WAITING_FOR_STOP = 2
        WAITING_FOR_START = 3
        RUNNING = 4
        
    def __init__(self, name, parameters, world_interface, vlm, verbose=False):
        self.world_interface = world_interface
        self.internal_state = self.PourStates.INIT
        self.target_object = parameters["interact_object"]
        self.reciptacle = world_interface.object_dict["SinkBasin"]
        preconditions = [Filled('', {"interact_object": self.target_object, "liquid": "any liquid"}, world_interface),
                        NearRobot('', {"destination": world_interface.object_dict["SinkBasin"]}, world_interface),
                          Grasped('', {"target_object": self.target_object}, world_interface),]
        postconditions = [Filled('', {"not": True,"interact_object": self.target_object, "liquid": "any liquid"}, world_interface)]
        if self.reciptacle is None:
            self.reciptacle = "SinkBasin"
        
        name = Pour.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, preconditions, postconditions, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        """ Creates a string """
        node_string = "Pour " + extract_name(parameters["interact_object"])
        node_string += "!"
        return node_string
    
    def initialise(self):
        self.internal_state = self.PourStates.INIT
        ActionBehavior.initialise(self)
        if not self.world_interface.is_filled(self.target_object):
            self.success()
                
    def check_for_success(self):
        """Check if object is on."""
        if not self.world_interface.is_filled(self.target_object):
            self.success()
            
    def execute(self):
        self.world_interface.pour(self.target_object, self.reciptacle)

def get_condition_nodes():
    """ Returns a list of all action nodes available for planning """
    return [AtPos, Grasped, LocationKnown, NearRobot, Opened, Unlocked, Toggled, Sliced, Cracked, Filled, Cleaned]


def get_action_nodes():
    """ Returns a list of all action nodes available for planning """
    return [Grasp, Place, Navigate, Open, Close, ToggleOn, ToggleOff, Slice, Pour, FillWithWater, FillWithCoffee]
