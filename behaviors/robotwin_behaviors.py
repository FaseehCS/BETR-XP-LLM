""" Behaviors for the RobotWin2 environment """

from enum import IntEnum
from behaviors.common_behaviors import Behavior, ActionBehavior
import behaviors.common_behaviors
import py_trees as pt

from interfaces.robotwin_world_interface import WorldInterface

def get_node(node_descriptor, world_interface, verbose = False):
    """ Returns a node object given the descriptor string """
    return behaviors.common_behaviors.get_node(node_descriptor, world_interface, verbose=verbose)

def extract_name(object_id):
    if object_id is None:
        return "None"
    return str(object_id)

# ----------- CONDITION NODES ------------

class Grasped(Behavior):
    condition_name = "Grasped"
    description = "Check if the object is currently held by the robot."

    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Grasped.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    @staticmethod
    def to_string(parameters):
        return f'grasped {extract_name(parameters["object"])} by {extract_name(parameters["arm_tag"])} arm?'

    def update(self):
        return self.check_negated(self.world_interface.is_grasped(self.parameters["arm_tag"], self.parameters["object"]))

class AtPos(Behavior):
    condition_name = "AtPos"
    description = "Check if object is at given pose."

    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        if not "relative_object" in parameters:
            parameters["relative_object"] = ""
        name = AtPos.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    @staticmethod
    def to_string(parameters):
        return f'{extract_name(parameters["object"])} {parameters["relation"]} {parameters["relative_object"]}?'

    def update(self):
        target_object = self.parameters["object"]
        relation = self.parameters["relation"]
        relative_object = self.parameters["relative_object"]
        if target_object == '"any object"':
            object_at = False
            return self.check_negated(object_at)
        else:
            return self.check_negated(self.world_interface.object_at(target_object, relation, relative_object))

class HeldByArm(Behavior):
    condition_name = "HeldByArm"
    description = "Check if object is held by specified arm."

    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = HeldByArm.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    @staticmethod
    def to_string(parameters):
        return f'{extract_name(parameters["object"])} held by {parameters["arm"]}?'

    def update(self):
        return self.check_negated(self.world_interface.held_by_arm(self.parameters["object"], self.parameters["arm"]))

class ContentsIn(Behavior):
    condition_name = "ContentsIn"
    description = "Check if contents are in the given container."

    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = ContentsIn.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    @staticmethod
    def to_string(parameters):
        return f'{extract_name(parameters["content"])} in {extract_name(parameters["container"])}?'

    def update(self):
        return self.check_negated(self.world_interface.contents_in(self.parameters["content"], self.parameters["container"]))

class BothArmsFree(Behavior):
    condition_name = "BothArmsFree"
    description = "Check if both arms are free (not holding anything)."

    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = "both arms free?"
        super().__init__(name, parameters, world_interface)

    def update(self):
        return self.check_negated(self.world_interface.both_arms_free())

class Toggled(Behavior):
    condition_name = "Toggled"
    description = "Check if the object has been toggled (e.g., button pressed/clicked)."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Toggled.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    @staticmethod
    def to_string(parameters):
        return f"{extract_name(parameters['object'])} toggled?"

    def update(self):
        return self.check_negated(self.world_interface.is_toggled(self.parameters["object"]))

class Beaten(Behavior):
    condition_name = "Beaten"
    description = "Check if the object has been beaten by a tool enough times."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Beaten.to_string(parameters)
        super().__init__(name, parameters, world_interface)
    @staticmethod
    def to_string(parameters):
        if "count" in parameters:
            return f"{extract_name(parameters['target'])} beaten by {extract_name(parameters['tool'])} ({parameters['count']}x)?"
        return f"{extract_name(parameters['target'])} beaten by {extract_name(parameters['tool'])}?"
    def update(self):
        # Assumes world_interface.is_beaten takes count as optional argument
        return self.check_negated(
            self.world_interface.is_beaten(
                self.parameters["target"],
                self.parameters["tool"],
                self.parameters.get("count", 1)
            )
        )

class Hung(Behavior):
    condition_name = "Hung"
    description = "Check if the object is hanging from the target (e.g., a rack)."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Hung.to_string(parameters)
        super().__init__(name, parameters, world_interface)

    @staticmethod
    def to_string(parameters):
        return f"{extract_name(parameters['object'])} hung on {extract_name(parameters['target'])}?"

    def update(self):
        return self.check_negated(self.world_interface.is_hung(self.parameters["object"], self.parameters["target"]))

class Opened(Behavior):
    condition_name = "Opened"
    description = "Check if the object is open (for articulated objects)."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Opened.to_string(parameters)
        super().__init__(name, parameters, world_interface)
    @staticmethod
    def to_string(parameters):
        return f"{extract_name(parameters['object'])} opened?"
    def update(self):
        return self.check_negated(self.world_interface.is_opened(self.parameters["object"]))

class Scanned(Behavior):
    condition_name = "Scanned"
    description = "Check if the object has been scanned by the scanner."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Scanned.to_string(parameters)
        super().__init__(name, parameters, world_interface)
    @staticmethod
    def to_string(parameters):
        return f"{extract_name(parameters['object'])} scanned by {extract_name(parameters['scanner'])}?"
    def update(self):
        return self.check_negated(self.world_interface.is_scanned(self.parameters["object"], self.parameters["scanner"]))

class Shaken(Behavior):
    condition_name = "Shaken"
    description = "Check if the object has been shaken (enough times, in the correct way)."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Shaken.to_string(parameters)
        super().__init__(name, parameters, world_interface)
    @staticmethod
    def to_string(parameters):
        s = f"{extract_name(parameters['object'])} shaken"
        if "count" in parameters:
            s += f" ({parameters['count']}x)"
        if "shake_orientation" in parameters:
            s += f" [{parameters['shake_orientation']}]"
        return s + "?"
    def update(self):
        return self.check_negated(
            self.world_interface.is_shaken(
                self.parameters["object"],
                self.parameters.get("shake_orientation", None),
                self.parameters.get("count", 1)
            )
        )

class On(Behavior):
    condition_name = "On"
    description = "Check if one object is on top of another."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = On.to_string(parameters)
        super().__init__(name, parameters, world_interface)
    @staticmethod
    def to_string(parameters):
        return f"{extract_name(parameters['object'])} on {extract_name(parameters['support'])}?"
    def update(self):
        return self.check_negated(self.world_interface.is_on(self.parameters["object"], self.parameters["support"]))

class Stamped(Behavior):
    condition_name = "Stamped"
    description = "Check if the stamp has been stamped on the target."
    def __init__(self, name, parameters, world_interface: WorldInterface, _verbose=False):
        name = Stamped.to_string(parameters)
        super().__init__(name, parameters, world_interface)
    @staticmethod
    def to_string(parameters):
        return f"{extract_name(parameters['stamp'])} stamped on {extract_name(parameters['target'])}?"
    def update(self):
        return self.check_negated(self.world_interface.is_stamped(self.parameters["stamp"], self.parameters["target"]))

# ----------- ACTION NODES ------------

class Pick(ActionBehavior):
    skill_name = "Pick"
    description = "Pick up an object"
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):

        if "arm_tag" not in parameters or parameters["arm_tag"] == "any":
            parameters["arm_tag"] = world_interface.get_closest_arm(parameters["object"])

        pre = [Grasped('', {"not": True, "object": '"any object"', "arm_tag": parameters["arm_tag"]}, world_interface)]
        opposite_arm = "left" if parameters["arm_tag"] == "right" else "right"
        pre.append(Grasped('', {"not": True, "object": parameters["object"], "arm_tag": opposite_arm}, world_interface))

        post = [Grasped('', {"object": parameters["object"], "arm_tag": parameters["arm_tag"]}, world_interface)]
        name = Pick.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=300, verbose=verbose)
        self.action_string = name

    def to_string(parameters):
        action_string = f"pick {extract_name(parameters['object'])} with {extract_name(parameters['arm_tag'])} arm!"
        return action_string

    # def execute(self):
    #     self.world_interface.pick(self.parameters["object"], self.parameters["arm_tag"])

    def execute(self):
        self.world_interface.gripper_open(self.parameters["arm_tag"])
        self.world_interface.generate_action(self.action_string)

class MoveByDisplacement(ActionBehavior):
    skill_name = "MoveByDisplacement"
    description = "Move the grasped object by a relative displacement"
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        pre = [Grasped('', {"object": parameters["object"]}, world_interface)]
        post = [Grasped('', {"object": parameters["object"]}, world_interface),
                AtPos('', {"object": parameters["object"], "pose": parameters["target_pose"]}, world_interface)]
        name = MoveByDisplacement.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=300, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        return f"move_by_displacement {extract_name(parameters['object'])} dz={parameters['dz']}!"

    def execute(self):
        self.world_interface.move_by_displacement(self.parameters["object"], self.parameters["dz"])

class Place(ActionBehavior):
    skill_name = "Place"
    description = "Place an object at a specified pose"
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):

        # Default behavior if relation and relative_object are not defined
        if not "relation" in parameters and not "relative_object" in parameters:
            parameters["relation"] = "away"

        if "arm_tag" not in parameters or parameters["arm_tag"] == "any":
            parameters["arm_tag"] = world_interface.get_closest_arm(parameters["object"])

        pre = [Grasped('', {"object": parameters["object"], "arm_tag": parameters["arm_tag"]}, world_interface)]

        post = [Grasped('', {"not": True, "object": '"any object"', "arm_tag": parameters["arm_tag"]}, world_interface)]

        if parameters["relation"] == "away":
            post.append(AtPos('',{"object": parameters["object"], "relation": parameters["relation"]}, world_interface))
        else:
            post.append(AtPos('',{"object": parameters["object"], "relation": parameters["relation"], "relative_object": parameters["relative_object"]}, world_interface))
 

        name = Place.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=300, verbose=verbose)

    def to_string(parameters):
        # We have five action variants: place on, place inside, place away, place to the left of, place to the right of. Text needs to be generated accordingly.
        if parameters["relation"] == "away":
            action_string = f"place {extract_name(parameters['object'])} {parameters['relation']} with {extract_name(parameters['arm_tag'])} arm!"
        else:
            action_string = f"place {extract_name(parameters['object'])} {parameters['relation']} {extract_name(parameters['relative_object'])} with {extract_name(parameters['arm_tag'])} arm!"
        return action_string
    
    def execute(self):
        self.world_interface.place(
            self.parameters["object"],
            self.parameters["relation"],
            self.parameters["relative_object"],
            self.parameters["arm_tag"]
        )

class Toggle(ActionBehavior):
    skill_name = "Toggle"
    description = "Toggle (press/click) an object with the gripper (requires empty gripper)."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        # pre = [Grasped('', {"object": None}, world_interface)]
        pre = []
        post = [Toggled('', {"object": parameters["object"]}, world_interface)]
        name = Toggle.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=200, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        return f"toggle {extract_name(parameters['object'])}!"

    def execute(self):
        self.world_interface.toggle(self.parameters["object"])

class Beat(ActionBehavior):
    skill_name = "Beat"
    description = "Beat or tap an object with a tool (possibly multiple times)."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        pre = [Grasped('', {"object": parameters["tool"]}, world_interface)]
        post = [Beaten('', {
            "target": parameters["target"],
            "tool": parameters["tool"],
            "count": parameters.get("count", 1)
        }, world_interface)]
        name = Beat.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=200, verbose=verbose)
    @staticmethod
    def to_string(parameters):
        count_str = f" {parameters['count']}x" if "count" in parameters else ""
        return f"beat {extract_name(parameters['target'])} with {extract_name(parameters['tool'])}{count_str}!"
    def execute(self):
        self.world_interface.beat(
            self.parameters["tool"],
            self.parameters["target"],
        )

class Pour(ActionBehavior):
    skill_name = "Pour"
    description = "Pour contents from one container into another."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        pre = [Grasped('', {"object": parameters["from"]}, world_interface)]
        post = [ContentsIn('', {"content": parameters["content"], "container": parameters["to"]}, world_interface)]
        name = Pour.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=400, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        return f"pour {extract_name(parameters['content'])} from {extract_name(parameters['from'])} to {extract_name(parameters['to'])}!"

    def execute(self):
        self.world_interface.pour(self.parameters["from"], self.parameters["to"], self.parameters["content"])

class GrabTogether(ActionBehavior):
    skill_name = "GrabTogether"
    description = "Use both arms to grab an object together."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        pre = [BothArmsFree('', {}, world_interface)]
        post = [HeldByArm('', {"object": parameters["object"], "arm": "both"}, world_interface)]
        name = GrabTogether.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=500, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        return f"grab together {extract_name(parameters['object'])}!"

    def execute(self):
        self.world_interface.grab_together(self.parameters["object"])

class Handover(ActionBehavior):
    skill_name = "Handover"
    description = "Pass an object from one arm to another."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        pre = [HeldByArm('', {"object": parameters["object"], "arm": parameters["from_arm"]}, world_interface)]
        post = [HeldByArm('', {"object": parameters["object"], "arm": parameters["to_arm"]}, world_interface)]
        name = Handover.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=400, verbose=verbose)

    @staticmethod
    def to_string(parameters):
        return f"handover {extract_name(parameters['object'])} from {parameters['from_arm']} to {parameters['to_arm']}!"

    def execute(self):
        self.world_interface.handover(self.parameters["object"], self.parameters["from_arm"], self.parameters["to_arm"])

class Hang(ActionBehavior):
    skill_name = "Hang"
    description = "Hang an object onto a target (e.g., rack, hook)."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        # Assumes object must be grasped, and at the hanging pose
        pre = [
            Grasped('', {"object": parameters["object"]}, world_interface),
            AtPos('', {"object": parameters["object"], "pose": parameters["hang_pose"]}, world_interface)
        ]
        post = [
            Grasped('', {"object": parameters["object"], "not": True}, world_interface),
            Hung('', {"object": parameters["object"], "target": parameters["target"]}, world_interface)
        ]
        name = Hang.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=300, verbose=verbose)

class Open(ActionBehavior):
    skill_name = "Open"
    description = "Open an articulated object (e.g., laptop, microwave)."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        pre = [Grasped('', {"object": parameters["object"]}, world_interface)]
        post = [Opened('', {"object": parameters["object"]}, world_interface)]
        name = Open.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=300, verbose=verbose)
    @staticmethod
    def to_string(parameters):
        return f"open {extract_name(parameters['object'])}!"
    def execute(self):
        self.world_interface.open(self.parameters["object"])


    @staticmethod
    def to_string(parameters):
        return f"hang {extract_name(parameters['object'])} on {extract_name(parameters['target'])}!"

    def execute(self):
        self.world_interface.hang(self.parameters["object"], self.parameters["target"], self.parameters["hang_pose"])

class Scan(ActionBehavior):
    skill_name = "Scan"
    description = "Scan an object with a scanner."
    def __init__(self, name, parameters, world_interface, vlm=None, verbose=False):
        pre = [Grasped('', {"object": parameters["scanner"]}, world_interface),
               Grasped('', {"object": parameters["object"]}, world_interface)]
        post = [Scanned('', {"object": parameters["object"], "scanner": parameters["scanner"]}, world_interface)]
        name = Scan.to_string(parameters)
        supActionBehaviorer().__init__(name, parameters, world_interface, pre, post, vlm, max_ticks=200, verbose=verbose)
    @staticmethod
    def to_string(parameters):
        return f"scan {extract_name(parameters['object'])} with {extract_name(parameters['scanner'])}!"
    def execute(self):
        self.world_interface.scan(self.parameters["scanner"], self.parameters["object"])

class Shake(ActionBehavior):
    skill_name = "Shake"
    description = "Shake an object (optionally a number of times and/or orientation)."
    def __init__(self, name, parameters, world_interface, vlm=None, verbose=False):
        pre = [Grasped('', {"object": parameters["object"]}, world_interface)]
        post = [Grasped('', {"object": parameters["object"]}, world_interface),
                Shaken('', {
                    "object": parameters["object"],
                    "count": parameters.get("count", 1),
                    "shake_orientation": parameters.get("shake_orientation", None)
                }, world_interface)]
        name = Shake.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=150, verbose=verbose)
    @staticmethod
    def to_string(parameters):
        s = f"shake {extract_name(parameters['object'])}"
        if "shake_orientation" in parameters:
            s += f" [{parameters['shake_orientation']}]"
        if "count" in parameters:
            s += f" ({parameters['count']}x)"
        return s + "!"
    def execute(self):
        self.world_interface.shake(
            self.parameters["object"],
            self.parameters.get("shake_orientation", None),
            self.parameters.get("count", 1)
        )

class Stamp(ActionBehavior):
    skill_name = "Stamp"
    description = "Stamp a target with a stamp."
    def __init__(self, name, parameters, world_interface: WorldInterface, vlm=None, verbose=False):
        pre = [Grasped('', {"object": parameters["stamp"]}, world_interface)]
        post = [Stamped('', {"stamp": parameters["stamp"], "target": parameters["target"]}, world_interface)]
        name = Stamp.to_string(parameters)
        ActionBehavior.__init__(self, name, parameters, world_interface, pre, post, vlm, max_ticks=150, verbose=verbose)
    @staticmethod
    def to_string(parameters):
        return f"stamp {extract_name(parameters['stamp'])} on {extract_name(parameters['target'])}!"
    def execute(self):
        self.world_interface.stamp(self.parameters["stamp"], self.parameters["target"])
# ----------- EXPORTS FOR PLANNER --------------

def get_condition_nodes():
    return [Grasped, AtPos, HeldByArm, Toggled, Beaten, ContentsIn, BothArmsFree, Hung, Opened, Scanned, Shaken, On, Stamped]

def get_action_nodes():
    return [Pick, MoveByDisplacement, Place, Toggle, Beat, Pour, GrabTogether, Handover, Hang, Open, Scan, Shake, Stamp]