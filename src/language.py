from dataclasses import dataclass
from typing import Any, Callable, List, Dict, Tuple

from lark import Lark, Transformer, v_args, Token

# ============ LANGUAGE ============

grammar = """
start: robot_def

robot_def: "robot" NAME "{" body parts actuator? "}"

body: "body" "{" "shape:" shape_def "stiffness:" NUMBER "mass:" NUMBER "}"

parts: "parts" "{" part_def+ "}"

part_def: NAME ":" part_type "{" part_attrs "}"  -> create_part

part_type: "leg" -> leg_type
        | "body_part" -> body_part_type

part_attrs: "position:" position "size:" size "stiffness:" NUMBER "mass:" NUMBER

actuator: "actuator" "{" "gait:" gait_type "frequency:" NUMBER "forces:" force_def "}"

gait_type: "quadruped" -> quadruped_gait
        | "worm" -> worm_gait
        | "custom" -> custom_gait

force_def: "{" "lift:" NUMBER "push:" NUMBER "swing:" NUMBER "}"

shape_def: box_shape -> create_box_shape
         | cylinder_shape -> create_cylinder_shape
         | "table" -> table_shape
         | "worm" -> worm_shape

box_shape: "box" "(" NUMBER "," NUMBER "," NUMBER ")"
cylinder_shape: "cylinder" "(" NUMBER "," NUMBER ")"

SIGNED_NUMBER: ["+"|"-"] NUMBER
?number: NUMBER | SIGNED_NUMBER

position: "(" number "," number "," number ")"
size: "(" number "," number "," number ")"

simulation: "simulate" "{" "duration:" number "gravity:" number "floor_friction:" friction_def "}"

friction_def: "(" number "," number ")"  // forward, backward

NAME: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.NUMBER
%import common.WS
%ignore WS
"""

# ============ AST NODES ============

@dataclass
class RobotDef:
    name: str
    body: 'BodyDef'
    parts: List['PartDef']
    actuator: 'ActuatorDef'

@dataclass
class BodyDef:
    shape: str
    dimensions: Tuple[float, float, float]
    stiffness: float
    mass: float

@dataclass
class PartDef:
    name: str
    type: str  # "leg" or "body_part"
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    stiffness: float
    mass: float

@dataclass
class ActuatorDef:
    gait: str
    frequency: float
    forces: Dict[str, float]

# ============ INTERPRETER ============

class RobotInterpreter:
    def __init__(self):
        self.robot = None
        
    def eval(self, tree):
        """Evaluate parsed tree or transformed object"""
        if isinstance(tree, RobotDef):
            print("DEBUG: Using already transformed RobotDef")
            self.robot = tree
            return self.robot
        else:
            print("DEBUG: Evaluating parse tree")
            return self.eval_robot_def(tree)
    
    def eval_robot_def(self, node):
        if isinstance(node, RobotDef):
            print("DEBUG: Using already transformed RobotDef")
            self.robot = node
            return self.robot
            
        name = None
        body = None
        parts = []
        actuator = None
        
        print("DEBUG: Evaluating robot definition from parse tree")
        print(f"DEBUG: Node children types: {[getattr(child, 'data', child.type if isinstance(child, Token) else type(child)) for child in node.children]}")
        
        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                name = child.value
                print(f"DEBUG: Found robot name: {name}")
            elif hasattr(child, 'data'):
                if child.data == 'body':
                    print("DEBUG: Found body section")
                    body = self.eval_body(child)
                elif child.data == 'parts':
                    parts = self.eval_parts(child)
                elif child.data == 'actuator':
                    actuator = self.eval_actuator(child)
        
        print(f"DEBUG: Robot components - Name: {name}, Body: {body}, Parts: {len(parts)}, Actuator: {actuator is not None}")
        self.robot = RobotDef(name, body, parts, actuator)
        return self.robot
    
    def eval_body(self, node):
        shape_def = None
        stiffness = 0
        mass = 0
        
        print("DEBUG: Evaluating body...")
        print(f"DEBUG: Body node children: {[getattr(child, 'data', child.type if isinstance(child, Token) else type(child)) for child in node.children]}")
        
        for child in node.children:
            if hasattr(child, 'data'):
                if child.data == 'shape_def':
                    print("DEBUG: Found shape_def")
                    shape_def = self.eval_shape_def(child)
                    print(f"DEBUG: Evaluated shape_def: {shape_def}")
                    
            if isinstance(child, Token):
                if child.type == 'NUMBER':
                    if stiffness == 0:
                        stiffness = float(child.value)
                        print(f"DEBUG: Found stiffness: {stiffness}")
                    else:
                        mass = float(child.value)
                        print(f"DEBUG: Found mass: {mass}")
        
        if shape_def is None:
            raise ValueError("Body must have a shape definition")
        
        print(f"DEBUG: Creating BodyDef with shape={shape_def[0]}, dims={shape_def[1]}, stiffness={stiffness}, mass={mass}")
        return BodyDef(shape_def[0], shape_def[1], stiffness, mass)
    
    def eval_shape_def(self, node):
        print("DEBUG: Evaluating shape_def...")
        print(f"DEBUG: Shape_def node children: {[child.value if isinstance(child, Token) else getattr(child, 'data', type(child)) for child in node.children]}")
        
        child = node.children[0]
        
        if isinstance(child, Token):
            print(f"DEBUG: Found token shape: {child.value}")
            if child.value == "table":
                return ("table", (20, 10, 20))
            elif child.value == "worm":
                return ("worm", (30, 4, 4))
            else:
                raise ValueError(f"Unknown shape: {child.value}")
        elif hasattr(child, 'data'):
            print(f"DEBUG: Found compound shape: {child.data}")
            if child.data == 'box':
                dims = [float(c.value) for c in child.children if isinstance(c, Token)]
                return ("box", tuple(dims))
            elif child.data == 'cylinder':
                dims = [float(c.value) for c in child.children if isinstance(c, Token)]
                return ("cylinder", tuple(dims))
            else:
                raise ValueError(f"Unknown shape type: {child.data}")
        else:
            raise ValueError("Invalid shape definition")
    
    def eval_parts(self, node):
        parts = []
        for child in node.children:
            if hasattr(child, 'data') and child.data == 'part_def':
                parts.append(self.eval_part_def(child))
        return parts
    
    def eval_part_def(self, node):
        name = node.children[0].value
        part_type = node.children[1].value  # "leg" or "body_part"
        
        position = None
        size = None
        stiffness = 0
        mass = 0
        
        for child in node.children[2:]:
            if hasattr(child, 'data'):
                if child.data == 'position':
                    position = tuple(float(c.value) for c in child.children if isinstance(c, Token))
                elif child.data == 'size':
                    size = tuple(float(c.value) for c in child.children if isinstance(c, Token))
            elif isinstance(child, Token) and child.type == 'NUMBER':
                if stiffness == 0:
                    stiffness = float(child.value)
                else:
                    mass = float(child.value)
        
        return PartDef(name, part_type, position, size, stiffness, mass)
    
    def eval_actuator(self, node):
        gait = None
        frequency = 0
        forces = {}
        
        for child in node.children:
            if hasattr(child, 'data'):
                if child.data == 'gait_type':
                    gait = child.children[0].value
                elif child.data == 'force_def':
                    forces = self.eval_force_def(child)
            elif isinstance(child, Token) and child.type == 'NUMBER':
                frequency = float(child.value)
        
        return ActuatorDef(gait, frequency, forces)
    
    def eval_force_def(self, node):
        numbers = [float(c.value) for c in node.children if isinstance(c, Token) and c.type == 'NUMBER']
        return {
            'lift': numbers[0] if len(numbers) > 0 else 18.0,
            'push': numbers[1] if len(numbers) > 1 else 15.0,
            'swing': numbers[2] if len(numbers) > 2 else 6.0
        }

@v_args(inline=True)
class ModelitoTransformer(Transformer):
    def start(self, robot_def):
        print("DEBUG: Transforming start node")
        return robot_def
        
    def robot_def(self, name, *args):
        print(f"DEBUG: Transforming robot_def with name {name} and {len(args)} args")
        body = None
        parts = []
        actuator = None
        
        for arg in args:
            if isinstance(arg, BodyDef):
                body = arg
            elif isinstance(arg, list):
                parts = arg
            elif isinstance(arg, ActuatorDef):
                actuator = arg
                
        return RobotDef(str(name), body, parts, actuator)
    
    def body(self, shape_def, stiffness, mass):
        print(f"DEBUG: Transforming body with shape={shape_def}, stiffness={stiffness}, mass={mass}")
        shape_type, dimensions = shape_def
        return BodyDef(shape_type, dimensions, float(stiffness), float(mass))
    
    def parts(self, *part_defs):
        print(f"DEBUG: Transforming parts with {len(part_defs)} parts")
        return list(part_defs)
    
    def create_part(self, name, type_, attrs):
        print(f"DEBUG: Creating part {name} of type {type_}")
        position, size, stiffness, mass = attrs
        return PartDef(str(name), str(type_), position, size, float(stiffness), float(mass))

    def leg_type(self):
        return "leg"

    def body_part_type(self):
        return "body_part"

    def part_attrs(self, position, size, stiffness, mass):
        print(f"DEBUG: Processing part attributes")
        return (position, size, float(stiffness), float(mass))
    
    def actuator(self, gait, frequency, forces):
        print(f"DEBUG: Transforming actuator with gait={gait}")
        return ActuatorDef(str(gait), float(frequency), forces)
    
    def create_box_shape(self, box_dims):
        x, y, z = box_dims
        print(f"DEBUG: Creating box shape ({x}, {y}, {z})")
        return ("box", (float(x), float(y), float(z)))
        
    def box_shape(self, x, y, z):
        print(f"DEBUG: Processing box dimensions")
        return (float(x), float(y), float(z))
        
    def create_cylinder_shape(self, cylinder_dims):
        radius, height = cylinder_dims
        print(f"DEBUG: Creating cylinder shape (r={radius}, h={height})")
        return ("cylinder", (float(radius), float(height)))
        
    def cylinder_shape(self, radius, height):
        print(f"DEBUG: Processing cylinder dimensions")
        return (float(radius), float(height))
        
    def table_shape(self):
        print("DEBUG: Creating table shape")
        return ("table", (20, 10, 20))
        
    def worm_shape(self):
        print("DEBUG: Creating worm shape")
        return ("worm", (30, 4, 4))
    
    def SIGNED_NUMBER(self, token):
        return float(token)
        
    def position(self, x, y, z):
        return (float(x), float(y), float(z))
    
    def size(self, x, y, z):
        return (float(x), float(y), float(z))
    
    def force_def(self, lift, push, swing):
        return {"lift": float(lift), "push": float(push), "swing": float(swing)}
    
    def quadruped_gait(self):
        print("DEBUG: Using quadruped gait")
        return "quadruped"
        
    def worm_gait(self):
        print("DEBUG: Using worm gait")
        return "worm"
        
    def custom_gait(self):
        print("DEBUG: Using custom gait")
        return "custom"
    
    def NAME(self, token):
        return str(token)
    
    def NUMBER(self, token):
        return float(token)

parser = Lark(grammar, parser='lalr', transformer=ModelitoTransformer())
