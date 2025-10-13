# MODELITO by the MODELO TEAM
import os
import sys
# if not linux remove this
if sys.platform.startswith('linux'):
    os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

from dataclasses import dataclass
from typing import Any, Callable, List, Dict, Tuple
import math
import numpy as np
from vedo import Volume, show, ProgressBar, Box
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QSplitter,
                             QLabel, QStatusBar, QToolBar, QFileDialog, QMessageBox,
                             QProgressBar, QMenu, QToolButton)
from PyQt6.QtCore import Qt, QTimer, QTime
from PyQt6.QtGui import (QFont, QTextCharFormat, QColor, QSyntaxHighlighter, 
                         QAction, QKeySequence, QPixmap)
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

# ============ GEOMETRY CREATION ============

def create_geometry(robot: RobotDef, resolution=20):
    """Create voxel geometry from robot definition"""
    
    # Determine grid size
    max_dim = max(robot.body.dimensions)
    X, Y, Z = np.mgrid[0:max_dim:resolution*1j, 
                       0:max_dim:resolution*1j, 
                       0:max_dim:resolution*1j]
    
    # Create body
    if robot.body.shape == "table":
        body_mask = create_table_body(X, Y, Z)
    elif robot.body.shape == "worm":
        body_mask = create_worm_body(X, Y, Z)
    elif robot.body.shape == "cylinder":
        body_mask = create_cylinder_body(X, Y, Z, robot.body.dimensions)
    else:
        body_mask = create_box_body(X, Y, Z, robot.body.dimensions)
    
    # Create parts (legs, etc)
    parts_masks = {}
    for part in robot.parts:
        parts_masks[part.name] = create_part(X, Y, Z, part)
    
    # Combine all
    full_shape = body_mask
    for mask in parts_masks.values():
        full_shape = full_shape | mask
    
    scalar_field = np.where(full_shape, 0, 1)
    
    return X, Y, Z, scalar_field, body_mask, parts_masks

def create_table_body(X, Y, Z):
    """Create table-shaped body"""
    return (X >= 3) & (X <= 17) & (Y >= 6) & (Y <= 8) & (Z >= 3) & (Z <= 17)

def create_worm_body(X, Y, Z):
    """Create worm-shaped body"""
    return (np.abs(Y - 1.5) < 1.5) & (np.abs(Z - 2) < 1.5)

def create_cylinder_body(X, Y, Z, dims):
    """Create cylinder-shaped body"""
    radius, height = dims
    return ((X - X.mean())**2 + (Z - Z.mean())**2 < radius**2) & (abs(Y) < height/2)

def create_box_body(X, Y, Z, dims):
    """Create box-shaped body"""
    w, h, d = dims
    return (abs(X - X.mean()) < w/2) & (abs(Y) < h/2) & (abs(Z - Z.mean()) < d/2) & (Z >= 0) & (Z <= d)

def create_part(X, Y, Z, part: PartDef):
    """Create a part (leg, etc) from definition"""
    px, py, pz = part.position
    sx, sy, sz = part.size
    
    return (X >= px) & (X <= px + sx) & \
           (Y >= py) & (Y <= py + sy) & \
           (Z >= pz) & (Z <= pz + sz)

# ============ SIMULATION ============

def simulate_robot(robot: RobotDef, duration=60.0, nsteps=1200):
    """Run physics simulation"""
    
    print("Creating geometry...")
    X, Y, Z, scalar_field, body_mask, parts_masks = create_geometry(robot)
    
    # Generate mesh
    volume = Volume(scalar_field)
    voxel_mesh = volume.legosurface(-0.5, 0.5)
    
    initial_points = voxel_mesh.vertices.copy()
    num_vertices = len(initial_points)
    
    x_coords = initial_points[:, 0]
    y_coords = initial_points[:, 1]
    z_coords = initial_points[:, 2]
    
    print(f"Mesh created: {num_vertices} vertices")
    
    # Pre-calculate neighbors
    print("Pre-calculating neighbors...")
    tree = cKDTree(initial_points)
    neighbors = []
    initial_distances = []
    
    for idx in range(num_vertices):
        indices = tree.query_ball_point(initial_points[idx], 3.0)
        indices = [i for i in indices if i != idx]
        neighbors.append(np.array(indices, dtype=np.int32))
        dists = np.array([np.linalg.norm(initial_points[idx] - initial_points[i]) for i in indices])
        initial_distances.append(dists)
    
    print(f"Done! Average neighbors: {np.mean([len(n) for n in neighbors]):.1f}")
    
    # Assign properties
    stiffness_per_vertex = np.zeros(num_vertices)
    mass_per_vertex = np.ones(num_vertices)
    vertex_part = []
    
    stiffness_field = np.zeros_like(X)
    mass_field = np.ones_like(X)
    
    # Body properties
    stiffness_field[body_mask] = robot.body.stiffness
    mass_field[body_mask] = robot.body.mass
    
    # Part properties
    for part in robot.parts:
        mask = parts_masks[part.name]
        stiffness_field[mask] = part.stiffness
        mass_field[mask] = part.mass
    
    # Sample at vertices
    max_dim = max(robot.body.dimensions)
    resolution = X.shape[0]
    
    for idx in range(num_vertices):
        xi = int(np.clip((x_coords[idx] / max_dim) * (resolution - 1), 0, resolution - 1))
        yi = int(np.clip((y_coords[idx] / max_dim) * (resolution - 1), 0, resolution - 1))
        zi = int(np.clip((z_coords[idx] / max_dim) * (resolution - 1), 0, resolution - 1))
        stiffness_per_vertex[idx] = stiffness_field[xi, yi, zi]
        mass_per_vertex[idx] = mass_field[xi, yi, zi]
        
        # Classify vertex by part
        part_name = "body"
        for pname, mask in parts_masks.items():
            if mask[xi, yi, zi]:
                part_name = pname
                break
        vertex_part.append(part_name)
    
    # Color by stiffness
    min_stiff = stiffness_per_vertex.min()
    max_stiff = stiffness_per_vertex.max()
    if max_stiff > min_stiff:
        normalized_stiffness = (stiffness_per_vertex - min_stiff) / (max_stiff - min_stiff)
    else:
        normalized_stiffness = np.ones(num_vertices) * 0.5
    
    cmap = plt.cm.coolwarm
    colors = cmap(normalized_stiffness)[:, :3]
    colors_255 = (colors * 255).astype(np.uint8)
    voxel_mesh.pointcolors = colors_255
    
    # Pre-compute part indices
    part_indices = {}
    for part in robot.parts:
        part_indices[part.name] = np.array([i for i, p in enumerate(vertex_part) if p == part.name], dtype=np.int64)
    body_indices = np.array([i for i, p in enumerate(vertex_part) if p == "body"], dtype=np.int64)
    
    # Identify foot vertices
    foot_indices = {}
    for part_name, indices in part_indices.items():
        if len(indices) > 0:
            leg_y = initial_points[indices, 1]
            min_y = np.min(leg_y)
            foot_mask = leg_y < min_y + 0.5
            foot_indices[part_name] = indices[foot_mask].astype(np.int64)
    
    # Floor
    floor = Box([100, -1.5, 10], length=400, width=2, height=40).c('grey').alpha(0.3)
    
    # Physics state
    velocity = np.zeros_like(initial_points)
    local_deformation = np.zeros_like(initial_points)
    global_position = np.array([0.0, 0.0, 0.0])
    global_velocity = np.array([0.0, 0.0, 0.0])
    
    # Pre-allocate arrays
    actuation_force = np.zeros_like(initial_points)
    restoring_force = np.zeros_like(initial_points)
    volume_force = np.zeros_like(initial_points)
    floor_force = np.zeros_like(initial_points)
    
    # Material properties
    damping = 0.6
    gravity = np.array([0, -8.0, 0])
    volume_stiffness = 25.0
    
    # Floor properties
    floor_y = -0.5
    floor_stiffness = 250.0
    floor_damping = 4.0
    
    friction_forward = 0.25
    friction_backward = 3.0
    friction_lateral = 0.8
    
    # Pre-compute gravity
    gravity_per_vertex = np.outer(mass_per_vertex, gravity)
    
    # Simulation parameters
    dt = duration / nsteps
    
    print(f"Starting simulation: {duration}s, {nsteps} steps")
    
    # MAIN LOOP
    pb = ProgressBar(0, nsteps, c='blue')
    
    for step in range(nsteps):
        t = step * dt
        
        # Current positions
        current_points = initial_points + global_position + local_deformation
        
        # Actuation based on gait
        actuation_force[:] = 0
        
        if robot.actuator and robot.actuator.gait == "quadruped":
            apply_quadruped_gait(actuation_force, t, robot.actuator, 
                               part_indices, foot_indices, body_indices)
        elif robot.actuator and robot.actuator.gait == "worm":
            apply_worm_gait(actuation_force, t, robot.actuator, x_coords, num_vertices)
        
        # Restoring force
        restoring_force = -stiffness_per_vertex[:, np.newaxis] * local_deformation
        
        # Volume preservation
        volume_force[:] = 0
        for idx in range(num_vertices):
            if len(neighbors[idx]) == 0:
                continue
            
            neighbor_idx = neighbors[idx]
            diffs = current_points[neighbor_idx] - current_points[idx]
            current_dists = np.linalg.norm(diffs, axis=1)
            
            valid = current_dists > 0.01
            if np.any(valid):
                target_dists = initial_distances[idx][valid]
                current_dists_valid = current_dists[valid]
                diffs_valid = diffs[valid]
                
                diff_ratios = (current_dists_valid - target_dists) / current_dists_valid
                volume_force[idx] = np.sum(volume_stiffness * diff_ratios[:, np.newaxis] * diffs_valid, axis=0)
        
        # Gravity
        gravity_force = gravity_per_vertex
        
        # Floor collision
        floor_force[:] = 0
        net_forward_push = 0.0
        
        below_floor_mask = current_points[:, 1] < floor_y + 0.3
        below_floor_indices = np.where(below_floor_mask)[0]
        
        if len(below_floor_indices) > 0:
            penetrations = np.maximum(0, floor_y + 0.3 - current_points[below_floor_indices, 1])
            normal_forces = floor_stiffness * penetrations
            
            floor_force[below_floor_indices, 1] = normal_forces
            
            # Friction
            vel_x = velocity[below_floor_indices, 0]
            friction_x = np.where(vel_x > 0, 
                                 -friction_forward * normal_forces,
                                 -friction_backward * normal_forces) * np.sign(vel_x)
            friction_x[np.abs(vel_x) <= 0.01] = 0
            floor_force[below_floor_indices, 0] += friction_x
            
            vel_z = velocity[below_floor_indices, 2]
            friction_z = -friction_lateral * normal_forces * np.sign(vel_z)
            friction_z[np.abs(vel_z) <= 0.01] = 0
            floor_force[below_floor_indices, 2] += friction_z
            
            # Net push
            push_mask = actuation_force[below_floor_indices, 1] < -10.0
            if np.any(push_mask):
                push_forces = np.abs(actuation_force[below_floor_indices[push_mask], 1])
                net_forward_push = np.sum(push_forces) * 0.5
            
            # Vertical damping
            downward_mask = velocity[below_floor_indices, 1] < 0
            downward_indices = below_floor_indices[downward_mask]
            floor_force[downward_indices, 1] -= floor_damping * velocity[downward_indices, 1]
        
        # Damping
        damping_force = -damping * velocity
        
        # Total force
        total_force = actuation_force + restoring_force + volume_force + gravity_force + floor_force + damping_force
        
        # Update
        acceleration = total_force / mass_per_vertex[:, np.newaxis]
        velocity += acceleration * dt
        velocity *= 0.92
        
        local_deformation += velocity * dt
        
        # Constrain deformation
        deform_mag = np.linalg.norm(local_deformation, axis=1, keepdims=True)
        max_local_deformation = 3.0
        scale = np.minimum(1.0, max_local_deformation / (deform_mag + 1e-8))
        local_deformation *= scale
        
        # Global movement
        global_velocity[0] += net_forward_push * dt / (np.sum(mass_per_vertex) * 8)
        global_velocity *= 0.98
        global_position += global_velocity * dt
        
        # Update mesh
        voxel_mesh.vertices = current_points
        
        # Display
        plt_instance = show(voxel_mesh, floor,
                           axes=1,
                           bg='white',
                           interactive=False,
                           resetcam=(step==0))
        
        if step == 0:
            plt_instance.camera.SetPosition(-80, 20, 40)
            plt_instance.camera.SetFocalPoint(50, 5, 10)
            plt_instance.camera.SetViewUp(0, 1, 0)
        
        plt_instance.clear()
        
        if step % 10 == 0:
            pb.print(f"Time: {t:.2f}s | Pos: {global_position[0]:.2f}")

def apply_quadruped_gait(actuation_force, t, actuator, part_indices, foot_indices, body_indices):
    """Apply quadruped walking gait"""
    freq = actuator.frequency
    forces = actuator.forces
    
    # Body motion
    body_lean = np.sin(2*np.pi*freq*t) * 2.0
    actuation_force[body_indices, 0] += body_lean
    body_sway = np.sin(2*np.pi*freq*t) * 2.0
    actuation_force[body_indices, 2] += body_sway
    
    # Find leg pairs
    legs = sorted(list(part_indices.keys()))
    if len(legs) >= 4:
        # For 6 legs, use tripod gait (front-back on one side + middle on other)
        if len(legs) == 6:
            pair1 = [legs[0], legs[2], legs[5]]  # front_left, mid_left, back_right
            pair2 = [legs[1], legs[3], legs[4]]  # front_right, mid_right, back_left
        else:
            # For 4 legs, use diagonal pairs
            pair1 = [legs[0], legs[3]]  # e.g., leg1, leg4
            pair2 = [legs[1], legs[2]]  # e.g., leg2, leg3
        
        # Pair 1
        phase1 = 2*np.pi*freq*t
        lift_cycle1 = np.sin(phase1)
        
        for leg_name in pair1:
            if leg_name in part_indices:
                leg_idx = part_indices[leg_name].astype(np.int64)
                foot_idx = foot_indices.get(leg_name, leg_idx).astype(np.int64)
                
                if lift_cycle1 > 0:
                    actuation_force[leg_idx, 1] += forces['lift'] * lift_cycle1
                    actuation_force[leg_idx, 0] += 15.0 * lift_cycle1
                else:
                    actuation_force[foot_idx, 1] -= forces['push'] * abs(lift_cycle1)
                    actuation_force[foot_idx, 0] -= 15.0 * abs(lift_cycle1)
        
        # Pair 2
        phase2 = 2*np.pi*freq*t + np.pi
        lift_cycle2 = np.sin(phase2)
        
        for leg_name in pair2:
            if leg_name in part_indices:
                leg_idx = part_indices[leg_name]
                foot_idx = foot_indices.get(leg_name, leg_idx)
                
                if lift_cycle2 > 0:
                    actuation_force[leg_idx, 1] += forces['lift'] * lift_cycle2
                    actuation_force[leg_idx, 0] += 15.0 * lift_cycle2
                else:
                    actuation_force[foot_idx, 1] -= forces['push'] * abs(lift_cycle2)
                    actuation_force[foot_idx, 0] -= 15.0 * abs(lift_cycle2)

def apply_worm_gait(actuation_force, t, actuator, x_coords, num_vertices):
    """Apply worm undulation gait"""
    freq = actuator.frequency
    forces = actuator.forces
    
    for idx in range(num_vertices):
        x_pos = x_coords[idx]
        
        wave_speed = freq
        wavelength = 10.0
        phase = 2*np.pi*(wave_speed*t - x_pos/wavelength)
        oscillation = np.sin(phase)
        
        position_factor = (x_pos / 30.0) ** 0.5
        
        # Lateral
        lateral_force = forces['swing'] * oscillation * (0.5 + position_factor)
        actuation_force[idx, 2] += lateral_force
        
        # Vertical
        y_phase = phase + np.pi/2
        y_oscillation = np.sin(y_phase)
        actuation_force[idx, 1] += forces['lift'] * y_oscillation * (0.5 + position_factor)

# ============ IDE ============

class ModelitoHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#8B4513"))
        self.keyword_format.setFontWeight(QFont.Weight.Bold)
        
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#CD853F"))
        
        self.keywords = [
            'robot', 'body', 'parts', 'actuator', 'leg', 'body_part',
            'shape', 'position', 'size', 'stiffness', 'mass',
            'gait', 'frequency', 'forces', 'lift', 'push', 'swing',
            'quadruped', 'worm', 'box', 'table', 'cylinder'
        ]
        
    def highlightBlock(self, text):
        for keyword in self.keywords:
            index = text.find(keyword)
            while index >= 0:
                length = len(keyword)
                self.setFormat(index, length, self.keyword_format)
                index = text.find(keyword, index + length)
        
        import re
        for match in re.finditer(r'\b\d+\.?\d*\b', text):
            self.setFormat(match.start(), match.end() - match.start(), self.number_format)

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        
        # Match IDE window size
        self.setFixedSize(1200, 800)
        
        # Set window background
        self.setStyleSheet("""
            QWidget {
                background-color: #FFF8DC;
                border: 3px solid #8B4513;
                border-radius: 15px;
            }
            QLabel {
                background: transparent;
                border: none;
                padding: 10px;
            }
        """)
        
        # Position at IDE window position
        self.move(100, 100)  # Same as IDE window position
        
        layout = QVBoxLayout(self)
        layout.setSpacing(40)
        layout.setContentsMargins(100, 100, 100, 100)
        
        # Add spacer at top
        layout.addStretch(1)
        
        # Title
        title_label = QLabel("MODELITO")
        title_label.setStyleSheet("""
            font-size: 72px;
            font-weight: bold;
            color: #8B4513;
        """)
        layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap("MODELITO_LOGO.svg")
        logo_label.setPixmap(logo_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        # Add spacer at bottom
        layout.addStretch(1)
        layout.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Subtitle
        subtitle_label = QLabel("Soft Robotics Simulator")
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #8B4513;
                font-size: 36px;
                font-style: italic;
            }
        """)
        layout.addWidget(subtitle_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Start timer to close splash
        self.timer = QTimer()
        self.timer.timeout.connect(self.close)
        self.timer.setSingleShot(True)
        self.timer.start(2000)  # Show for 2 seconds
        
    def center_on_screen(self):
        # Get the screen geometry
        screen = QApplication.primaryScreen().geometry()
        # Calculate center position
        center_x = (screen.width() - self.width()) // 2
        center_y = (screen.height() - self.height()) // 2
        # Move window to center
        self.move(center_x, center_y)
    
    def update_progress(self):
        self.progress_value += 1
        self.progress.setValue(self.progress_value)
        if self.progress_value >= 100:
            self.timer.stop()
            self.close()

class ModelitoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Show splash screen
        self.splash = SplashScreen()
        self.splash.show()
        QApplication.processEvents()  # Make sure splash is displayed
        
        # Keep splash visible for 2 seconds
        start_time = QTime.currentTime()
        while start_time.msecsTo(QTime.currentTime()) < 2000:
            QApplication.processEvents()
        self.setWindowTitle("Modelito Soft Robotics IDE")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setStyleSheet("QMainWindow { background-color: #FFF8DC; }")
        
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #FFF8DC;")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        self.create_toolbar()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Code editor
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        code_label = QLabel("📝 Robot Definition")
        code_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; color: #8B4513; background-color: #FFF8DC;")
        left_layout.addWidget(code_label)
        
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Consolas", 11))
        self.code_editor.setPlaceholderText("Write your robot definition here...")
        self.code_editor.setStyleSheet("""
            QTextEdit {
                background-color: #FFFAF0;
                color: #5C4033;
                border: 2px solid #D2B48C;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        
        self.highlighter = ModelitoHighlighter(self.code_editor.document())
        
        left_layout.addWidget(self.code_editor)
        
        # Run button
        self.run_button = QPushButton("🤖 Simulate Robot (Ctrl+Enter)")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #D2691E;
                color: #FFFAF0;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
                border-radius: 8px;
                border: 2px solid #8B4513;
            }
            QPushButton:hover {
                background-color: #CD853F;
            }
        """)
        self.run_button.clicked.connect(self.run_simulation)
        left_layout.addWidget(self.run_button)
        
        splitter.addWidget(left_widget)
        
        # Output
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        output_label = QLabel("📊 Output")
        output_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; color: #8B4513; background-color: #FFF8DC;")
        right_layout.addWidget(output_label)
        
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Consolas", 10))
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #FAF0E6;
                color: #5C4033;
                border: 2px solid #D2B48C;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        right_layout.addWidget(self.output_area)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        self.set_example_code()
        
    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar { 
                spacing: 5px; 
                padding: 8px; 
                background-color: #F5DEB3;
                border-bottom: 2px solid #D2B48C;
            }
            QToolBar QToolButton {
                background-color: #FAEBD7;
                color: #8B4513;
                border: 1px solid #D2B48C;
                border-radius: 5px;
                padding: 5px 10px;
                margin: 2px;
            }
            QToolBar QToolButton:hover {
                background-color: #FFE4B5;
            }
        """)
        self.addToolBar(toolbar)
        
        new_action = QAction("📄 New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_file)
        toolbar.addAction(new_action)
        
        open_action = QAction("📂 Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        save_action = QAction("💾 Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        self.show_examples_action = QAction("Examples", self)
        self.show_examples_action.triggered.connect(self.show_examples)
        help_menu = QMenu("Help", self)
        help_menu.addAction(self.show_examples_action)
        
        # About action
        self.about_action = QAction("About Modelito", self)
        self.about_action.triggered.connect(self.show_about)
        help_menu.addAction(self.about_action)
        
        # Help menu button
        help_button = QToolButton()
        help_button.setText("❓ Help")
        help_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        help_button.setStyleSheet("""
            QToolButton {
                background-color: #FFF8DC;
                border: 1px solid #8B4513;
                padding: 5px 10px;
                border-radius: 4px;
                color: #8B4513;
            }
            QToolButton:hover {
                background-color: #DEB887;
            }
            QToolButton::menu-indicator {
                image: none;
            }
        """)
        
        help_menu = QMenu(self)
        help_menu.setStyleSheet("""
            QMenu {
                background-color: #FFF8DC !important;
                border: 1px solid #8B4513;
                padding: 5px;
            }
            QMenu::item {
                background-color: #FFF8DC;
                padding: 8px 20px;
                color: #8B4513;
                border: none;
            }
            QMenu::item:selected {
                background-color: #DEB887;
                color: #8B4513;
            }
            QMenu::separator {
                height: 1px;
                background-color: #8B4513;
                margin: 5px 15px;
            }
        """)
        
        help_action = QAction("Documentation", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        examples_action = QAction("Examples", self)
        examples_action.triggered.connect(self.show_examples)
        help_menu.addAction(examples_action)
        
        about_action = QAction("About Modelito", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        help_button.setMenu(help_menu)
        toolbar.addWidget(help_button)
        
    def set_example_code(self):
        example = """// Welcome to Modelito Soft Robotics!
// Press Ctrl+Enter to simulate

robot QuadrupedBot {
    body {
        shape: table
        stiffness: 40.0
        mass: 2.5
    }
    
    parts {
        leg1: leg {
            position: (3, 0, 3)
            size: (3, 6, 3)
            stiffness: 25.0
            mass: 0.8
        }
        
        leg2: leg {
            position: (14, 0, 3)
            size: (3, 6, 3)
            stiffness: 25.0
            mass: 0.8
        }
        
        leg3: leg {
            position: (3, 0, 14)
            size: (3, 6, 3)
            stiffness: 25.0
            mass: 0.8
        }
        
        leg4: leg {
            position: (14, 0, 14)
            size: (3, 6, 3)
            stiffness: 25.0
            mass: 0.8
        }
    }
    
    actuator {
        gait: quadruped
        frequency: 0.35
        forces: {
            lift: 22.0
            push: 25.0
            swing: 6.0
        }
    }
}
"""
        self.code_editor.setText(example)
        
    def new_file(self):
        self.code_editor.clear()
        self.output_area.clear()
        self.status_bar.showMessage("New file created")
        
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Modelito Robot Files (*.robot);;All Files (*)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.code_editor.setText(f.read())
                self.status_bar.showMessage(f"Opened: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
                
    def save_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Modelito Robot Files (*.robot);;All Files (*)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.code_editor.toPlainText())
                self.status_bar.showMessage(f"Saved: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")
                
    def show_examples(self):
        examples = """
=== MODELITO ROBOT EXAMPLES ===

1. Simple Quadruped:
robot Walker {
    body {
        shape: table
        stiffness: 40.0
        mass: 2.5
    }
    parts {
        leg1: leg { position: (3, 0, 3) size: (3, 6, 3) stiffness: 25.0 mass: 0.8 }
        leg2: leg { position: (14, 0, 3) size: (3, 6, 3) stiffness: 25.0 mass: 0.8 }
        leg3: leg { position: (3, 0, 14) size: (3, 6, 3) stiffness: 25.0 mass: 0.8 }
        leg4: leg { position: (14, 0, 14) size: (3, 6, 3) stiffness: 25.0 mass: 0.8 }
    }
    actuator {
        gait: quadruped
        frequency: 0.35
        forces: { lift: 22.0 push: 25.0 swing: 6.0 }
    }
}

2. Undulating Worm:
robot Crawler {
    body {
        shape: worm
        stiffness: 35.0
        mass: 2.0
    }
    parts {
    }
    actuator {
        gait: worm
        frequency: 0.5
        forces: { lift: 10.0 push: 15.0 swing: 8.0 }
    }
}

3. Soft Quadruped (More Flexible):
robot SoftBot {
    body {
        shape: table
        stiffness: 20.0
        mass: 2.0
    }
    parts {
        leg1: leg { position: (3, 0, 3) size: (3, 6, 3) stiffness: 15.0 mass: 0.6 }
        leg2: leg { position: (14, 0, 3) size: (3, 6, 3) stiffness: 15.0 mass: 0.6 }
        leg3: leg { position: (3, 0, 14) size: (3, 6, 3) stiffness: 15.0 mass: 0.6 }
        leg4: leg { position: (14, 0, 14) size: (3, 6, 3) stiffness: 15.0 mass: 0.6 }
    }
    actuator {
        gait: quadruped
        frequency: 0.3
        forces: { lift: 18.0 push: 20.0 swing: 5.0 }
    }
}
"""
        QMessageBox.information(self, "Examples", examples)
        
    def show_about(self):
        about_text = """
        🤖 Modelito Soft Robotics Simulator
        Version 1.0
        
        Created by the MODELO TEAM
        
        A powerful simulator for designing and testing
        soft robotic systems with various gaits and
        configurations.
        
        Learn more at: docs/technical_guide.md
        """
        
        about_dialog = QMessageBox(self)
        about_dialog.setWindowTitle("About Modelito")
        about_dialog.setText(about_text)
        about_dialog.setIconPixmap(QPixmap("MODELITO_LOGO.svg").scaled(
            128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        about_dialog.exec()
    
    def show_help(self):
        help_text = """
=== MODELITO SOFT ROBOTICS LANGUAGE ===

GAIT TYPES:
- quadruped: Four-legged walking pattern
  * Alternates diagonal pairs of legs
  * Like how a dog or horse walks
  * Best for stable, walking robots

- worm: Undulating movement pattern
  * Creates wave-like motion through body
  * Like how a caterpillar moves
  * Best for flexible, snake-like robots

STRUCTURE:
robot <Name> {
    body { ... }
    parts { ... }
    actuator { ... }
}

BODY:
  shape: table | worm | box(w,h,d)
  stiffness: <number>  - Higher = more rigid
  mass: <number>       - Weight of body

PARTS:
  <name>: leg | body_part {
      position: (x, y, z)
      size: (width, height, depth)
      stiffness: <number>
      mass: <number>
  }

ACTUATOR:
  gait: quadruped | worm
  frequency: <number>  - Speed of motion
  forces: {
      lift: <number>   - Upward force
      push: <number>   - Downward/backward force
      swing: <number>  - Lateral force
  }

TIPS:
- Lower stiffness = more flexible/bouncy
- Higher frequency = faster movement
- Experiment with force values!
- Use Comments with //

SHORTCUTS:
  Ctrl+Enter - Run simulation
  Ctrl+N     - New file
  Ctrl+O     - Open file
  Ctrl+S     - Save file
"""
        QMessageBox.information(self, "Help", help_text)
        
    def run_simulation(self):
        code = self.code_editor.toPlainText().strip()
        
        if not code:
            self.output_area.setText("⚠️ No code to run!")
            return
            
        self.output_area.clear()
        self.output_area.append("🚀 Parsing robot definition...\n")
        self.status_bar.showMessage("Parsing...")
        
        try:
            # Remove comments
            lines = code.split('\n')
            code_clean = '\n'.join(line.split('//')[0] for line in lines)
            
            # Parse and transform
            robot = parser.parse(code_clean)
            self.output_area.append("✅ Parsing successful!\n")
            
            # No need for interpreter since the transformer already created the RobotDef
            if not isinstance(robot, RobotDef):
                raise ValueError("Failed to create robot definition")
                
            self.output_area.append(f"✅ Robot '{robot.name}' loaded!\n")
            self.output_area.append(f"   Body: {robot.body.shape}\n")
            self.output_area.append(f"   Parts: {len(robot.parts)}\n")
            self.output_area.append(f"   Gait: {robot.actuator.gait if robot.actuator else 'None'}\n\n")
            
            self.status_bar.showMessage("Starting simulation...")
            self.output_area.append("🤖 Starting physics simulation...\n")
            
            # Run simulation
            simulate_robot(robot, duration=60.0, nsteps=1200)
            
            self.output_area.append("\n✅ Simulation complete!\n")
            self.status_bar.showMessage("✅ Simulation completed!")
            
        except Exception as e:
            import traceback
            self.output_area.append(f"❌ Error: {str(e)}\n")
            self.output_area.append(f"\n📋 Traceback:\n{traceback.format_exc()}")
            self.status_bar.showMessage("❌ Error occurred!")
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.run_simulation()
        else:
            super().keyPressEvent(event)

# ============ MAIN ============

def main():
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#FFF8DC"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#5C4033"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#FFFAF0"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#F5DEB3"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FAEBD7"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#5C4033"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#5C4033"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#F5DEB3"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#5C4033"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#D2691E"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFAF0"))
    app.setPalette(palette)
    
    window = ModelitoGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
