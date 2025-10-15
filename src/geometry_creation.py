import numpy as np
from vedo import Volume, show, ProgressBar, Box

from language import RobotDef, PartDef

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

