import numpy as np
from vedo import Volume, show, ProgressBar, Box
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

from language import RobotDef
from geometry_creation import create_geometry

# ============ SIMULATION ============

def plot_gait_data(time_points, positions, heights, part_names):
    """Plot gait visualization in a separate window"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Soft Robot Gait Analysis', fontsize=14)
    
    # Set a clean style
    for ax in (ax1, ax2):
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f8f8f8')
    
    # Plot forward progress
    ax1.plot(time_points, positions, 'b-', label='Forward Position')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position')
    ax1.set_title('Forward Progress')
    ax1.grid(True)
    ax1.legend()
    
    # Plot height variations for each part
    for part_name, height_data in heights.items():
        ax2.plot(time_points, height_data, label=part_name)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (Y)')
    ax2.set_title('Part Height Variations')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show(block=False)

def simulate_robot(robot: RobotDef, duration=60.0, nsteps=1200, plot_gait=True):
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
    
    # Data collection for gait analysis
    time_points = []
    positions = []
    heights = {name: [] for name in part_indices.keys()}
    heights['body'] = []
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
            
            # Collect data for gait analysis
            time_points.append(t)
            positions.append(global_position[0])
            
            # Record heights of each part
            for part_name, indices in part_indices.items():
                avg_height = np.mean(current_points[indices, 1])
                heights[part_name].append(avg_height)
            avg_body_height = np.mean(current_points[body_indices, 1])
            heights['body'].append(avg_body_height)
    
    # Plot gait analysis if requested
    if plot_gait:
        plot_gait_data(time_points, positions, heights, part_indices.keys())

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

