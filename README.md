# Modelito Soft Robotics Simulator

A Python-based soft robotics simulator for designing and testing deformable robots with various gaits and configurations.

Created by the MODELO KIND ENGINEERING TEAM:

Daniel Motilla (M0TH)
Frederick Ayala
Davide Vigano
Oyvind Soroy (Colonthree)
Drake (Yoshi)

We hope you have fun your our software :3

## Quick Start

1. Install dependencies:
```bash
conda create -n MODELITO python=3.13
conda activate MODELITO
pip install numpy scipy matplotlib vedo PyQt6 lark
```

2. Run the simulator:
```bash
python modelito.py
```

3. Create a robot definition (e.g., `my_robot.robot`) and load it in the editor.

## Example Robots

The repository includes example robots:
- `quadruped_robot.robot`: A four-legged walking robot
- `worm_robot.robot`: A segmented robot with undulating motion

## Key Features

- Visual design interface with syntax highlighting
- Real-time physics simulation
- Multiple gait patterns (quadruped, worm)
- Customizable robot parameters
- 3D visualization with vedo

## Robot Definition Language

Modelito uses a custom language for defining robots. Basic structure:

```
robot MyRobot {
    body {
        shape: box(w, h, d) | cylinder(r, h) | table | worm
        stiffness: <number>
        mass: <number>
    }
    
    parts {
        part_name: leg | body_part {
            position: (x, y, z)
            size: (w, h, d)
            stiffness: <number>
            mass: <number>
        }
    }
    
    actuator {
        gait: quadruped | worm
        frequency: <number>
        forces: {
            lift: <number>
            push: <number>
            swing: <number>
        }
    }
}
```

## Controls

- `Ctrl+Enter`: Run simulation
- `Ctrl+N`: New file
- `Ctrl+O`: Open file
- `Ctrl+S`: Save file
