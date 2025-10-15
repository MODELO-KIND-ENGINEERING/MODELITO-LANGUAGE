<div align="center">

<img src="src/MODELITO_LOGO.svg" alt="Modelito Logo" width="200"/>

# Modelito Soft Robotics Simulator

*Bringing soft robotics to life through intuitive simulation*

A friendly and powerful Python-based soft robotics simulator that makes designing and testing deformable robots a delightful experience! Whether you're a robotics enthusiast, researcher, or just curious about soft robots, Modelito is here to help you bring your ideas to life.

</div>

## 👋 Welcome to Modelito!

Modelito is crafted with love by the MODELO KIND ENGINEERING TEAM:

- Daniel Motilla (M0TH)
- Frederick Ayala
- Davide Vigano
- Oyvind Soroy (Colonthree)
- Drake (Yoshi)

We hope you have fun with our software and create something wonderful :3

## ⚡ Quick Start

1. **Set up your environment:**
```bash
conda create -n MODELITO python=3.13
conda activate MODELITO
pip install numpy scipy matplotlib vedo PyQt6 lark
```

2. **Launch Modelito:**
```bash
python modelito.py
```

3. **Start Creating!** Create a robot definition (e.g., `my_robot.robot`) and load it in the editor.

## 🤖 Example Robots

We've included some fun example robots to help you get started:
- `quadruped_robot.robot`: A lively four-legged walking robot that can traverse various terrains
- `worm_robot.robot`: An elegant segmented robot with smooth undulating motion

Feel free to modify these examples or use them as inspiration for your own creations!

## ✨ Key Features

- 🎨 **Intuitive Visual Interface** with syntax highlighting for easy robot design
- ⚡ **Real-time Physics Simulation** for immediate feedback
- 🦿 **Multiple Gait Patterns** including quadruped and worm-like motion
- 🛠️ **Highly Customizable** robot parameters for endless possibilities
- 🌟 **Beautiful 3D Visualization** powered by vedo

## 📝 Robot Definition Language

Modelito uses an intuitive, easy-to-learn language for defining robots. Here's a basic structure to get you started:

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

## ⌨️ Controls

We've kept the controls simple and familiar:

- `Ctrl+Enter`: Bring your robot to life (Run simulation)
- `Ctrl+N`: Start a new creation
- `Ctrl+O`: Open an existing robot
- `Ctrl+S`: Save your work

## 🤝 Contributing & Support

We love seeing our community grow! Here's how you can get involved:

- 🐛 Found a bug? Open an issue
- 💡 Have an idea? Share your feature requests
- 🔧 Want to contribute code? Submit a pull request
- 📚 Need help? Join our community discussions
- ☕ Love the project? [Buy us a coffee on Ko-fi](https://ko-fi.com/mothxyz)

Let's create amazing soft robots together! 🚀
