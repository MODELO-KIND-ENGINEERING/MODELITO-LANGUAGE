from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QSplitter,
                             QLabel, QStatusBar, QToolBar, QFileDialog, QMessageBox,
                             QProgressBar, QMenu, QToolButton)
from PyQt6.QtCore import Qt, QTimer, QTime
from PyQt6.QtGui import (QFont, QTextCharFormat, QColor, QSyntaxHighlighter, 
                         QAction, QKeySequence, QPixmap)

from language import *
from simulation import simulate_robot

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
