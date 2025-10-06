# By Daniel Motilla (M0TH)
# Modelito Language - Prototype GUI with PyQt6
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QSplitter,
                             QLabel, QStatusBar, QToolBar, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QAction, QKeySequence
from lark import Lark, Transformer, v_args, Token
from build123d import *
from dataclasses import dataclass
from typing import Any, Callable, List, Dict
import math

# ============ Global variable to store results (for ocp_vscode) ============
result_part = None

# ============ Modelito Language Core (from your code) ============

grammar = """
start: expr

// Expressions (Algebraic Data Types)
expr: literal
    | binary_op
    | unary_op  
    | lambda
    | application
    | let_binding
    | if_expr
    | variable
    | "(" expr ")"

// Literals
literal: NUMBER -> number
       | shape

shape: "box" "(" expr "," expr "," expr ")"
     | "cube" "(" expr ")" -> cube_single

// Binary operations
binary_op: expr "+" expr -> add
         | expr "-" expr -> sub
         | expr "*" expr -> mul
         | expr "|" expr -> beside
         | expr "/" expr -> above
         | expr ">>>" expr -> compose

// Unary operations  
unary_op: "rotate" "(" expr "," expr ")" -> rotate_op
        | "scale" "(" expr "," expr ")" -> scale_op
        | "mirror" "(" expr ")" -> mirror_op
        | "repeat" "(" expr "," expr ")" -> repeat_op

// Lambda abstraction
lambda: "fn" "(" params ")" "=>" expr

// Function application
application: variable "(" args ")"

// Let bindings
let_binding: "let" NAME "=" expr "in" expr

// Conditionals
if_expr: "if" expr "then" expr "else" expr

// Variable reference
variable: NAME

params: NAME ("," NAME)*
args: expr ("," expr)*

NAME: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.NUMBER
%import common.WS
%ignore WS
"""

@dataclass
class Part3D:
    """Represents a 3D part"""
    part: Any
    label: str = "part"
    
    def __repr__(self):
        return f"Part3D({self.label})"

@dataclass  
class Lambda:
    """Represents a lambda function"""
    params: List[str]
    body: Any
    env: Dict[str, Any]
    
    def __repr__(self):
        return f"Lambda({self.params})"

@dataclass
class Composition:
    """Represents function composition"""
    f: Callable
    g: Callable
    
    def __call__(self, x):
        return self.f(self.g(x))

class Interpreter:
    def __init__(self):
        self.env = {}
        self.result_parts = []
        self.result_part = None
        
    def eval(self, node, env=None):
        """Evaluate an AST node"""
        if env is None:
            env = self.env
            
        if isinstance(node, Token):
            node_type = node.type
            node_value = node.value
        elif hasattr(node, 'data'):
            node_type = node.data
            node_value = None
        else:
            return node
            
        if node_type == 'NUMBER':
            return float(node_value)
            
        if node_type == 'NAME':
            if node_value in env:
                return env[node_value]
            raise NameError(f"Variable '{node_value}' not found")
            
        if node_type == 'box':
            w = self.eval(node.children[0], env)
            h = self.eval(node.children[1], env)
            d = self.eval(node.children[2], env)
            with BuildPart() as bp:
                Box(w, h, d)
            part = Part3D(bp.part, f"box_{w}x{h}x{d}")
            self.result_part = bp.part
            self.result_part.label = part.label
            self.result_parts.append(part.part)
            return part
            
        if node_type == 'cube_single':
            s = self.eval(node.children[0], env)
            with BuildPart() as bp:
                Box(s, s, s)
            part = Part3D(bp.part, f"cube_{s}")
            self.result_part = bp.part
            self.result_part.label = part.label
            self.result_parts.append(part.part)
            return part
            
        if node_type == 'add':
            left = self.eval(node.children[0], env)
            right = self.eval(node.children[1], env)
            
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right
                
            with BuildPart() as bp:
                if isinstance(left, Part3D):
                    with Locations((0, 0, 0)):
                        add(left.part)
                else:
                    for i in range(int(left)):
                        size = 1 + i * 0.3
                        with Locations((i * size, 0, 0)):
                            Box(size, size, size)
                            
                offset = sum(1 + i * 0.3 for i in range(int(left))) if isinstance(left, (int, float)) else left.part.bounding_box().max.X - left.part.bounding_box().min.X
                if isinstance(right, Part3D):
                    with Locations((offset, 0, 0)):
                        add(right.part)
                else:
                    for i in range(int(right)):
                        size = 1 + i * 0.3
                        with Locations((offset + sum(1 + j * 0.3 for j in range(i)), 0, 0)):
                            Box(size, size, size)
                            
            part = Part3D(bp.part, f"add")
            self.result_part = bp.part
            self.result_part.label = part.label
            self.result_parts.append(part.part)
            return part
            
        if node_type == 'mul':
            left = self.eval(node.children[0], env)
            right = self.eval(node.children[1], env)
            
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left * right
                
            rows = int(left) if isinstance(left, (int, float)) else 3
            cols = int(right) if isinstance(right, (int, float)) else 3
            
            with BuildPart() as bp:
                for i in range(rows):
                    for j in range(cols):
                        with Locations((j * 1.0, i * 1.0, 0)):
                            Box(1, 1, 1)
                            
            part = Part3D(bp.part, f"grid_{rows}x{cols}")
            self.result_part = bp.part
            self.result_part.label = part.label
            self.result_parts.append(part.part)
            return part
            
        if node_type == 'beside':
            left = self.eval(node.children[0], env)
            right = self.eval(node.children[1], env)
            
            with BuildPart() as bp:
                add(left.part)
                bbox = left.part.bounding_box()
                offset = bbox.max.X - bbox.min.X
                with Locations((offset, 0, 0)):
                    add(right.part)
                    
            part = Part3D(bp.part, f"beside")
            self.result_part = bp.part
            self.result_part.label = part.label
            self.result_parts.append(part.part)
            return part
            
        if node_type == 'above':
            left = self.eval(node.children[0], env)
            right = self.eval(node.children[1], env)
            
            with BuildPart() as bp:
                add(left.part)
                bbox = left.part.bounding_box()
                offset = bbox.max.Z - bbox.min.Z
                with Locations((0, 0, offset)):
                    add(right.part)
                    
            part = Part3D(bp.part, f"above")
            self.result_part = bp.part
            self.result_part.label = part.label
            self.result_parts.append(part.part)
            return part
            
        if node_type == 'compose':
            f = self.eval(node.children[0], env)
            g = self.eval(node.children[1], env)
            return Composition(f, g)
            
        if node_type == 'rotate_op':
            part = self.eval(node.children[0], env)
            angle = self.eval(node.children[1], env)
            
            with BuildPart() as bp:
                add(part.part)
                bp.part = Rot(0, 0, angle) * bp.part
                
            result = Part3D(bp.part, f"rotated_{angle}")
            self.result_part = bp.part
            self.result_part.label = result.label
            self.result_parts.append(result.part)
            return result
            
        if node_type == 'scale_op':
            part = self.eval(node.children[0], env)
            factor = self.eval(node.children[1], env)
            
            with BuildPart() as bp:
                add(part.part)
                bp.part = bp.part.scale(factor)
                
            result = Part3D(bp.part, f"scaled_{factor}")
            self.result_part = bp.part
            self.result_part.label = result.label
            self.result_parts.append(result.part)
            return result
            
        if node_type == 'repeat_op':
            part_expr = node.children[0]
            count = int(self.eval(node.children[1], env))
            
            with BuildPart() as bp:
                for i in range(count):
                    part = self.eval(part_expr, env)
                    bbox = part.part.bounding_box()
                    width = bbox.max.X - bbox.min.X
                    with Locations((i * width, 0, 0)):
                        add(part.part)
                        
            result = Part3D(bp.part, f"repeat_{count}")
            self.result_part = bp.part
            self.result_part.label = result.label
            self.result_parts.append(result.part)
            return result
            
        if node_type == 'lambda':
            params = [p.value for p in node.children[0].children]
            body = node.children[1]
            return Lambda(params, body, env.copy())
            
        if node_type == 'application':
            func_name = node.children[0].children[0].value
            func = env.get(func_name)
            
            if func is None:
                raise NameError(f"Function '{func_name}' not found")
                
            args = [self.eval(arg, env) for arg in node.children[1].children]
            
            if isinstance(func, Lambda):
                new_env = func.env.copy()
                for param, arg in zip(func.params, args):
                    new_env[param] = arg
                return self.eval(func.body, new_env)
            elif isinstance(func, Composition):
                return func(args[0])
            else:
                raise TypeError(f"'{func_name}' is not callable")
                
        if node_type == 'let_binding':
            var_name = node.children[0].value
            var_value = self.eval(node.children[1], env)
            body = node.children[2]
            
            new_env = env.copy()
            new_env[var_name] = var_value
            
            if isinstance(var_value, Lambda):
                var_value.env[var_name] = var_value
            
            return self.eval(body, new_env)
            
        if node_type == 'if_expr':
            condition = self.eval(node.children[0], env)
            then_expr = node.children[1]
            else_expr = node.children[2]
            
            if condition:
                return self.eval(then_expr, env)
            else:
                return self.eval(else_expr, env)
                
        if hasattr(node, 'children'):
            return self.eval(node.children[0], env)
            
        return node

parser = Lark(grammar, parser='lalr')

# ============ Syntax Highlighter ============

class ModelitoHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Define formats (creamy theme colors)
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#8B4513"))  # Saddle brown
        self.keyword_format.setFontWeight(QFont.Weight.Bold)
        
        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor("#D2691E"))  # Chocolate
        
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#CD853F"))  # Peru
        
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#B8860B"))  # Dark goldenrod
        
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#A0826D"))  # Beige brown
        
        # Keywords
        self.keywords = [
            'let', 'in', 'fn', 'if', 'then', 'else',
            'cube', 'box', 'rotate', 'scale', 'repeat', 'mirror'
        ]
        
    def highlightBlock(self, text):
        # Highlight keywords
        for keyword in self.keywords:
            index = text.find(keyword)
            while index >= 0:
                length = len(keyword)
                self.setFormat(index, length, self.keyword_format)
                index = text.find(keyword, index + length)
        
        # Highlight numbers
        import re
        for match in re.finditer(r'\b\d+\.?\d*\b', text):
            self.setFormat(match.start(), match.end() - match.start(), self.number_format)

# ============ Main GUI ============

class ModelitoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modelito Language Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set creamy background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFF8DC;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #FFF8DC;")
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create splitter for code and output
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Code editor
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        code_label = QLabel("📝 Code Editor")
        code_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; color: #8B4513; background-color: #FFF8DC;")
        left_layout.addWidget(code_label)
        
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Consolas", 12))
        self.code_editor.setPlaceholderText("Write your Modelito code here...\n\nExamples:\ncube(2)\ncube(1) | cube(2)\nlet x = cube(1) in repeat(x, 5)")
        self.code_editor.setStyleSheet("""
            QTextEdit {
                background-color: #FFFAF0;
                color: #5C4033;
                border: 2px solid #D2B48C;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        
        # Add syntax highlighter
        self.highlighter = ModelitoHighlighter(self.code_editor.document())
        
        left_layout.addWidget(self.code_editor)
        
        # Run button
        self.run_button = QPushButton("▶ Run Code (Ctrl+Enter)")
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
                border: 2px solid #A0522D;
            }
            QPushButton:pressed {
                background-color: #8B4513;
            }
        """)
        self.run_button.clicked.connect(self.run_code)
        left_layout.addWidget(self.run_button)
        
        splitter.addWidget(left_widget)
        
        # Right side: Output
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
        
        # Set splitter sizes
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set default code
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
                border: 1px solid #CD853F;
            }
        """)
        self.addToolBar(toolbar)
        
        # New file
        new_action = QAction("📄 New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_file)
        toolbar.addAction(new_action)
        
        # Open file
        open_action = QAction("📂 Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        # Save file
        save_action = QAction("💾 Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Examples menu
        examples_action = QAction("📚 Examples", self)
        examples_action.triggered.connect(self.show_examples)
        toolbar.addAction(examples_action)
        
        # Help
        help_action = QAction("❓ Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
        
    def set_example_code(self):
        example = """// Welcome to Modelito!
// Press Ctrl+Enter or click Run to execute

cube(2)"""
        self.code_editor.setText(example)
        
    def new_file(self):
        self.code_editor.clear()
        self.output_area.clear()
        self.status_bar.showMessage("New file created")
        
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Modelito Files (*.mod);;All Files (*)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.code_editor.setText(f.read())
                self.status_bar.showMessage(f"Opened: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
                
    def save_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Modelito Files (*.mod);;All Files (*)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.code_editor.toPlainText())
                self.status_bar.showMessage(f"Saved: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")
                
    def show_examples(self):
        examples = """
=== MODELITO EXAMPLES ===

1. Simple Cube:
cube(2)

2. Side by Side:
cube(1) | cube(2) | cube(3)

3. Stacking:
cube(3) / cube(2) / cube(1)

4. Grid:
3 * 4

5. Repeat:
repeat(cube(1), 5)

6. Variables:
let x = cube(2) in x | x

7. Rotation:
rotate(cube(2), 45)

8. Recursive Tower:
let tower = fn(n) => if n then cube(1) / tower(0) else cube(1) in tower(5)

9. Complex Pattern:
let x = cube(1) in repeat(x, 3) / repeat(x, 5) / repeat(x, 3)
"""
        QMessageBox.information(self, "Examples", examples)
        
    def show_help(self):
        help_text = """
=== MODELITO LANGUAGE HELP ===

SHAPES:
  cube(size)         - Create a cube
  box(w, h, d)       - Create a box

OPERATORS:
  a | b              - Place beside
  a / b              - Stack above
  a + b              - Addition
  a * b              - Grid

TRANSFORMATIONS:
  rotate(shape, angle)
  scale(shape, factor)
  repeat(shape, count)

FUNCTIONS:
  let x = value in expr
  fn(param) => expr
  if condition then a else b

SHORTCUTS:
  Ctrl+Enter         - Run code
  Ctrl+N             - New file
  Ctrl+O             - Open file
  Ctrl+S             - Save file
"""
        QMessageBox.information(self, "Help", help_text)
        
    def run_code(self):
        code = self.code_editor.toPlainText().strip()
        
        if not code:
            self.output_area.setText("⚠️ No code to run!")
            return
            
        self.output_area.clear()
        self.output_area.append("🚀 Running code...\n")
        self.status_bar.showMessage("Executing...")
        
        try:
            # Parse
            tree = parser.parse(code)
            self.output_area.append("✅ Parsing successful!\n")
            self.output_area.append("📋 AST:\n" + tree.pretty() + "\n")
            
            # Evaluate
            interpreter = Interpreter()
            result = interpreter.eval(tree)
            
            self.output_area.append(f"✅ Evaluation successful!\n")
            self.output_area.append(f"📦 Result: {result}\n")
            
            if interpreter.result_part:
                self.output_area.append(f"📐 Volume: {interpreter.result_part.volume:.2f}\n")
                bbox = interpreter.result_part.bounding_box()
                self.output_area.append(f"📏 Bounding Box: {bbox}\n")
                self.output_area.append(f"✨ Parts created: {len(interpreter.result_parts)}\n")
                
                # IMPORTANT: Show in 3D viewer (like original code)
                try:
                    from ocp_vscode import show, show_all
                    
                    # Set global result_part for show_all() to find
                    global result_part
                    result_part = interpreter.result_part
                    result_part.label = interpreter.result_part.label
                    
                    # Show all parts
                    show_all()
                    
                    self.output_area.append("\n✨ 3D Viewer updated! Check your OCP viewer.\n")
                    
                    # Also try direct show as fallback
                    try:
                        show(result_part)
                    except:
                        pass
                        
                except ImportError:
                    self.output_area.append("\n⚠️ ocp_vscode not available. 3D viewing disabled.\n")
                except Exception as e:
                    self.output_area.append(f"\n⚠️ Could not update 3D viewer: {str(e)}\n")
                
                self.output_area.append("\n💡 Tip: The geometry is now visible in your OCP viewer!")
                
            self.status_bar.showMessage("✅ Execution completed successfully!")
            
        except Exception as e:
            import traceback
            self.output_area.append(f"❌ Error: {str(e)}\n")
            self.output_area.append(f"\n📋 Traceback:\n{traceback.format_exc()}")
            self.status_bar.showMessage("❌ Execution failed!")
            
    def keyPressEvent(self, event):
        # Ctrl+Enter to run
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.run_code()
        else:
            super().keyPressEvent(event)

# ============ Main ============

def main():
    app = QApplication(sys.argv)
    
    # Set light/creamy theme
    app.setStyle("Fusion")
    
    # Custom palette for creamy colors
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