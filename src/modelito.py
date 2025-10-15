# MODELITO by the MODELO TEAM

import os
import sys

# if not linux remove this
if sys.platform.startswith('linux'):
    os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QSplitter,
                             QLabel, QStatusBar, QToolBar, QFileDialog, QMessageBox,
                             QProgressBar, QMenu, QToolButton)
from PyQt6.QtCore import Qt, QTimer, QTime
from PyQt6.QtGui import (QFont, QTextCharFormat, QColor, QSyntaxHighlighter, 
                         QAction, QKeySequence, QPixmap)

from ide import ModelitoGUI
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
