"""Dark theme and high-contrast DAW workstation styling for Concerto GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication, QWidget

# DAW workstation aesthetic: deep charcoal/slate surfaces, crisp borders,
# clear hierarchy, vibrant cyan/blue accent with subtle success/warning/error colors.
DARK_STYLE_SHEET: str = """
/* Global Reset & Base */
QWidget {
    background-color: #181a1f;
    color: #abb2bf;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    selection-background-color: #61afef;
    selection-color: #181a1f;
}

QMainWindow, QDialog {
    background-color: #181a1f;
}

/* Headers and Titles */
QLabel#headerTitle {
    color: #e5c07b;
    font-size: 16px;
    font-weight: 600;
}

QLabel#sectionTitle {
    color: #d7dae0;
    font-size: 14px;
    font-weight: 600;
    padding-bottom: 4px;
}

QLabel#mutedText {
    color: #5c6370;
    font-size: 11px;
}

/* Group Boxes (Cards/Panels) */
QGroupBox {
    background-color: #21252b;
    border: 1px solid #2d3139;
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px;
    font-weight: 600;
    color: #d7dae0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
    background-color: #21252b;
    color: #61afef;
}

/* Tab Widget & Bars */
QTabWidget::pane {
    border: 1px solid #2d3139;
    background-color: #181a1f;
    border-radius: 4px;
    top: -1px;
}

QTabBar::tab {
    background-color: #21252b;
    color: #7f848e;
    border: 1px solid #2d3139;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 90px;
    padding: 7px 14px;
    margin-right: 2px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #181a1f;
    color: #61afef;
    border-top: 2px solid #61afef;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background-color: #282c34;
    color: #abb2bf;
}

/* Buttons */
QPushButton {
    background-color: #282c34;
    color: #d7dae0;
    border: 1px solid #3e4451;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #353b45;
    border-color: #61afef;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #1e2227;
    border-color: #4b5263;
}

QPushButton:disabled {
    background-color: #1e2227;
    color: #5c6370;
    border-color: #282c34;
}

QPushButton#primaryButton {
    background-color: #61afef;
    color: #181a1f;
    font-weight: 600;
    border: 1px solid #61afef;
}

QPushButton#primaryButton:hover {
    background-color: #74b9f2;
    border-color: #74b9f2;
    color: #0f1115;
}

QPushButton#primaryButton:pressed {
    background-color: #4fa0df;
}

QPushButton#successButton {
    background-color: #98c379;
    color: #181a1f;
    font-weight: 600;
    border: 1px solid #98c379;
}

QPushButton#warningButton {
    background-color: #e5c07b;
    color: #181a1f;
    font-weight: 600;
    border: 1px solid #e5c07b;
}

QPushButton#dangerButton {
    background-color: #e06c75;
    color: #ffffff;
    font-weight: 600;
    border: 1px solid #e06c75;
}

QToolButton {
    background-color: #282c34;
    color: #abb2bf;
    border: 1px solid #3e4451;
    border-radius: 4px;
    padding: 4px 8px;
}

QToolButton:hover {
    background-color: #353b45;
    border-color: #61afef;
}

QToolButton:pressed {
    background-color: #1e2227;
}

/* Input Controls */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #1e2227;
    color: #d7dae0;
    border: 1px solid #3e4451;
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: #61afef;
    selection-color: #181a1f;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #61afef;
    background-color: #21252b;
}

QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
    background-color: #181a1f;
    color: #5c6370;
    border-color: #282c34;
}

QComboBox {
    background-color: #282c34;
    color: #d7dae0;
    border: 1px solid #3e4451;
    border-radius: 4px;
    padding: 5px 10px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #61afef;
}

QComboBox:focus {
    border: 1px solid #61afef;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #3e4451;
}

QComboBox QAbstractItemView {
    background-color: #21252b;
    border: 1px solid #3e4451;
    color: #abb2bf;
    selection-background-color: #61afef;
    selection-color: #181a1f;
    padding: 4px;
}

QSpinBox, QDoubleSpinBox {
    background-color: #1e2227;
    color: #d7dae0;
    border: 1px solid #3e4451;
    border-radius: 4px;
    padding: 4px 6px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #61afef;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #282c34;
    border: 1px solid #3e4451;
    width: 14px;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #2d3139;
    height: 6px;
    background: #1e2227;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #61afef;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #d7dae0;
    border: 1px solid #61afef;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
    box-shadow: 0 0 4px #61afef;
}

/* Progress Bars */
QProgressBar {
    border: 1px solid #2d3139;
    border-radius: 4px;
    background-color: #1e2227;
    text-align: center;
    color: #abb2bf;
    font-size: 11px;
    font-weight: 600;
}

QProgressBar::chunk {
    background-color: #61afef;
    border-radius: 3px;
}

/* Tables and Headers */
QTableWidget, QTableView {
    background-color: #1e2227;
    border: 1px solid #2d3139;
    gridline-color: #282c34;
    color: #abb2bf;
    selection-background-color: #2c313c;
    selection-color: #61afef;
}

QHeaderView::section {
    background-color: #21252b;
    color: #d7dae0;
    padding: 6px;
    border: 1px solid #2d3139;
    font-weight: 600;
    font-size: 12px;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #181a1f;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #3e4451;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #5c6370;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    border: none;
}

QScrollBar:horizontal {
    background: #181a1f;
    height: 10px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #3e4451;
    min-width: 20px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #5c6370;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    border: none;
}

/* Splitters */
QSplitter::handle {
    background-color: #282c34;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Checkboxes and Radios */
QCheckBox, QRadioButton {
    color: #abb2bf;
    spacing: 6px;
}

QCheckBox:hover, QRadioButton:hover {
    color: #ffffff;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #3e4451;
    background-color: #1e2227;
    border-radius: 3px;
}

QRadioButton::indicator {
    border-radius: 8px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #61afef;
    border-color: #61afef;
}

/* Status Bar */
QStatusBar {
    background-color: #1e2227;
    color: #abb2bf;
    border-top: 1px solid #2d3139;
}
"""


def apply_dark_theme(app_or_window: QApplication | QWidget) -> None:
    """Apply the high-contrast DAW workstation theme to a Qt Application or Window."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPalette

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#181a1f"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#abb2bf"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#1e2227"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#21252b"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#21252b"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#abb2bf"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#d7dae0"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#282c34"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#d7dae0"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#61afef"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#181a1f"))
        palette.setColor(QPalette.ColorRole.Link, QColor("#61afef"))

        if hasattr(app_or_window, "setPalette"):
            app_or_window.setPalette(palette)
    except Exception:
        # Fallback if Qt palettes cannot be initialized
        pass

    if hasattr(app_or_window, "setStyleSheet"):
        app_or_window.setStyleSheet(DARK_STYLE_SHEET)
