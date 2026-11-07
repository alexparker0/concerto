"""Concerto desktop application entrypoint and initialization."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

try:
    from concerto.gui.main_window import MainWindow
    from concerto.gui.theme import apply_dark_theme
except ImportError:
    from .main_window import MainWindow
    from .theme import apply_dark_theme


def run_gui(initial_file: str | None = None) -> int:
    """Launch the Concerto GUI application."""
    # Ensure High DPI support
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("Concerto")
    app.setApplicationDisplayName("Concerto — Stereo-to-Surround DSP Workstation")
    app.setOrganizationName("Concerto")
    app.setOrganizationDomain("concerto.audio")

    # Apply modern DAW dark theme stylesheet
    apply_dark_theme(app)

    window = MainWindow(initial_file=initial_file)
    window.show()

    return app.exec()


if __name__ == "__main__":
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_gui(initial_path))
