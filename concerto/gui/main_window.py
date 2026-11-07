"""Main Window for the Concerto Stereo-to-Surround DSP Workstation."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..native import library_available
from .tabs.ai_tab import AiTab
from .tabs.analysis_tab import AnalysisTab
from .tabs.benchmark_tab import BenchmarkTab
from .tabs.config_tab import ConfigTab
from .tabs.diagnostics_tab import DiagnosticsTab
from .tabs.render_tab import RenderTab
from .tabs.routing_tab import RoutingTab
from .tabs.validation_tab import ValidationTab
from .widgets.audio_player import AudioTransportBar


class MainWindow(QMainWindow):
    """Concerto desktop GUI main window."""

    def __init__(self, initial_file: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Concerto — Stereo-to-Surround DSP Workstation")
        self.resize(1300, 880)
        self.setMinimumSize(1040, 720)

        self._build_ui()
        self._setup_connections()

        if initial_file:
            self.load_input_file(initial_file)

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # Header bar
        header = QFrame(self)
        header.setObjectName("appHeader")
        header.setStyleSheet("""
            QFrame#appHeader {
                background-color: #1e2227;
                border: 1px solid #282c34;
                border-radius: 8px;
                padding: 6px 12px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        title_label = QLabel("CONCERTO", header)
        title_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #61afef; letter-spacing: 2px;")

        subtitle_label = QLabel("STEREO → 5.1/7.1 SURROUND ENGINE", header)
        subtitle_font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet("color: #5c6370; margin-left: 8px; letter-spacing: 1px;")

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addStretch()

        # Engine badges
        is_native = library_available()
        self.native_badge = QLabel(
            "⚡ Native C++20 Core" if is_native else "🐢 Python DSP Fallback",
            header,
        )
        self.native_badge.setStyleSheet(
            f"background-color: {'#1b3222' if is_native else '#3a2e1d'};"
            f"color: {'#98c379' if is_native else '#e5c07b'};"
            f"border: 1px solid {'#2b5735' if is_native else '#634e2c'};"
            "border-radius: 12px; padding: 4px 12px; font-weight: bold; font-size: 11px;"
        )
        header_layout.addWidget(self.native_badge)

        try:
            import torch
            torch_device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
            ai_text = f"🧠 Demucs ({torch_device})"
        except ImportError:
            ai_text = "🧠 Demucs AI"

        self.ai_badge = QLabel(ai_text, header)
        self.ai_badge.setStyleSheet(
            "background-color: #1a2736; color: #61afef; border: 1px solid #284b63;"
            "border-radius: 12px; padding: 4px 12px; font-weight: bold; font-size: 11px;"
        )
        header_layout.addWidget(self.ai_badge)

        main_layout.addWidget(header)

        # Tab widget hosting all 8 functional areas
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)

        self.render_tab = RenderTab(self)
        self.analysis_tab = AnalysisTab(self)
        self.validation_tab = ValidationTab(self)
        self.benchmark_tab = BenchmarkTab(self)
        self.ai_tab = AiTab(self)
        self.routing_tab = RoutingTab(self)
        self.config_tab = ConfigTab(self)
        self.diagnostics_tab = DiagnosticsTab(self)

        self.tabs.addTab(self.render_tab, "🎛 Render Studio")
        self.tabs.addTab(self.analysis_tab, "📊 Stereo Analysis")
        self.tabs.addTab(self.validation_tab, "✅ Validation & QA")
        self.tabs.addTab(self.benchmark_tab, "⚡ Benchmarks")
        self.tabs.addTab(self.ai_tab, "🧠 AI Stem Lab")
        self.tabs.addTab(self.routing_tab, "🔊 Speaker Routing")
        self.tabs.addTab(self.config_tab, "⚙️ Preset Studio")
        self.tabs.addTab(self.diagnostics_tab, "🩺 Diagnostics")

        main_layout.addWidget(self.tabs, stretch=1)

        # Bottom persistent audio transport bar
        self.transport_bar = AudioTransportBar(parent=self)
        main_layout.addWidget(self.transport_bar)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select or drag & drop a stereo file to begin.")

    def _setup_connections(self) -> None:
        # When render finishes, load output into transport bar
        self.render_tab.audio_rendered.connect(self._on_audio_rendered)

        # When routing test finishes, load into transport bar
        if hasattr(self.routing_tab, "test_file_generated"):
            self.routing_tab.test_file_generated.connect(self._on_audio_rendered)

        # When config is changed in Preset Studio, update Render tab's config
        if hasattr(self.config_tab, "config_saved"):
            self.config_tab.config_saved.connect(self._on_config_updated)

    def _on_audio_rendered(self, file_path: str) -> None:
        self.transport_bar.load_file(file_path)
        self.status_bar.showMessage(f"Loaded rendered audio into player: {file_path}", 6000)

    def _on_config_updated(self, config_dict: dict) -> None:
        self.status_bar.showMessage("Preset configuration updated.", 4000)

    def load_input_file(self, file_path: str) -> None:
        """Load an initial file into the active workspace."""
        if hasattr(self.render_tab, "set_input_file"):
            self.render_tab.set_input_file(file_path)
        elif hasattr(self.render_tab, "input_edit"):
            self.render_tab.input_edit.setText(file_path)
            if hasattr(self.render_tab, "_on_input_changed"):
                self.render_tab._on_input_changed()

    def closeEvent(self, event) -> None:
        # Cleanly stop audio playback on application close
        try:
            self.transport_bar.player.stop()
            self.transport_bar.player.close()
        except Exception:
            pass
        event.accept()
