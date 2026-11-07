"""Concerto GUI - System & Native C++ Core Diagnostics Tab.

Provides hardware accelerator detection (Intel Arc / DirectML / OpenVINO / CUDA),
C++20 DSP native core verification, live CMake compilation streaming,
and audio output device enumeration.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets.log_viewer import StyledLogViewer
from ..workers import NativeBuildWorker


class DiagnosticsTab(QWidget):
    """System & Native C++ Core Diagnostics Studio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_worker: NativeBuildWorker | None = None

        self._setup_ui()
        self.refresh_all_diagnostics()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(14)

        # 1. Native C++20 DSP Core Status Card
        native_box = QGroupBox("Native C++20 DSP Core Accelerator")
        native_layout = QVBoxLayout(native_box)
        native_layout.setContentsMargins(14, 16, 14, 14)
        native_layout.setSpacing(10)

        # Top row: Status Badge & Benchmark Info
        top_row = QHBoxLayout()
        self.badge_native_status = QLabel("DETECTING CORE...")
        self.badge_native_status.setFixedHeight(28)
        self.badge_native_status.setStyleSheet(
            """
            background-color: #3e4451;
            color: #abb2bf;
            font-weight: 800;
            font-size: 11px;
            border-radius: 4px;
            padding: 4px 12px;
            """
        )
        top_row.addWidget(self.badge_native_status)

        top_row.addStretch()

        self.btn_benchmark = QPushButton("⚡ Benchmark DSP Speed")
        self.btn_benchmark.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #61afef;
                border: 1px solid #61afef;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #212b38;
            }
            """
        )
        self.btn_benchmark.clicked.connect(self._run_dsp_benchmark)
        top_row.addWidget(self.btn_benchmark)

        self.btn_compile = QPushButton("🔨 Compile Native Core")
        self.btn_compile.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #98c379;
                border: 1px solid #98c379;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #283c2e;
            }
            QPushButton:disabled {
                background-color: #3e4451;
                color: #5c6370;
            }
            """
        )
        self.btn_compile.clicked.connect(self._start_native_build)
        top_row.addWidget(self.btn_compile)
        native_layout.addLayout(top_row)

        # Details Grid
        details_grid = QGridLayout()
        details_grid.setHorizontalSpacing(14)
        details_grid.setVerticalSpacing(6)

        details_grid.addWidget(QLabel("Library Location:"), 0, 0)
        self.lbl_lib_path = QLabel("--")
        self.lbl_lib_path.setStyleSheet("color: #abb2bf; font-family: monospace;")
        details_grid.addWidget(self.lbl_lib_path, 0, 1)

        details_grid.addWidget(QLabel("Architecture / ABI:"), 1, 0)
        self.lbl_abi = QLabel(f"{platform.machine()} | C++20 Standard | SIMD Vectorized")
        self.lbl_abi.setStyleSheet("color: #abb2bf;")
        details_grid.addWidget(self.lbl_abi, 1, 1)

        details_grid.addWidget(QLabel("Measured Throughput:"), 2, 0)
        self.lbl_speed = QLabel("Not Benchmarked (Click 'Benchmark DSP Speed')")
        self.lbl_speed.setStyleSheet("color: #e5c07b; font-weight: bold;")
        details_grid.addWidget(self.lbl_speed, 2, 1)

        native_layout.addLayout(details_grid)

        # Embedded Live Compilation Log Viewer
        self.log_viewer = StyledLogViewer()
        self.log_viewer.setFixedHeight(160)
        native_layout.addWidget(self.log_viewer)

        layout.addWidget(native_box)


        # 3. Audio Devices & Hardware Channel Enumeration Card
        audio_box = QGroupBox("Audio Output Devices & Multichannel Capability")
        audio_layout = QVBoxLayout(audio_box)
        audio_layout.setContentsMargins(14, 16, 14, 14)
        audio_layout.setSpacing(10)

        dev_header = QHBoxLayout()
        self.lbl_default_dev = QLabel("Default Output Device: Detecting...")
        self.lbl_default_dev.setStyleSheet("font-weight: bold; color: #abb2bf;")
        dev_header.addWidget(self.lbl_default_dev)

        dev_header.addStretch()

        btn_refresh_audio = QPushButton("Refresh Devices")
        btn_refresh_audio.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #abb2bf;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #3e4451;
            }
            """
        )
        btn_refresh_audio.clicked.connect(self._refresh_audio_devices)
        dev_header.addWidget(btn_refresh_audio)
        audio_layout.addLayout(dev_header)

        # Devices Table
        self.table_audio = QTableWidget()
        self.table_audio.setColumnCount(5)
        self.table_audio.setHorizontalHeaderLabels([
            "ID", "Device Name", "Host API", "Max Output Ch", "Default Sample Rate"
        ])
        self.table_audio.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_audio.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_audio.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_audio.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_audio.verticalHeader().setVisible(False)
        self.table_audio.setFixedHeight(180)
        self.table_audio.setStyleSheet(
            """
            QTableWidget {
                background-color: #181a1f;
                border: 1px solid #2d3139;
                color: #abb2bf;
            }
            QHeaderView::section {
                background-color: #21252b;
                color: #abb2bf;
                font-weight: bold;
                border: 1px solid #2d3139;
                padding: 4px;
            }
            """
        )
        audio_layout.addWidget(self.table_audio)

        layout.addWidget(audio_box)
        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # Diagnostics Operations
    # -------------------------------------------------------------------------

    def refresh_all_diagnostics(self) -> None:
        """Run all diagnostic checks."""
        self._check_native_core()
        self._refresh_audio_devices()

    def _check_native_core(self) -> None:
        """Inspect C++20 DLL presence and availability."""
        from concerto.native import _find_library, library_available

        lib_p = _find_library()
        if lib_p and library_available():
            self.lbl_lib_path.setText(str(lib_p))
            self.badge_native_status.setText("ACTIVE (Native C++20 Core)")
            self.badge_native_status.setStyleSheet(
                """
                background-color: #283c2e;
                color: #98c379;
                font-weight: 800;
                font-size: 11px;
                border: 1px solid #98c379;
                border-radius: 4px;
                padding: 4px 12px;
                """
            )
        else:
            self.lbl_lib_path.setText(
                "Not Found (native/build/bin/libconcerto_core.dll). Using Python reference."
            )
            self.badge_native_status.setText("FALLBACK (Python Reference Core)")
            self.badge_native_status.setStyleSheet(
                """
                background-color: #3e3226;
                color: #e5c07b;
                font-weight: 800;
                font-size: 11px;
                border: 1px solid #e5c07b;
                border-radius: 4px;
                padding: 4px 12px;
                """
            )

    def _run_dsp_benchmark(self) -> None:
        """Execute a quick DSP throughput benchmark on synthetic stereo audio."""
        from concerto.dsp.upmix import load_config
        from concerto.native import library_available, make_renderer

        cfg_p = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "default.toml"
        cfg = load_config(cfg_p if cfg_p.exists() else "configs/default.toml")

        sr = 48000
        n_samples = 48000  # 1.0 second of audio
        left = (0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(n_samples) / sr)).astype(np.float32)
        right = (0.5 * np.cos(2.0 * np.pi * 440.0 * np.arange(n_samples) / sr)).astype(np.float32)

        try:
            renderer = make_renderer(sr, cfg)
            # Warm up
            _ = renderer.render_block(left[:1024], right[:1024])

            t0 = time.perf_counter()
            _ = renderer.render_block(left, right)
            elapsed = time.perf_counter() - t0

            speed_rt = 1.0 / max(elapsed, 1e-6)
            is_native = library_available()
            core_type = "Native C++20" if is_native else "Python Reference"
            self.lbl_speed.setText(f"{speed_rt:.1f}x Realtime ({elapsed * 1000.0:.2f} ms / sec) [{core_type}]")
            self.lbl_speed.setStyleSheet("color: #98c379; font-weight: bold;")
        except Exception as e:
            self.lbl_speed.setText(f"Benchmark Error: {e}")
            self.lbl_speed.setStyleSheet("color: #e06c75; font-weight: bold;")

    def _start_native_build(self) -> None:
        """Launch background CMake build worker."""
        self.btn_compile.setEnabled(False)
        self.log_viewer.clear()
        self.log_viewer.append_log("Starting C++20 native core compilation via CMake...", "info")

        self._build_worker = NativeBuildWorker(parent=self)
        self._build_worker.log_signal.connect(lambda line: self.log_viewer.append_log(line, "info"))
        self._build_worker.finished_signal.connect(self._on_build_finished)
        self._build_worker.start()

    def _on_build_finished(self, success: bool, message: str) -> None:
        self.btn_compile.setEnabled(True)
        if success:
            self.log_viewer.append_log(f"Build succeeded: {message}", "success")
            QMessageBox.information(self, "Native Core Built", "Native core compiled successfully!")
        else:
            self.log_viewer.append_log(f"Build failed: {message}", "error")
            QMessageBox.warning(self, "Build Failed", f"Native compilation failed:\n{message}")
        self._check_native_core()

    def _refresh_audio_devices(self) -> None:
        """Enumerate host sound devices via sounddevice."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            default_out = sd.default.device[1]
        except Exception as e:
            self.lbl_default_dev.setText(f"Error querying audio devices: {e}")
            return

        # Find default device name
        def_name = "Unknown"
        if 0 <= default_out < len(devices):
            def_name = f"[{default_out}] {devices[default_out].get('name', 'Unknown')}"
        self.lbl_default_dev.setText(f"Default Output Device: {def_name}")

        # Filter output devices
        out_devs = [
            (idx, d) for idx, d in enumerate(devices) if int(d.get("max_output_channels", 0)) > 0
        ]
        self.table_audio.setRowCount(len(out_devs))

        for row, (idx, dev) in enumerate(out_devs):
            is_def = (idx == default_out)
            item_idx = QTableWidgetItem(f"{idx}{'  ★' if is_def else ''}")
            item_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_def:
                item_idx.setForeground(QColor("#61afef"))
                item_idx.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.table_audio.setItem(row, 0, item_idx)

            name = dev.get("name", "")
            item_name = QTableWidgetItem(name)
            if is_def:
                item_name.setForeground(QColor("#61afef"))
            self.table_audio.setItem(row, 1, item_name)

            api_idx = dev.get("hostapi", 0)
            api_name = hostapis[api_idx].get("name", f"API {api_idx}") if api_idx < len(hostapis) else str(api_idx)
            self.table_audio.setItem(row, 2, QTableWidgetItem(api_name))

            max_ch = int(dev.get("max_output_channels", 0))
            ch_desc = f"{max_ch} ch"
            if max_ch >= 8:
                ch_desc += " (7.1 Surround)"
            elif max_ch >= 6:
                ch_desc += " (5.1 Surround)"
            elif max_ch == 2:
                ch_desc += " (Stereo)"
            item_ch = QTableWidgetItem(ch_desc)
            if max_ch >= 6:
                item_ch.setForeground(QColor("#98c379"))
            self.table_audio.setItem(row, 3, item_ch)

            sr = int(dev.get("default_samplerate", 48000))
            item_sr = QTableWidgetItem(f"{sr} Hz")
            item_sr.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_audio.setItem(row, 4, item_sr)
