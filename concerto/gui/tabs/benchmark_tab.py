"""Multi-Profile Benchmark Studio tab for Concerto GUI.

Features:
- Source audio selection with drag-and-drop.
- Multi-profile selection: Baseline, Adaptive, Adaptive-AI, and custom profiles.
- Layout selection (5.1 vs 7.1) and configurable output reports directory.
- Background BenchmarkWorker execution with progress tracking.
- Comparative Results Table displaying Profile, Output Path, Peak Level (dBFS),
  Surround RMS, Front RMS, and Band Correlation / Coherence.
- Visual comparative delta cards highlighting differences in headroom,
  surround immersion, and phase correlation.
- One-click export for benchmark summary JSON and direct output directory navigation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..workers import BenchmarkWorker


def _format_db(linear: float) -> str:
    """Format linear amplitude to dBFS."""
    if linear <= 1e-9:
        return "-inf dBFS"
    db = 20.0 * np.log10(linear)
    return f"{db:+.1f} dBFS"


def _create_summary_card(title: str, value: str = "--", subtitle: str = "") -> tuple[QFrame, QLabel, QLabel]:
    card = QFrame()
    card.setStyleSheet(
        "QFrame { background-color: #21252b; border: 1px solid #2d3139; border-radius: 5px; padding: 8px; }"
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(2)

    lbl_t = QLabel(title)
    lbl_t.setStyleSheet("color: #7f848e; font-size: 11px; font-weight: 500;")
    layout.addWidget(lbl_t)

    lbl_v = QLabel(value)
    lbl_v.setStyleSheet("color: #61afef; font-size: 16px; font-weight: bold; font-family: monospace;")
    layout.addWidget(lbl_v)

    lbl_s = QLabel(subtitle)
    lbl_s.setStyleSheet("color: #5c6370; font-size: 10px;")
    layout.addWidget(lbl_s)

    return card, lbl_v, lbl_s


class BenchmarkTab(QWidget):
    """Multi-profile DSP comparative benchmark studio tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._active_worker: BenchmarkWorker | None = None
        self._last_results: dict[str, Any] = {}
        self._last_summary_path: str = ""

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # 1. Source Audio & Parameters Section
        layout.addWidget(self._build_config_section())

        # 2. Execution & Progress Bar
        layout.addWidget(self._build_exec_section())

        # 3. Comparative Results Table
        layout.addWidget(self._build_table_section())

        # 4. Comparative Visual Summary Cards
        layout.addWidget(self._build_summary_cards_section())

        # 5. Export & Action Footer
        layout.addWidget(self._build_footer_section())

        layout.addStretch(1)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # 1. Source Audio & Benchmark Parameters
    # -------------------------------------------------------------------------
    def _build_config_section(self) -> QGroupBox:
        box = QGroupBox("Benchmark Configuration")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # File picker row
        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select or drag & drop stereo audio file (.flac, .wav, .aiff, .mp3)...")
        file_row.addWidget(self.input_edit, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_input)
        file_row.addWidget(self.btn_browse)
        layout.addLayout(file_row)

        # Options grid
        grid = QGridLayout()
        grid.setSpacing(10)

        # Layout selector
        grid.addWidget(QLabel("Target Layout:"), 0, 0)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["7.1 Surround (8 ch)", "5.1 Surround (6 ch)"])
        grid.addWidget(self.combo_layout, 0, 1)

        # Reports directory
        grid.addWidget(QLabel("Reports Directory:"), 0, 2)
        dir_row = QHBoxLayout()
        dir_row.setSpacing(6)
        self.dir_edit = QLineEdit("reports")
        dir_row.addWidget(self.dir_edit, 1)
        self.btn_browse_dir = QPushButton("Browse...")
        self.btn_browse_dir.clicked.connect(self._browse_out_dir)
        dir_row.addWidget(self.btn_browse_dir)
        grid.addLayout(dir_row, 0, 3)

        layout.addLayout(grid)

        # Profile selection row
        prof_frame = QFrame()
        prof_frame.setStyleSheet(
            "QFrame { background-color: #1e2227; border: 1px solid #2d3139; border-radius: 4px; padding: 6px; }"
        )
        p_layout = QHBoxLayout(prof_frame)
        p_layout.setContentsMargins(8, 4, 8, 4)
        p_layout.setSpacing(16)

        p_layout.addWidget(QLabel("Profiles to Benchmark:"))
        self.chk_baseline = QCheckBox("Baseline (Fixed Matrix DSP)")
        self.chk_baseline.setChecked(True)
        p_layout.addWidget(self.chk_baseline)

        self.chk_adaptive = QCheckBox("Adaptive (Signal-Adaptive β)")
        self.chk_adaptive.setChecked(True)
        p_layout.addWidget(self.chk_adaptive)

        self.chk_adaptive_ai = QCheckBox("Adaptive-AI (Vocal Centre)")
        p_layout.addWidget(self.chk_adaptive_ai)

        p_layout.addStretch(1)
        layout.addWidget(prof_frame)

        return box

    # -------------------------------------------------------------------------
    # 2. Execution & Progress
    # -------------------------------------------------------------------------
    def _build_exec_section(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.btn_run = QPushButton("Run Comparative Benchmark")
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.setStyleSheet("QPushButton { font-weight: 600; font-size: 13px; padding: 6px 20px; }")
        self.btn_run.clicked.connect(self.run_benchmark)
        layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready for benchmark")
        layout.addWidget(self.progress_bar, 1)

        return container

    # -------------------------------------------------------------------------
    # 3. Comparative Results Table
    # -------------------------------------------------------------------------
    def _build_table_section(self) -> QGroupBox:
        box = QGroupBox("Comparative Benchmark Results")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels([
            "Profile",
            "Output Multichannel File",
            "Peak Level",
            "Surround RMS",
            "Front RMS",
            "Surround Ratio",
            "Band Correlation (Avg)",
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setMinimumHeight(150)
        layout.addWidget(self.results_table)

        return box

    # -------------------------------------------------------------------------
    # 4. Comparative Visual Summary Cards
    # -------------------------------------------------------------------------
    def _build_summary_cards_section(self) -> QGroupBox:
        box = QGroupBox("Acoustic Delta & Immersion Comparison")
        grid = QGridLayout(box)
        grid.setSpacing(10)

        # Headroom Card
        c1, self.lbl_headroom, self.lbl_headroom_sub = _create_summary_card(
            "HEADROOM MARGIN", "--", "Peak level relative to 0 dBFS"
        )
        grid.addWidget(c1, 0, 0)

        # Surround Delta Card
        c2, self.lbl_surround_delta, self.lbl_surround_delta_sub = _create_summary_card(
            "SURROUND ENERGY RATIO", "--", "Surround RMS / Front RMS"
        )
        grid.addWidget(c2, 0, 1)

        # Correlation Card
        c3, self.lbl_corr_summary, self.lbl_corr_summary_sub = _create_summary_card(
            "SPATIAL COHERENCE", "--", "Mean band Pearson rho"
        )
        grid.addWidget(c3, 0, 2)

        return box

    # -------------------------------------------------------------------------
    # 5. Export & Footer
    # -------------------------------------------------------------------------
    def _build_footer_section(self) -> QWidget:
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.btn_export_json = QPushButton("Export Summary JSON...")
        self.btn_export_json.setEnabled(False)
        self.btn_export_json.clicked.connect(self._export_json)
        layout.addWidget(self.btn_export_json)

        self.btn_open_folder = QPushButton("Open Reports Folder")
        self.btn_open_folder.clicked.connect(self._open_reports_folder)
        layout.addWidget(self.btn_open_folder)

        layout.addStretch(1)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #7f848e; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        return footer

    # -------------------------------------------------------------------------
    # File Browsing & Drag & Drop
    # -------------------------------------------------------------------------
    def load_input_file(self, file_path: str) -> None:
        path = Path(file_path).resolve()
        if not path.is_file():
            return
        self.input_edit.setText(str(path))

    def _browse_input(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File for Benchmark",
            "",
            "Audio Files (*.flac *.wav *.aiff *.aif *.mp3);;All Files (*.*)",
        )
        if file_path:
            self.load_input_file(file_path)

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Reports Directory", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith((".flac", ".wav", ".aiff", ".aif", ".mp3")) for url in urls):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                self.load_input_file(file_path)
                event.acceptProposedAction()
                break

    # -------------------------------------------------------------------------
    # Benchmark Execution
    # -------------------------------------------------------------------------
    def run_benchmark(self) -> None:
        in_path = self.input_edit.text().strip()
        if not in_path or not os.path.isfile(in_path):
            self.lbl_status.setText("Error: Please select a valid input audio file.")
            self.lbl_status.setStyleSheet("color: #e06c75;")
            return

        profiles: list[str] = []
        if self.chk_baseline.isChecked():
            profiles.append("baseline")
        if self.chk_adaptive.isChecked():
            profiles.append("adaptive")
        if self.chk_adaptive_ai.isChecked():
            profiles.append("adaptive-ai")

        if not profiles:
            self.lbl_status.setText("Error: Please select at least one profile to benchmark.")
            self.lbl_status.setStyleSheet("color: #e06c75;")
            return

        layout_str = "7.1" if "7.1" in self.combo_layout.currentText() else "5.1"
        out_dir = self.dir_edit.text().strip() or "reports"

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"Running benchmarks for: {', '.join(profiles)}...")
        self.lbl_status.setStyleSheet("color: #abb2bf;")

        self._active_worker = BenchmarkWorker(
            input_path=in_path,
            profiles=profiles,
            layout=layout_str,
            out_dir=out_dir,
            parent=self,
        )
        self._active_worker.progress_signal.connect(self._on_worker_progress)
        self._active_worker.finished_signal.connect(self._on_worker_finished)
        self._active_worker.error_signal.connect(self._on_worker_error)
        self._active_worker.start()

    def _on_worker_progress(self, current: int, total: int, stage_name: str) -> None:
        pct = int(round((current / max(1, total)) * 100))
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}% - {stage_name}")

    def _on_worker_finished(self, results: dict[str, Any], summary_path: str) -> None:
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Benchmark Completed")

        self._last_results = results
        self._last_summary_path = summary_path
        self.btn_export_json.setEnabled(True)

        self.lbl_status.setText(f"Benchmark finished. Summary saved to {Path(summary_path).name}")
        self.lbl_status.setStyleSheet("color: #98c379;")

        # Populate Results Table
        self.results_table.setRowCount(len(results))
        peaks: list[float] = []
        surround_ratios: list[float] = []
        corrs: list[float] = []

        for r, (prof, data) in enumerate(results.items()):
            out_file = Path(data.get("output", "")).name
            peak_val = float(data.get("peak", 0.0))
            sr_rms = float(data.get("surround_rms", 0.0))
            fr_rms = float(data.get("front_rms", 0.0))
            ratio = sr_rms / (fr_rms + 1e-12)

            peaks.append(peak_val)
            surround_ratios.append(ratio)

            bands = data.get("stereo_bands", [])
            avg_rho = float(np.mean([b["rho"] for b in bands])) if bands else 0.0
            avg_coh = float(np.mean([b["coherence"] for b in bands])) if bands else 0.0
            corrs.append(avg_rho)

            # Profile Item
            self.results_table.setItem(r, 0, QTableWidgetItem(prof.upper()))
            self.results_table.setItem(r, 1, QTableWidgetItem(out_file))

            # Peak Item
            peak_item = QTableWidgetItem(f"{peak_val:.4f} ({_format_db(peak_val)})")
            if peak_val > 1.0:
                peak_item.setForeground(QColor("#e06c75"))
            self.results_table.setItem(r, 2, peak_item)

            # Surround RMS
            self.results_table.setItem(r, 3, QTableWidgetItem(f"{sr_rms:.4f} ({_format_db(sr_rms)})"))

            # Front RMS
            self.results_table.setItem(r, 4, QTableWidgetItem(f"{fr_rms:.4f} ({_format_db(fr_rms)})"))

            # Surround Ratio
            self.results_table.setItem(r, 5, QTableWidgetItem(f"{ratio:.3f}x"))

            # Band Correlation Summary
            self.results_table.setItem(r, 6, QTableWidgetItem(f"ρ={avg_rho:+.3f}, γ²={avg_coh:.3f}"))

        # Update Visual Comparison Summary Cards
        if peaks:
            max_pk = max(peaks)
            headroom_db = 20.0 * np.log10(max(1e-9, 1.0 / max(1e-9, max_pk)))
            self.lbl_headroom.setText(f"{headroom_db:+.1f} dB")
            self.lbl_headroom_sub.setText("Peak clipping safe" if max_pk <= 1.0 else "⚠️ Exceeds 0 dBFS clipping")
            if max_pk > 1.0:
                self.lbl_headroom.setStyleSheet("color: #e06c75; font-size: 16px; font-weight: bold; font-family: monospace;")

        if surround_ratios:
            avg_ratio = float(np.mean(surround_ratios))
            self.lbl_surround_delta.setText(f"{avg_ratio:.3f}x")
            self.lbl_surround_delta_sub.setText("Surround-to-Front energy balance")

        if corrs:
            mean_corr = float(np.mean(corrs))
            self.lbl_corr_summary.setText(f"{mean_corr:+.3f}")
            self.lbl_corr_summary_sub.setText("Correlated stereo imaging" if mean_corr > 0.3 else "Wide / diffuse imaging")

    def _on_worker_error(self, message: str) -> None:
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Benchmark Failed: {message}")
        self.lbl_status.setStyleSheet("color: #e06c75;")

    # -------------------------------------------------------------------------
    # Export and Directory Handlers
    # -------------------------------------------------------------------------
    def _export_json(self) -> None:
        if not self._last_summary_path or not os.path.exists(self._last_summary_path):
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Benchmark Summary JSON",
            Path(self._last_summary_path).name,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if dest:
            shutil.copy2(self._last_summary_path, dest)
            self.lbl_status.setText(f"Exported benchmark summary to {dest}")

    def _open_reports_folder(self) -> None:
        out_dir = Path(self.dir_edit.text().strip() or "reports").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(out_dir))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(out_dir)], check=False)
        else:
            subprocess.run(["xdg-open", str(out_dir)], check=False)
