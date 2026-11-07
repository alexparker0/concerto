"""Stereo & Directness Analysis Lab tab for Concerto GUI.

Features:
- Source audio selection with drag-and-drop and one-click analysis trigger.
- High-performance interactive pyqtgraph SpectralCorrelationPlot showing frequency-dependent
  Pearson correlation rho(f), coherence gamma^2(f), and directness d(f).
- PhaseGoniometerWidget with Lissajous stereo vectorscope and [-1, +1] phase meter.
- Summary Metric Cards: Stereo Width, Mid RMS, Side RMS, Mean Directness,
  Ambience Energy Fraction, and Transient Density / Crest Factor.
- Frequency sub-band table with acoustic characterization.
- Report export triggers for CSV and JSON summaries.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets.plots import PhaseGoniometerWidget, SpectralCorrelationPlot
from ..workers import AnalysisWorker


def _create_metric_card(title: str, value: str = "--", subtitle: str = "") -> tuple[QFrame, QLabel, QLabel]:
    """Create a styled DAW metric card with title, large value, and subtitle."""
    card = QFrame()
    card.setStyleSheet(
        "QFrame { background-color: #21252b; border: 1px solid #2d3139; border-radius: 5px; padding: 6px; }"
    )
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(8, 6, 8, 6)
    card_layout.setSpacing(2)

    lbl_title = QLabel(title)
    lbl_title.setStyleSheet("color: #7f848e; font-size: 11px; font-weight: 500;")
    card_layout.addWidget(lbl_title)

    lbl_val = QLabel(value)
    lbl_val.setStyleSheet("color: #61afef; font-size: 16px; font-weight: bold; font-family: monospace;")
    card_layout.addWidget(lbl_val)

    lbl_sub = QLabel(subtitle)
    lbl_sub.setStyleSheet("color: #5c6370; font-size: 10px;")
    card_layout.addWidget(lbl_sub)

    return card, lbl_val, lbl_sub


def _format_db(linear: float) -> str:
    """Format linear amplitude to dBFS."""
    if linear <= 1e-9:
        return "-inf dBFS"
    db = 20.0 * np.log10(linear)
    return f"{db:+.1f} dBFS"


class AnalysisTab(QWidget):
    """Stereo & Directness Analysis Lab tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._active_worker: AnalysisWorker | None = None
        self._current_csv_path: str | None = None
        self._current_json_path: str | None = None
        self._current_summary: dict[str, Any] | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Scroll area for clean display across screens
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # 1. Top Control Bar (File selection, Browse, Analyze button, Progress)
        layout.addWidget(self._build_top_bar())

        # 2. Main Visualization Section (Correlation Plot + Goniometer)
        layout.addWidget(self._build_viz_section())

        # 3. Summary Metric Cards
        layout.addWidget(self._build_metrics_section())

        # 4. Octave / Sub-Band Breakdown Table
        layout.addWidget(self._build_bands_section())

        # 5. Export and Status Footer
        layout.addWidget(self._build_footer_section())

        layout.addStretch(1)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # 1. Top Control Bar
    # -------------------------------------------------------------------------
    def _build_top_bar(self) -> QGroupBox:
        box = QGroupBox("Audio Analysis Input")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select or drag & drop stereo audio file (.flac, .wav, .aiff, .mp3)...")
        row.addWidget(self.input_edit, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_input_file)
        row.addWidget(self.btn_browse)

        self.btn_analyze = QPushButton("Analyze Audio")
        self.btn_analyze.setObjectName("primaryButton")
        self.btn_analyze.setStyleSheet("QPushButton { font-weight: 600; padding: 6px 18px; }")
        self.btn_analyze.clicked.connect(self.start_analysis)
        row.addWidget(self.btn_analyze)

        layout.addLayout(row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready for analysis")
        self.progress_bar.setFixedHeight(16)
        layout.addWidget(self.progress_bar)

        return box

    # -------------------------------------------------------------------------
    # 2. Visualizations (Spectral Plot + Goniometer)
    # -------------------------------------------------------------------------
    def _build_viz_section(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Spectral Correlation Plot (Left / Main)
        plot_box = QGroupBox("Frequency-Dependent Spectral Correlation & Coherence")
        p_layout = QVBoxLayout(plot_box)
        p_layout.setContentsMargins(6, 6, 6, 6)
        self.spectral_plot = SpectralCorrelationPlot(plot_box)
        self.spectral_plot.setMinimumHeight(280)
        p_layout.addWidget(self.spectral_plot)
        layout.addWidget(plot_box, 3)

        # Lissajous Goniometer (Right)
        gonio_box = QGroupBox("Stereo Field Goniometer & Phase")
        g_layout = QVBoxLayout(gonio_box)
        g_layout.setContentsMargins(6, 6, 6, 6)
        self.goniometer = PhaseGoniometerWidget(gonio_box)
        self.goniometer.setMinimumSize(220, 280)
        g_layout.addWidget(self.goniometer)
        layout.addWidget(gonio_box, 2)

        return container

    # -------------------------------------------------------------------------
    # 3. Summary Metric Cards
    # -------------------------------------------------------------------------
    def _build_metrics_section(self) -> QGroupBox:
        box = QGroupBox("Stereo Field & Ambience Metrics")
        grid = QGridLayout(box)
        grid.setSpacing(10)

        # 1. Stereo Width
        c1, self.lbl_width, self.lbl_width_sub = _create_metric_card(
            "STEREO WIDTH", "--", "Side RMS / Mid RMS ratio"
        )
        grid.addWidget(c1, 0, 0)

        # 2. Mid RMS
        c2, self.lbl_mid_rms, self.lbl_mid_rms_sub = _create_metric_card(
            "MID ENERGY (M)", "--", "Sum (L+R)/2 RMS"
        )
        grid.addWidget(c2, 0, 1)

        # 3. Side RMS
        c3, self.lbl_side_rms, self.lbl_side_rms_sub = _create_metric_card(
            "SIDE ENERGY (S)", "--", "Difference (L-R)/2 RMS"
        )
        grid.addWidget(c3, 0, 2)

        # 4. Directness
        c4, self.lbl_directness, self.lbl_directness_sub = _create_metric_card(
            "DIRECTNESS d(f)", "--", "Dominant eigenvalue separation"
        )
        grid.addWidget(c4, 0, 3)

        # 5. Ambience Fraction
        c5, self.lbl_ambience, self.lbl_ambience_sub = _create_metric_card(
            "AMBIENCE FRACTION (A)", "--", "Diffuse / uncorrelated energy"
        )
        grid.addWidget(c5, 0, 4)

        # 6. Transient Crest Factor
        c6, self.lbl_crest, self.lbl_crest_sub = _create_metric_card(
            "CREST FACTOR (T)", "--", "Peak-to-RMS dynamic ratio"
        )
        grid.addWidget(c6, 0, 5)

        return box

    # -------------------------------------------------------------------------
    # 4. Frequency Sub-Bands Breakdown Table
    # -------------------------------------------------------------------------
    def _build_bands_section(self) -> QGroupBox:
        box = QGroupBox("Octave Sub-Band Analysis Breakdown")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        self.band_table = QTableWidget(0, 5)
        self.band_table.setHorizontalHeaderLabels([
            "Acoustic Band",
            "Frequency Range",
            "Correlation ρ(f)",
            "Coherence γ²(f)",
            "Stereo Character",
        ])
        self.band_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.band_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.band_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.band_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.band_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.band_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.band_table.setAlternatingRowColors(True)
        self.band_table.setMinimumHeight(130)
        layout.addWidget(self.band_table)

        return box

    # -------------------------------------------------------------------------
    # 5. Export & Status Footer
    # -------------------------------------------------------------------------
    def _build_footer_section(self) -> QWidget:
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.btn_export_csv = QPushButton("Export CSV Report...")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_csv)
        layout.addWidget(self.btn_export_csv)

        self.btn_export_json = QPushButton("Export JSON Report...")
        self.btn_export_json.setEnabled(False)
        self.btn_export_json.clicked.connect(self._export_json)
        layout.addWidget(self.btn_export_json)

        layout.addStretch(1)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #7f848e; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        return footer

    # -------------------------------------------------------------------------
    # Audio Probing & Input Handling
    # -------------------------------------------------------------------------
    def load_input_file(self, file_path: str) -> None:
        path = Path(file_path).resolve()
        if not path.is_file():
            return
        self.input_edit.setText(str(path))
        self.start_analysis()

    def _browse_input_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File to Analyze",
            "",
            "Audio Files (*.flac *.wav *.aiff *.aif *.mp3 *.ogg *.opus);;All Files (*.*)",
        )
        if file_path:
            self.input_edit.setText(file_path)

    # -------------------------------------------------------------------------
    # Drag and Drop Support
    # -------------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith((".flac", ".wav", ".aiff", ".aif", ".mp3", ".ogg", ".opus")) for url in urls):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                self.input_edit.setText(file_path)
                event.acceptProposedAction()
                self.start_analysis()
                break

    # -------------------------------------------------------------------------
    # Analysis Execution
    # -------------------------------------------------------------------------
    def start_analysis(self) -> None:
        in_path = self.input_edit.text().strip()
        if not in_path or not os.path.isfile(in_path):
            self.lbl_status.setText("Error: Please select a valid input audio file.")
            self.lbl_status.setStyleSheet("color: #e06c75;")
            return

        self.btn_analyze.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Decoding audio and computing spectral correlation...")
        self.lbl_status.setStyleSheet("color: #abb2bf;")

        # Run analysis worker
        self._active_worker = AnalysisWorker(
            input_path=in_path,
            out_dir="reports",
            parent=self,
        )
        self._active_worker.progress_signal.connect(self._on_worker_progress)
        self._active_worker.finished_signal.connect(self._on_worker_finished)
        self._active_worker.error_signal.connect(self._on_worker_error)
        self._active_worker.start()

        # Update goniometer with real audio snippet
        try:
            data, _ = sf.read(in_path, stop=48000 * 5, dtype="float32", always_2d=True)
            if data.shape[1] >= 2:
                self.goniometer.set_audio(data[:, 0], data[:, 1])
            elif data.shape[1] == 1:
                self.goniometer.set_audio(data[:, 0], data[:, 0])
        except Exception:
            pass

    def _on_worker_progress(self, current: int, total: int, stage_name: str) -> None:
        pct = int(round((current / max(1, total)) * 100))
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}% - {stage_name}")

    def _on_worker_finished(self, summary: dict[str, Any], csv_path: str, json_path: str) -> None:
        self.btn_analyze.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Analysis Complete")

        self._current_summary = summary
        self._current_csv_path = csv_path
        self._current_json_path = json_path

        self.btn_export_csv.setEnabled(True)
        self.btn_export_json.setEnabled(True)
        self.lbl_status.setText(f"Reports saved to {Path(csv_path).name} and {Path(json_path).name}")
        self.lbl_status.setStyleSheet("color: #98c379;")

        # 1. Update Spectral Correlation Plot
        freqs = np.asarray(summary["frequencies"], dtype=np.float64)
        rho = np.asarray(summary["rho_freq"], dtype=np.float64)
        coh = np.asarray(summary["coherence_freq"], dtype=np.float64)
        directness = np.asarray(summary["directness_freq"], dtype=np.float64)
        self.spectral_plot.set_data(freqs, rho, coh, directness)

        # 2. Update Metric Cards
        mid_rms = float(summary.get("mid_rms", 0.0))
        side_rms = float(summary.get("side_rms", 0.0))
        width_ratio = side_rms / (mid_rms + 1e-12)
        mean_direct = float(np.mean(directness)) if len(directness) > 0 else 0.0
        ambience_fraction = float(np.clip(1.0 - mean_direct, 0.0, 1.0))

        # Calculate crest factor
        try:
            in_path = self.input_edit.text().strip()
            data, _ = sf.read(in_path, stop=48000 * 5, dtype="float32", always_2d=True)
            peak_val = float(np.max(np.abs(data)))
            rms_val = float(np.sqrt(np.mean(data**2)))
            crest_factor = peak_val / (rms_val + 1e-12)
            crest_db = 20.0 * np.log10(crest_factor + 1e-12)
            self.lbl_crest.setText(f"{crest_factor:.2f}x")
            self.lbl_crest_sub.setText(f"{crest_db:+.1f} dB peak-to-RMS")
        except Exception:
            self.lbl_crest.setText("--")
            self.lbl_crest_sub.setText("Peak-to-RMS ratio")

        self.lbl_width.setText(f"{width_ratio:.3f}")
        self.lbl_width_sub.setText(f"{'Centered / Narrow' if width_ratio < 0.3 else 'Balanced' if width_ratio < 0.8 else 'Ultra-Wide'}")

        self.lbl_mid_rms.setText(_format_db(mid_rms))
        self.lbl_mid_rms_sub.setText(f"Linear: {mid_rms:.4f}")

        self.lbl_side_rms.setText(_format_db(side_rms))
        self.lbl_side_rms_sub.setText(f"Linear: {side_rms:.4f}")

        self.lbl_directness.setText(f"{mean_direct:.3f}")
        self.lbl_directness_sub.setText(f"{'Highly Diffuse' if mean_direct < 0.3 else 'Mixed' if mean_direct < 0.7 else 'Highly Direct'}")

        self.lbl_ambience.setText(f"{ambience_fraction * 100.0:.1f}%")
        self.lbl_ambience_sub.setText(f"Ambient energy estimate")

        # 3. Update Sub-Bands Table
        bands = summary.get("bands", [])
        self.band_table.setRowCount(len(bands))

        band_ranges = {
            "<100": "20 Hz - 100 Hz (Sub-Bass)",
            "100-1000": "100 Hz - 1 kHz (Low/Mid)",
            "1000-8000": "1 kHz - 8 kHz (Presence)",
            "8000-16000": "8 kHz - 16 kHz (Brilliance)",
            ">16000": "16 kHz - 20 kHz (Air)",
        }

        for r, item in enumerate(bands):
            b_name = item.get("band", str(r))
            b_rho = float(item.get("rho", 0.0))
            b_coh = float(item.get("coherence", 0.0))

            # Character descriptor
            if b_rho > 0.8:
                char_desc = "Mono / Centered"
                char_color = "#98c379"
            elif b_rho > 0.3:
                char_desc = "Stereo Correlated"
                char_color = "#61afef"
            elif b_rho > -0.2:
                char_desc = "Uncorrelated / Diffuse"
                char_color = "#e5c07b"
            else:
                char_desc = "Anti-Phase / Cancelled"
                char_color = "#e06c75"

            f_range = band_ranges.get(b_name, b_name)

            self.band_table.setItem(r, 0, QTableWidgetItem(f"Band {b_name}"))
            self.band_table.setItem(r, 1, QTableWidgetItem(f_range))

            rho_item = QTableWidgetItem(f"{b_rho:+.4f}")
            self.band_table.setItem(r, 2, rho_item)

            coh_item = QTableWidgetItem(f"{b_coh:.4f}")
            self.band_table.setItem(r, 3, coh_item)

            desc_item = QTableWidgetItem(char_desc)
            desc_item.setForeground(QColor(char_color))
            self.band_table.setItem(r, 4, desc_item)

    def _on_worker_error(self, message: str) -> None:
        self.btn_analyze.setEnabled(True)
        self.lbl_status.setText(f"Analysis Failed: {message}")
        self.lbl_status.setStyleSheet("color: #e06c75;")

    # -------------------------------------------------------------------------
    # Report Export Handlers
    # -------------------------------------------------------------------------
    def _export_csv(self) -> None:
        if not self._current_csv_path or not os.path.exists(self._current_csv_path):
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis CSV Report",
            Path(self._current_csv_path).name,
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if dest:
            shutil.copy2(self._current_csv_path, dest)
            self.lbl_status.setText(f"Exported CSV to {dest}")

    def _export_json(self) -> None:
        if not self._current_json_path or not os.path.exists(self._current_json_path):
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis JSON Report",
            Path(self._current_json_path).name,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if dest:
            shutil.copy2(self._current_json_path, dest)
            self.lbl_status.setText(f"Exported JSON to {dest}")
