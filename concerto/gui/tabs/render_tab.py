"""Upmix & Render Studio tab for Concerto GUI.

Features:
- Source audio file selection with drag-and-drop and audio metadata inspection.
- Output destination configuration.
- Comprehensive render parameters: 5.1 / 7.1 layout, Baseline vs Adaptive modes,
  Neural stem separation & vocal centre controls, streaming vs in-memory, TPDF dither,
  and preset loading.
- Real-time execution dashboard with multi-stage progress, speed (rt factor),
  elapsed/ETA timers, and peak dBFS warning badge.
- Batch queue management table supporting multi-file and folder processing.
- Seamless integration with RenderWorker and AudioPlayer.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QIcon
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...dsp.upmix import load_config
from ..workers import RenderWorker


def _format_time(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS string."""
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_db(linear: float) -> str:
    """Format linear amplitude to dBFS string."""
    if linear <= 1e-9:
        return "-inf dBFS"
    db = 20.0 * np.log10(linear)
    return f"{db:+.1f} dBFS"


class RenderTab(QWidget):
    """Main Upmix & Render Studio tab."""

    # Emitted when a render finishes successfully, passing the output file path
    audio_rendered = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._active_worker: RenderWorker | None = None
        self._batch_queue: list[dict[str, Any]] = []
        self._batch_running: bool = False
        self._current_batch_index: int = -1
        self._current_input_meta: dict[str, Any] | None = None

        self._init_ui()

    # -------------------------------------------------------------------------
    # UI Initialization
    # -------------------------------------------------------------------------
    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Scroll area so UI fits all display resolutions comfortably
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        scroll_content = QWidget(scroll_area)
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(12)

        # 1. Source Audio Section & Info Card
        content_layout.addWidget(self._build_source_section())

        # 2. Output Configuration Section
        content_layout.addWidget(self._build_output_section())

        # 3. Render Parameters Section (Layout, Mode, AI, Options)
        content_layout.addWidget(self._build_params_section())

        # 4. Live Execution Dashboard
        content_layout.addWidget(self._build_execution_section())

        # 5. Batch Processing Queue
        content_layout.addWidget(self._build_batch_section())

        content_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

    # -------------------------------------------------------------------------
    # 1. Source Audio Section
    # -------------------------------------------------------------------------
    def _build_source_section(self) -> QGroupBox:
        box = QGroupBox("1. Source Audio")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # File picker row
        file_row = QHBoxLayout()
        file_row.setSpacing(8)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select or drag & drop stereo audio file (.flac, .wav, .aiff, .mp3)...")
        self.input_edit.textChanged.connect(self._on_input_path_changed)
        file_row.addWidget(self.input_edit, 1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_input_file)
        file_row.addWidget(self.browse_btn)

        layout.addLayout(file_row)

        # Audio Info Card Container
        self.info_card = QFrame()
        self.info_card.setStyleSheet(
            "QFrame { background-color: #1e2227; border: 1px solid #2d3139; border-radius: 4px; padding: 6px; }"
        )
        info_grid = QGridLayout(self.info_card)
        info_grid.setContentsMargins(8, 6, 8, 6)
        info_grid.setHorizontalSpacing(16)
        info_grid.setVerticalSpacing(4)

        # Labels
        self.lbl_sample_rate = QLabel("Sample Rate: --")
        self.lbl_channels = QLabel("Channels: --")
        self.lbl_bit_depth = QLabel("Format: --")
        self.lbl_duration = QLabel("Duration: --")
        self.lbl_peak = QLabel("Peak: --")
        self.lbl_rms = QLabel("RMS: --")

        for lbl in (self.lbl_sample_rate, self.lbl_channels, self.lbl_bit_depth,
                    self.lbl_duration, self.lbl_peak, self.lbl_rms):
            lbl.setStyleSheet("color: #abb2bf; font-size: 11px;")

        info_grid.addWidget(self.lbl_sample_rate, 0, 0)
        info_grid.addWidget(self.lbl_channels, 0, 1)
        info_grid.addWidget(self.lbl_bit_depth, 0, 2)
        info_grid.addWidget(self.lbl_duration, 1, 0)
        info_grid.addWidget(self.lbl_peak, 1, 1)
        info_grid.addWidget(self.lbl_rms, 1, 2)

        layout.addWidget(self.info_card)
        return box

    # -------------------------------------------------------------------------
    # 2. Output Section
    # -------------------------------------------------------------------------
    def _build_output_section(self) -> QGroupBox:
        box = QGroupBox("2. Destination")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output multichannel file path (auto-derived or custom)...")
        out_row.addWidget(self.output_edit, 1)

        self.out_browse_btn = QPushButton("Browse...")
        self.out_browse_btn.clicked.connect(self._browse_output_file)
        out_row.addWidget(self.out_browse_btn)

        layout.addLayout(out_row)
        return box

    # -------------------------------------------------------------------------
    # 3. Render Parameters Section
    # -------------------------------------------------------------------------
    def _build_params_section(self) -> QGroupBox:
        box = QGroupBox("3. Render Parameters")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        # Primary controls grid
        grid = QGridLayout()
        grid.setSpacing(10)

        # Layout selector
        grid.addWidget(QLabel("Surround Layout:"), 0, 0)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["7.1 Surround (8 Channels)", "5.1 Surround (6 Channels)"])
        self.combo_layout.currentIndexChanged.connect(self._on_layout_changed)
        grid.addWidget(self.combo_layout, 0, 1)

        # Mode selector
        grid.addWidget(QLabel("Processing Mode:"), 0, 2)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Baseline (Matrix DSP)", "Adaptive (Signal-Adaptive β & Transient Envelopes)"])
        grid.addWidget(self.combo_mode, 0, 3)

        # Preset selector
        grid.addWidget(QLabel("Config Preset:"), 1, 0)
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["default.toml", "adaptive.toml", "adaptive-ai.toml", "Custom TOML..."])
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        grid.addWidget(self.combo_preset, 1, 1)

        # Architecture switches
        switches_box = QHBoxLayout()
        switches_box.setSpacing(16)

        self.chk_streaming = QCheckBox("Streaming Mode (--stream)")
        self.chk_streaming.setToolTip("Low-memory ring buffer mode for long files. Disables neural AI separation.")
        self.chk_streaming.toggled.connect(self._on_streaming_toggled)
        switches_box.addWidget(self.chk_streaming)

        self.chk_dither = QCheckBox("TPDF Dither (--dither)")
        self.chk_dither.setToolTip("Apply triangular probability density function dither on 16/24-bit export.")
        switches_box.addWidget(self.chk_dither)

        switches_box.addStretch(1)
        grid.addLayout(switches_box, 1, 2, 1, 2)

        layout.addLayout(grid)

        # AI Vocal Centre & Stem Separation Card
        ai_card = QFrame()
        ai_card.setStyleSheet("QFrame { background-color: #1e2227; border: 1px solid #2d3139; border-radius: 4px; padding: 8px; }")
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setSpacing(8)

        ai_header = QHBoxLayout()
        self.chk_ai = QCheckBox("Enable Neural Stem Separation & Vocal Centre (--ai on)")
        self.chk_ai.setStyleSheet("font-weight: 600; color: #61afef;")
        self.chk_ai.toggled.connect(self._on_ai_toggled)
        ai_header.addWidget(self.chk_ai)

        self.chk_cache = QCheckBox("Cache Stems (.concerto_cache)")
        self.chk_cache.setChecked(True)
        self.chk_cache.setToolTip("Store separated stem buffers on disk to accelerate repeated renders.")
        ai_header.addWidget(self.chk_cache)
        ai_header.addStretch(1)
        ai_layout.addLayout(ai_header)

        # AI options row
        self.ai_options_widget = QWidget()
        ai_opts_layout = QGridLayout(self.ai_options_widget)
        ai_opts_layout.setContentsMargins(0, 0, 0, 0)
        ai_opts_layout.setSpacing(8)

        ai_opts_layout.addWidget(QLabel("Neural Model:"), 0, 0)
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "htdemucs (PyTorch)",
        ])
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        ai_opts_layout.addWidget(self.combo_model, 0, 1)

        ai_opts_layout.addWidget(QLabel("Vocal Centre Gain:"), 0, 2)
        gain_box = QHBoxLayout()
        gain_box.setSpacing(6)
        self.slider_vocal_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_vocal_gain.setRange(-120, 0)  # -12.0 dB to 0.0 dB
        self.slider_vocal_gain.setValue(-60)     # -6.0 dB default
        self.slider_vocal_gain.valueChanged.connect(self._on_vocal_gain_changed)
        gain_box.addWidget(self.slider_vocal_gain, 1)

        self.lbl_vocal_gain = QLabel("-6.0 dB")
        self.lbl_vocal_gain.setMinimumWidth(50)
        self.lbl_vocal_gain.setStyleSheet("color: #e5c07b; font-family: monospace;")
        gain_box.addWidget(self.lbl_vocal_gain)
        ai_opts_layout.addLayout(gain_box, 0, 3)

        ai_layout.addWidget(self.ai_options_widget)
        self.ai_options_widget.setEnabled(False)  # Disabled until AI is checked

        layout.addWidget(ai_card)
        return box

    # -------------------------------------------------------------------------
    # 4. Live Execution Dashboard
    # -------------------------------------------------------------------------
    def _build_execution_section(self) -> QGroupBox:
        box = QGroupBox("4. Live Execution & Progress")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # Action row
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.btn_render = QPushButton("Start Render")
        self.btn_render.setObjectName("primaryButton")
        self.btn_render.setStyleSheet("QPushButton { font-size: 13px; font-weight: 600; padding: 6px 20px; }")
        self.btn_render.clicked.connect(self.start_render)
        action_row.addWidget(self.btn_render)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("dangerButton")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_render)
        action_row.addWidget(self.btn_cancel)

        self.lbl_stage = QLabel("Status: Ready")
        self.lbl_stage.setStyleSheet("color: #abb2bf; font-weight: 500;")
        action_row.addWidget(self.lbl_stage, 1)

        layout.addLayout(action_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - Ready")
        layout.addWidget(self.progress_bar)

        # Realtime metrics row
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("QFrame { background-color: #1e2227; border: 1px solid #2d3139; border-radius: 4px; padding: 4px; }")
        m_layout = QHBoxLayout(metrics_frame)
        m_layout.setContentsMargins(10, 4, 10, 4)
        m_layout.setSpacing(20)

        self.lbl_speed = QLabel("Speed: --")
        self.lbl_elapsed = QLabel("Elapsed: 00:00")
        self.lbl_eta = QLabel("ETA: --:--")
        self.lbl_out_peak = QLabel("Peak: --")

        for lbl in (self.lbl_speed, self.lbl_elapsed, self.lbl_eta, self.lbl_out_peak):
            lbl.setStyleSheet("color: #abb2bf; font-family: monospace; font-size: 11px;")
        self.lbl_speed.setStyleSheet("color: #61afef; font-family: monospace; font-size: 11px; font-weight: 600;")

        m_layout.addWidget(self.lbl_speed)
        m_layout.addWidget(self.lbl_elapsed)
        m_layout.addWidget(self.lbl_eta)
        m_layout.addWidget(self.lbl_out_peak)
        m_layout.addStretch(1)

        layout.addWidget(metrics_frame)

        # Peak warning badge (hidden unless peak > 1.0)
        self.lbl_peak_warning = QLabel("⚠️ WARNING: Output peak exceeds 0 dBFS (clipping detected) - consider lowering surround/vocal gain")
        self.lbl_peak_warning.setStyleSheet(
            "QLabel { background-color: #3e1b1e; color: #e06c75; border: 1px solid #e06c75; "
            "border-radius: 4px; padding: 4px 8px; font-weight: bold; font-size: 11px; }"
        )
        self.lbl_peak_warning.setVisible(False)
        layout.addWidget(self.lbl_peak_warning)

        return box

    # -------------------------------------------------------------------------
    # 5. Batch Processing Queue Section
    # -------------------------------------------------------------------------
    def _build_batch_section(self) -> QGroupBox:
        box = QGroupBox("5. Batch Queue")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # Button toolbar
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_batch_add_files = QPushButton("Add Files...")
        self.btn_batch_add_files.clicked.connect(self._batch_add_files)
        btn_row.addWidget(self.btn_batch_add_files)

        self.btn_batch_add_folder = QPushButton("Add Folder...")
        self.btn_batch_add_folder.clicked.connect(self._batch_add_folder)
        btn_row.addWidget(self.btn_batch_add_folder)

        self.btn_batch_remove = QPushButton("Remove Selected")
        self.btn_batch_remove.clicked.connect(self._batch_remove_selected)
        btn_row.addWidget(self.btn_batch_remove)

        self.btn_batch_clear = QPushButton("Clear Finished")
        self.btn_batch_clear.clicked.connect(self._batch_clear_finished)
        btn_row.addWidget(self.btn_batch_clear)

        btn_row.addStretch(1)

        self.btn_batch_start = QPushButton("Start Batch")
        self.btn_batch_start.setObjectName("primaryButton")
        self.btn_batch_start.clicked.connect(self._batch_start_queue)
        btn_row.addWidget(self.btn_batch_start)

        layout.addLayout(btn_row)

        # Queue table
        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(["Source File", "Layout", "Mode", "Status", "Output Path"])
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setMinimumHeight(140)
        layout.addWidget(self.batch_table)

        return box

    # -------------------------------------------------------------------------
    # Audio Probing and File Handling
    # -------------------------------------------------------------------------
    def load_input_file(self, file_path: str) -> None:
        """Set the input file path and inspect audio metadata."""
        path = Path(file_path).resolve()
        if not path.is_file():
            return
        self.input_edit.setText(str(path))

    def _browse_input_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Stereo Audio File",
            "",
            "Audio Files (*.flac *.wav *.aiff *.aif *.mp3 *.ogg *.opus);;All Files (*.*)",
        )
        if file_path:
            self.load_input_file(file_path)

    def _browse_output_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Destination Multichannel File",
            self.output_edit.text() or "rendered_surround.flac",
            "FLAC Multichannel (*.flac);;WAV Multichannel (*.wav);;All Files (*.*)",
        )
        if file_path:
            self.output_edit.setText(file_path)

    def _on_input_path_changed(self, text: str) -> None:
        path_str = text.strip()
        if not path_str or not os.path.isfile(path_str):
            self._reset_info_card()
            return

        try:
            info = sf.info(path_str)
            dur_str = _format_time(info.duration)
            sr_str = f"{info.samplerate:,} Hz"
            ch_str = f"{info.channels} ({'Stereo' if info.channels == 2 else 'Multi/Mono'})"
            fmt_str = f"{info.subtype} / {info.format}"

            # Calculate fast peak / RMS approximation from initial block
            peak_val = 0.0
            rms_val = 0.0
            try:
                # Read up to 10 seconds or 480k samples for responsive preview
                block, _ = sf.read(path_str, stop=min(int(info.samplerate * 10), info.frames), dtype="float32", always_2d=True)
                if block.size > 0:
                    peak_val = float(np.max(np.abs(block)))
                    rms_val = float(np.sqrt(np.mean(block**2)))
            except Exception:
                pass

            peak_str = _format_db(peak_val)
            rms_str = _format_db(rms_val)

            self.lbl_sample_rate.setText(f"Sample Rate: {sr_str}")
            self.lbl_channels.setText(f"Channels: {ch_str}")
            self.lbl_bit_depth.setText(f"Format: {fmt_str}")
            self.lbl_duration.setText(f"Duration: {dur_str}")
            self.lbl_peak.setText(f"Peak: {peak_str}")
            self.lbl_rms.setText(f"RMS: {rms_str}")

            self._current_input_meta = {
                "samplerate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
                "peak": peak_val,
                "rms": rms_val,
            }

            # Auto-derive output path if not already customized or if matching standard convention
            current_out = self.output_edit.text().strip()
            p = Path(path_str)
            layout_tag = "7.1" if "7.1" in self.combo_layout.currentText() else "5.1"
            derived_out = str(p.parent / f"{p.stem}_{layout_tag}.flac")
            if not current_out or current_out.endswith("_7.1.flac") or current_out.endswith("_5.1.flac"):
                self.output_edit.setText(derived_out)

        except Exception as e:
            self._reset_info_card(error=str(e))

    def _reset_info_card(self, error: str | None = None) -> None:
        self._current_input_meta = None
        msg = f"Error: {error}" if error else "--"
        self.lbl_sample_rate.setText(f"Sample Rate: {msg}")
        self.lbl_channels.setText("Channels: --")
        self.lbl_bit_depth.setText("Format: --")
        self.lbl_duration.setText("Duration: --")
        self.lbl_peak.setText("Peak: --")
        self.lbl_rms.setText("RMS: --")

    def _on_layout_changed(self) -> None:
        in_path = self.input_edit.text().strip()
        if in_path:
            p = Path(in_path)
            layout_tag = "7.1" if "7.1" in self.combo_layout.currentText() else "5.1"
            current_out = self.output_edit.text().strip()
            if not current_out or current_out.endswith("_7.1.flac") or current_out.endswith("_5.1.flac"):
                self.output_edit.setText(str(p.parent / f"{p.stem}_{layout_tag}.flac"))

    def _on_preset_changed(self) -> None:
        preset_name = self.combo_preset.currentText()
        if preset_name == "Custom TOML...":
            custom_path, _ = QFileDialog.getOpenFileName(
                self, "Select Custom TOML Config", "", "TOML Config (*.toml);;All Files (*.*)"
            )
            if custom_path:
                self.combo_preset.setItemText(self.combo_preset.currentIndex(), Path(custom_path).name)
                self.combo_preset.setToolTip(custom_path)
        elif "ai" in preset_name:
            self.chk_ai.setChecked(True)
            self.combo_mode.setCurrentIndex(1)  # Adaptive
        elif "adaptive" in preset_name:
            self.combo_mode.setCurrentIndex(1)  # Adaptive

    def _on_streaming_toggled(self, checked: bool) -> None:
        if checked and self.chk_ai.isChecked():
            self.chk_ai.setChecked(False)

    def _on_ai_toggled(self, checked: bool) -> None:
        self.ai_options_widget.setEnabled(checked)
        if checked and self.chk_streaming.isChecked():
            self.chk_streaming.setChecked(False)

    def _on_model_changed(self) -> None:
        pass

    def _on_vocal_gain_changed(self, value: int) -> None:
        db = value / 10.0
        self.lbl_vocal_gain.setText(f"{db:+.1f} dB")

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
                self.load_input_file(file_path)
                event.acceptProposedAction()
                break

    # -------------------------------------------------------------------------
    # Render Execution
    # -------------------------------------------------------------------------
    def start_render(self) -> None:
        """Validate inputs and launch RenderWorker."""
        in_path = self.input_edit.text().strip()
        if not in_path or not os.path.isfile(in_path):
            self.lbl_stage.setText("Status Error: Please select a valid input audio file.")
            self.lbl_stage.setStyleSheet("color: #e06c75; font-weight: 500;")
            return

        out_path = self.output_edit.text().strip()
        if not out_path:
            p = Path(in_path)
            layout_tag = "7.1" if "7.1" in self.combo_layout.currentText() else "5.1"
            out_path = str(p.parent / f"{p.stem}_{layout_tag}.flac")
            self.output_edit.setText(out_path)

        # Build render config
        layout_str = "7.1" if "7.1" in self.combo_layout.currentText() else "5.1"
        mode_str = "adaptive" if "Adaptive" in self.combo_mode.currentText() else "baseline"
        ai_str = "on" if self.chk_ai.isChecked() else "off"
        stream_mode = self.chk_streaming.isChecked()
        dither_mode = self.chk_dither.isChecked()
        no_cache = not self.chk_cache.isChecked()

        repo_root = Path(__file__).resolve().parent.parent.parent.parent

        # Model path (alpha uses htdemucs)
        onnx_model_arg: str | None = None
        # Load base configuration with guaranteed default fallback
        preset_name = self.combo_preset.currentText()
        if preset_name == "Custom TOML...":
            preset_path = Path(self.combo_preset.toolTip())
        else:
            candidates = [
                Path("configs") / preset_name,
                repo_root / "configs" / preset_name,
                Path(preset_name),
            ]
            preset_path = candidates[0]
            for c in candidates:
                if c.is_file():
                    preset_path = c
                    break

        try:
            cfg = load_config(preset_path)
        except Exception:
            try:
                cfg = load_config(repo_root / "configs" / "default.toml")
            except Exception:
                cfg = {}
        # Overlay GUI overrides
        cfg["layout"] = layout_str
        cfg["dither"] = dither_mode
        if ai_str == "on":
            cfg["centre_mode"] = "vocal_mask"
            cfg["vocal_gain_db"] = self.slider_vocal_gain.value() / 10.0

        # UI state transitions
        self.btn_render.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_peak_warning.setVisible(False)
        self.lbl_stage.setText("Starting render pipeline...")
        self.lbl_stage.setStyleSheet("color: #abb2bf; font-weight: 500;")

        # Launch worker
        self._active_worker = RenderWorker(
            input_path=in_path,
            output_path=out_path,
            cfg=cfg,
            mode=mode_str,
            layout=layout_str,
            ai=ai_str,
            onnx_model=onnx_model_arg,
            stream=stream_mode,
            dither=dither_mode,
            no_cache=no_cache,
            parent=self,
        )

        self._active_worker.progress_signal.connect(self._on_worker_progress)
        self._active_worker.metrics_signal.connect(self._on_worker_metrics)
        self._active_worker.status_signal.connect(self._on_worker_status)
        self._active_worker.finished_signal.connect(self._on_worker_finished)
        self._active_worker.error_signal.connect(self._on_worker_error)
        self._active_worker.start()

    def cancel_render(self) -> None:
        """Cancel the ongoing render worker."""
        if self._active_worker and self._active_worker.isRunning():
            self.lbl_stage.setText("Cancelling render...")
            self.lbl_stage.setStyleSheet("color: #e5c07b; font-weight: 500;")
            self._active_worker.cancel()
            self.btn_cancel.setEnabled(False)

    # -------------------------------------------------------------------------
    # Worker Signal Slots
    # -------------------------------------------------------------------------
    def _on_worker_progress(self, current: int, total: int, stage_name: str) -> None:
        pct = int(round((current / max(1, total)) * 100))
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"%p% - Stage: {stage_name}")
        self.lbl_stage.setText(f"Stage: {stage_name} ({current}/{total})")

    def _on_worker_metrics(self, speed_rt: float, elapsed_s: float, eta_s: float, peak: float) -> None:
        self.lbl_speed.setText(f"Speed: {speed_rt:.2f}x rt")
        self.lbl_elapsed.setText(f"Elapsed: {_format_time(elapsed_s)}")
        self.lbl_eta.setText(f"ETA: {_format_time(eta_s)}")
        self.lbl_out_peak.setText(f"Peak: {_format_db(peak)}")

        if peak > 1.0:
            self.lbl_peak_warning.setVisible(True)

    def _on_worker_status(self, msg: str) -> None:
        self.lbl_stage.setText(msg)

    def _on_worker_finished(self, summary: dict[str, Any]) -> None:
        self.btn_render.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100% - Finished")

        out_path = summary.get("output", self.output_edit.text())
        dur_s = summary.get("duration_s", 0.0)
        peak = summary.get("peak", 0.0)

        self.lbl_stage.setText(f"Render completed: {Path(out_path).name} ({dur_s:.1f}s, peak: {_format_db(peak)})")
        self.lbl_stage.setStyleSheet("color: #98c379; font-weight: 600;")

        if peak > 1.0:
            self.lbl_peak_warning.setVisible(True)

        # Emit completion signal
        self.audio_rendered.emit(out_path)

        # If batch is running, advance to next queue item
        if self._batch_running:
            self._on_batch_item_finished(success=True, out_path=out_path)

    def _on_worker_error(self, message: str) -> None:
        self.btn_render.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_stage.setText(f"Render Failed: {message}")
        self.lbl_stage.setStyleSheet("color: #e06c75; font-weight: 600;")

        if self._batch_running:
            self._on_batch_item_finished(success=False, error_msg=message)

    # -------------------------------------------------------------------------
    # Batch Queue Operations
    # -------------------------------------------------------------------------
    def _batch_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files for Batch Queue",
            "",
            "Audio Files (*.flac *.wav *.aiff *.aif *.mp3 *.ogg *.opus);;All Files (*.*)",
        )
        for f in files:
            self._add_to_queue(f)

    def _batch_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Audio Files")
        if folder:
            p = Path(folder)
            for ext in ("*.flac", "*.wav", "*.aiff", "*.aif", "*.mp3", "*.ogg", "*.opus"):
                for f in p.glob(ext):
                    self._add_to_queue(str(f))

    def _add_to_queue(self, file_path: str) -> None:
        p = Path(file_path).resolve()
        layout_str = "7.1" if "7.1" in self.combo_layout.currentText() else "5.1"
        mode_str = "adaptive" if "Adaptive" in self.combo_mode.currentText() else "baseline"
        out_path = str(p.parent / f"{p.stem}_{layout_str}.flac")

        item = {
            "file": str(p),
            "layout": layout_str,
            "mode": mode_str,
            "status": "Pending",
            "output": out_path,
        }
        self._batch_queue.append(item)

        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)
        self.batch_table.setItem(row, 0, QTableWidgetItem(p.name))
        self.batch_table.setItem(row, 1, QTableWidgetItem(layout_str))
        self.batch_table.setItem(row, 2, QTableWidgetItem(mode_str))

        status_item = QTableWidgetItem("Pending")
        status_item.setForeground(QColor("#abb2bf"))
        self.batch_table.setItem(row, 3, status_item)
        self.batch_table.setItem(row, 4, QTableWidgetItem(out_path))

    def _batch_remove_selected(self) -> None:
        rows = sorted(set(idx.row() for idx in self.batch_table.selectedIndexes()), reverse=True)
        for r in rows:
            if 0 <= r < len(self._batch_queue):
                del self._batch_queue[r]
            self.batch_table.removeRow(r)

    def _batch_clear_finished(self) -> None:
        for r in range(self.batch_table.rowCount() - 1, -1, -1):
            status_item = self.batch_table.item(r, 3)
            if status_item and status_item.text() in ("Completed", "Failed"):
                if 0 <= r < len(self._batch_queue):
                    del self._batch_queue[r]
                self.batch_table.removeRow(r)

    def _batch_start_queue(self) -> None:
        if not self._batch_queue or self._batch_running:
            return
        self._batch_running = True
        self._current_batch_index = -1
        self._process_next_batch_item()

    def _process_next_batch_item(self) -> None:
        self._current_batch_index += 1
        if self._current_batch_index >= len(self._batch_queue):
            # Batch queue complete
            self._batch_running = False
            self.lbl_stage.setText("Batch processing queue finished.")
            self.lbl_stage.setStyleSheet("color: #98c379; font-weight: 600;")
            return

        item = self._batch_queue[self._current_batch_index]
        self.input_edit.setText(item["file"])
        self.output_edit.setText(item["output"])

        # Update table status
        status_item = self.batch_table.item(self._current_batch_index, 3)
        if status_item:
            status_item.setText("Rendering...")
            status_item.setForeground(QColor("#61afef"))

        self.start_render()

    def _on_batch_item_finished(self, success: bool, out_path: str = "", error_msg: str = "") -> None:
        if 0 <= self._current_batch_index < self.batch_table.rowCount():
            status_item = self.batch_table.item(self._current_batch_index, 3)
            if status_item:
                if success:
                    status_item.setText("Completed")
                    status_item.setForeground(QColor("#98c379"))
                else:
                    status_item.setText("Failed")
                    status_item.setForeground(QColor("#e06c75"))

        # Advance to next item
        self._process_next_batch_item()
