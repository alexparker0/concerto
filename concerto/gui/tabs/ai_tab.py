"""Concerto GUI - AI Stem Separation and Acceptance Lab Tab.

Provides an interactive laboratory for evaluating neural stem separation
and executing the 4-point acceptance harness (SPEC.md Section 7)
independently of rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..widgets.audio_player import AudioPlayer
from ..widgets.plots import StemEnergyWidget
from ..workers import AiEvalWorker


class AcceptanceCard(QFrame):
    """Visual card displaying a single acceptance criterion."""

    def __init__(
        self,
        title: str,
        target_rule: str,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            AcceptanceCard {
                background-color: #21252b;
                border: 1px solid #2d3139;
                border-radius: 6px;
                padding: 10px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: 700; color: #d7dae0; font-size: 13px;")
        header_layout.addWidget(self.lbl_title)

        header_layout.addStretch()

        self.badge_status = QLabel("PENDING")
        self.badge_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_status.setFixedSize(70, 20)
        self.badge_status.setStyleSheet(
            """
            background-color: #3e4451;
            color: #abb2bf;
            font-weight: bold;
            font-size: 10px;
            border-radius: 4px;
            """
        )
        header_layout.addWidget(self.badge_status)
        layout.addLayout(header_layout)

        self.lbl_rule = QLabel(f"Target: {target_rule}")
        self.lbl_rule.setStyleSheet("color: #61afef; font-size: 11px;")
        layout.addWidget(self.lbl_rule)

        self.lbl_measured = QLabel("Measured: --")
        self.lbl_measured.setStyleSheet("color: #abb2bf; font-size: 12px; font-weight: 600;")
        layout.addWidget(self.lbl_measured)

        self.lbl_desc = QLabel(description)
        self.lbl_desc.setStyleSheet("color: #5c6370; font-size: 10px;")
        self.lbl_desc.setWordWrap(True)
        layout.addWidget(self.lbl_desc)

    def set_result(self, passed: bool | None, measured_text: str) -> None:
        self.lbl_measured.setText(f"Measured: {measured_text}")
        if passed is True:
            self.badge_status.setText("PASS")
            self.badge_status.setStyleSheet(
                """
                background-color: #283c2e;
                color: #98c379;
                font-weight: bold;
                font-size: 10px;
                border-radius: 4px;
                border: 1px solid #98c379;
                """
            )
        elif passed is False:
            self.badge_status.setText("FAIL")
            self.badge_status.setStyleSheet(
                """
                background-color: #3e262c;
                color: #e06c75;
                font-weight: bold;
                font-size: 10px;
                border-radius: 4px;
                border: 1px solid #e06c75;
                """
            )
        else:
            self.badge_status.setText("PENDING")
            self.badge_status.setStyleSheet(
                """
                background-color: #3e4451;
                color: #abb2bf;
                font-weight: bold;
                font-size: 10px;
                border-radius: 4px;
                """
            )

    def reset(self) -> None:
        self.set_result(None, "--")


class StemRowWidget(QFrame):
    """Single row showing stem metrics and dedicated audition button."""

    def __init__(
        self,
        stem_name: str,
        color_hex: str,
        on_audition_callback: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stem_name = stem_name
        self.color_hex = color_hex
        self.on_audition = on_audition_callback
        self.is_playing = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"""
            StemRowWidget {{
                background-color: #1e2227;
                border: 1px solid #2d3139;
                border-left: 4px solid {color_hex};
                border-radius: 4px;
                padding: 4px;
            }}
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        self.lbl_name = QLabel(stem_name)
        self.lbl_name.setFixedWidth(110)
        self.lbl_name.setStyleSheet(f"font-weight: bold; color: {color_hex}; font-size: 12px;")
        layout.addWidget(self.lbl_name)

        self.lbl_rms = QLabel("RMS: -- dB")
        self.lbl_rms.setFixedWidth(90)
        self.lbl_rms.setStyleSheet("color: #abb2bf; font-size: 11px;")
        layout.addWidget(self.lbl_rms)

        self.lbl_peak = QLabel("Peak: -- dB")
        self.lbl_peak.setFixedWidth(90)
        self.lbl_peak.setStyleSheet("color: #abb2bf; font-size: 11px;")
        layout.addWidget(self.lbl_peak)

        self.bar_level = QProgressBar()
        self.bar_level.setRange(0, 100)
        self.bar_level.setValue(0)
        self.bar_level.setTextVisible(False)
        self.bar_level.setFixedHeight(12)
        self.bar_level.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: #181a1f;
                border: 1px solid #2d3139;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {color_hex};
                border-radius: 2px;
            }}
            """
        )
        layout.addWidget(self.bar_level, 1)

        self.btn_audition = QPushButton("▶ Audition")
        self.btn_audition.setFixedWidth(100)
        self.btn_audition.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #abb2bf;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3e4451;
                color: #ffffff;
            }
            """
        )
        self.btn_audition.clicked.connect(self._handle_click)
        layout.addWidget(self.btn_audition)

    def _handle_click(self) -> None:
        if self.on_audition:
            self.on_audition(self.stem_name)

    def set_active_playback(self, playing: bool) -> None:
        self.is_playing = playing
        if playing:
            self.btn_audition.setText("■ Stop")
            self.btn_audition.setStyleSheet(
                """
                QPushButton {
                    background-color: #98c379;
                    color: #181a1f;
                    border: 1px solid #98c379;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #88b369;
                }
                """
            )
        else:
            self.btn_audition.setText("▶ Audition")
            self.btn_audition.setStyleSheet(
                """
                QPushButton {
                    background-color: #2c313a;
                    color: #abb2bf;
                    border: 1px solid #3e4451;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #3e4451;
                    color: #ffffff;
                }
                """
            )

    def update_metrics(self, rms_lin: float, peak_lin: float) -> None:
        rms_db = 20.0 * np.log10(max(rms_lin, 1e-6))
        peak_db = 20.0 * np.log10(max(peak_lin, 1e-6))
        self.lbl_rms.setText(f"RMS: {rms_db:.1f} dB")
        self.lbl_peak.setText(f"Peak: {peak_db:.1f} dB")
        # Level bar mapping [-60 dB, 0 dB] -> [0, 100]
        bar_val = int(np.clip((rms_db + 60.0) / 60.0 * 100.0, 0, 100))
        self.bar_level.setValue(bar_val)

    def reset(self) -> None:
        self.lbl_rms.setText("RMS: -- dB")
        self.lbl_peak.setText("Peak: -- dB")
        self.bar_level.setValue(0)
        self.set_active_playback(False)


class AiTab(QWidget):
    """Dedicated laboratory for AI stem separation and acceptance harness."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: AiEvalWorker | None = None
        self._audio_player = AudioPlayer(self)
        self._stems_obj: Any = None
        self._raw_report: dict[str, Any] | None = None
        self._source_audio: tuple[np.ndarray, np.ndarray, int] | None = None
        self._currently_playing_stem: str | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Scroll area for clean workstation sizing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(14)

        # 1. Source and Model Card
        src_box = QGroupBox("Neural Stem Separation & Model Configuration")
        src_layout = QGridLayout(src_box)
        src_layout.setContentsMargins(14, 16, 14, 14)
        src_layout.setHorizontalSpacing(10)
        src_layout.setVerticalSpacing(8)

        # Input audio selector
        src_layout.addWidget(QLabel("Input Audio File:"), 0, 0)
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Select stereo FLAC, WAV, AIFF, MP3, Opus...")
        src_layout.addWidget(self.txt_input, 0, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #abb2bf;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3e4451;
                color: #ffffff;
            }
            """
        )
        self.btn_browse.clicked.connect(self._browse_input)
        src_layout.addWidget(self.btn_browse, 0, 2)

        # Model Selector
        src_layout.addWidget(QLabel("Separation Model:"), 1, 0)
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(8)

        from PySide6.QtWidgets import QComboBox

        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "htdemucs (Demucs Hybrid Transformer - PyTorch)",
        ])
        model_row.addWidget(self.combo_model, 1)
        self.txt_custom_model = QLineEdit()
        self.txt_custom_model.setVisible(False)

        src_layout.addLayout(model_row, 1, 1, 1, 2)

        # Cache bypass & Action
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(12)

        from PySide6.QtWidgets import QCheckBox

        self.chk_no_cache = QCheckBox("Bypass stem cache (--no-cache)")
        self.chk_no_cache.setToolTip("Force re-running neural inference even if cached stems exist.")
        action_row.addWidget(self.chk_no_cache)

        action_row.addStretch()

        self.btn_run = QPushButton("⚡ Run Separation & Harness")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(
            """
            QPushButton {
                background-color: #61afef;
                color: #181a1f;
                font-weight: 700;
                font-size: 13px;
                border-radius: 4px;
                padding: 6px 18px;
            }
            QPushButton:hover {
                background-color: #529fd8;
            }
            QPushButton:disabled {
                background-color: #3e4451;
                color: #5c6370;
            }
            """
        )
        self.btn_run.clicked.connect(self._start_evaluation)
        action_row.addWidget(self.btn_run)

        src_layout.addLayout(action_row, 2, 0, 1, 3)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #181a1f;
                border: 1px solid #2d3139;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #61afef;
                border-radius: 3px;
            }
            """
        )
        src_layout.addWidget(self.progress_bar, 3, 0, 1, 3)

        self.lbl_status = QLabel("Ready to evaluate stem separation.")
        self.lbl_status.setStyleSheet("color: #5c6370; font-size: 11px;")
        src_layout.addWidget(self.lbl_status, 4, 0, 1, 3)

        layout.addWidget(src_box)

        # 2. 4-Point Acceptance Dashboard Card
        harness_box = QGroupBox("4-Point AI Acceptance Dashboard (SPEC.md Section 7)")
        harness_layout = QVBoxLayout(harness_box)
        harness_layout.setContentsMargins(14, 16, 14, 14)
        harness_layout.setSpacing(10)

        # Overall Status Banner
        banner_layout = QHBoxLayout()
        banner_layout.setContentsMargins(0, 0, 0, 0)

        self.banner_overall = QLabel("OVERALL STATUS: NOT RUN")
        self.banner_overall.setFixedHeight(30)
        self.banner_overall.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_overall.setStyleSheet(
            """
            background-color: #21252b;
            color: #5c6370;
            border: 1px solid #2d3139;
            border-radius: 4px;
            font-weight: 800;
            font-size: 12px;
            padding: 4px 12px;
            """
        )
        banner_layout.addWidget(self.banner_overall, 1)

        self.btn_json_report = QPushButton("View Raw JSON Report")
        self.btn_json_report.setEnabled(False)
        self.btn_json_report.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #abb2bf;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3e4451;
                color: #ffffff;
            }
            """
        )
        self.btn_json_report.clicked.connect(self._show_raw_json)
        banner_layout.addWidget(self.btn_json_report)
        harness_layout.addLayout(banner_layout)

        # Grid of 4 Acceptance Cards
        cards_grid = QGridLayout()
        cards_grid.setContentsMargins(0, 4, 0, 4)
        cards_grid.setHorizontalSpacing(10)
        cards_grid.setVerticalSpacing(10)

        self.card_align = AcceptanceCard(
            title="1. Timeline Alignment",
            target_rule="max lag ≤ 4 samples",
            description="Evaluates cross-correlation lag of stem sum vs original stereo mix.",
        )
        cards_grid.addWidget(self.card_align, 0, 0)

        self.card_recomb = AcceptanceCard(
            title="2. Recombination Residual SDR",
            target_rule="residual ≥ 20.0 dB below mix",
            description="Verifies that stem sum reconstructs the original mix without phase cancellation.",
        )
        cards_grid.addWidget(self.card_recomb, 0, 1)

        self.card_artefact = AcceptanceCard(
            title="3. Transient Artefacts",
            target_rule="max envelope jump ≤ 3.0",
            description="Detects segmentation glitches or frame-to-frame unnatural envelope spikes.",
        )
        cards_grid.addWidget(self.card_artefact, 1, 0)

        self.card_stereo = AcceptanceCard(
            title="4. Stereo Interpretability",
            target_rule="L/R correlation floor ≥ -0.95",
            description="Prevents unnatural anti-phase acoustic energy across decomposed stems.",
        )
        cards_grid.addWidget(self.card_stereo, 1, 1)

        harness_layout.addLayout(cards_grid)
        layout.addWidget(harness_box)

        # 3. Stem Visualizer & Preview Card
        stems_box = QGroupBox("Separated Stem Energy & Auditioning")
        stems_layout = QVBoxLayout(stems_box)
        stems_layout.setContentsMargins(14, 16, 14, 14)
        stems_layout.setSpacing(12)

        # Top row: StemEnergyWidget + Overview info
        top_stems_row = QHBoxLayout()
        top_stems_row.setContentsMargins(0, 0, 0, 0)
        top_stems_row.setSpacing(14)

        self.stem_energy_widget = StemEnergyWidget()
        self.stem_energy_widget.setMinimumHeight(150)
        top_stems_row.addWidget(self.stem_energy_widget, 1)

        # Global transport control for preview
        transport_box = QFrame()
        transport_box.setStyleSheet(
            """
            QFrame {
                background-color: #1e2227;
                border: 1px solid #2d3139;
                border-radius: 6px;
                padding: 10px;
            }
            """
        )
        trans_layout = QVBoxLayout(transport_box)
        trans_layout.setContentsMargins(10, 8, 10, 8)
        trans_layout.setSpacing(8)

        lbl_trans = QLabel("Preview Audition Controls")
        lbl_trans.setStyleSheet("font-weight: bold; color: #abb2bf; font-size: 11px;")
        trans_layout.addWidget(lbl_trans)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume:"))
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(80)
        self.slider_vol.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.slider_vol)
        self.lbl_vol = QLabel("80%")
        vol_row.addWidget(self.lbl_vol)
        trans_layout.addLayout(vol_row)

        self.btn_stop_all = QPushButton("■ Stop Audition")
        self.btn_stop_all.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #e06c75;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3e262c;
                border-color: #e06c75;
            }
            """
        )
        self.btn_stop_all.clicked.connect(self._stop_audition)
        trans_layout.addWidget(self.btn_stop_all)
        trans_layout.addStretch()

        top_stems_row.addWidget(transport_box, 1)
        stems_layout.addLayout(top_stems_row)

        # Rows for each stem
        self.stem_rows: dict[str, StemRowWidget] = {}
        stems_meta = [
            ("Vocals", "#61afef"),
            ("Drums", "#e06c75"),
            ("Bass", "#c678dd"),
            ("Other", "#98c379"),
            ("Residual", "#e5c07b"),
        ]
        for name, color in stems_meta:
            row = StemRowWidget(name, color, self._toggle_stem_audition)
            self.stem_rows[name] = row
            stems_layout.addWidget(row)

        layout.addWidget(stems_box)
        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _connect_signals(self) -> None:
        self._audio_player.state_changed.connect(self._on_player_state_changed)
        self._audio_player.error_signal.connect(self._on_player_error)

    def _on_model_changed(self, index: int) -> None:
        pass

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File for AI Separation",
            "",
            "Audio Files (*.flac *.wav *.aiff *.aif *.mp3 *.ogg *.opus);;All Files (*.*)",
        )
        if path:
            self.txt_input.setText(path)
    def _start_evaluation(self) -> None:
        input_path = self.txt_input.text().strip()
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(self, "Invalid Input", "Please select a valid audio file.")
            return

        # Stop any active auditioning
        self._stop_audition()

        onnx_model: str | None = None
        no_cache = self.chk_no_cache.isChecked()

        # Update UI state
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting AI stem separation worker...")
        self.banner_overall.setText("OVERALL STATUS: RUNNING...")
        self.banner_overall.setStyleSheet(
            """
            background-color: #21252b;
            color: #61afef;
            border: 1px solid #61afef;
            border-radius: 4px;
            font-weight: 800;
            font-size: 12px;
            padding: 4px 12px;
            """
        )
        self.card_align.reset()
        self.card_recomb.reset()
        self.card_artefact.reset()
        self.card_stereo.reset()
        for r in self.stem_rows.values():
            r.reset()
        self.stem_energy_widget.clear()

        # Launch worker
        self._worker = AiEvalWorker(
            input_path=input_path,
            onnx_model=onnx_model,
            no_cache=no_cache,
            parent=self,
        )
        self._worker.progress_signal.connect(self._on_worker_progress)
        self._worker.finished_signal.connect(self._on_worker_finished)
        self._worker.error_signal.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_progress(self, current: int, total: int, stage: str) -> None:
        pct = int(current / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(stage)

    def _on_worker_finished(self, report_dict: dict, stems: Any) -> None:
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.btn_json_report.setEnabled(True)
        self._raw_report = report_dict
        self._stems_obj = stems

        # Decode original stereo file for residual playback reference
        try:
            from concerto.audio.decode import decode_stereo

            buf, meta = decode_stereo(self.txt_input.text().strip())
            self._source_audio = (buf.left, buf.right, buf.sample_rate)
        except Exception:
            self._source_audio = None

        passed = report_dict.get("passed", False)
        if passed:
            self.banner_overall.setText("OVERALL STATUS: PASSED ✓")
            self.banner_overall.setStyleSheet(
                """
                background-color: #283c2e;
                color: #98c379;
                border: 1px solid #98c379;
                border-radius: 4px;
                font-weight: 800;
                font-size: 12px;
                padding: 4px 12px;
                """
            )
        else:
            self.banner_overall.setText("OVERALL STATUS: FAILED ✗")
            self.banner_overall.setStyleSheet(
                """
                background-color: #3e262c;
                color: #e06c75;
                border: 1px solid #e06c75;
                border-radius: 4px;
                font-weight: 800;
                font-size: 12px;
                padding: 4px 12px;
                """
            )

        self.lbl_status.setText("Stem evaluation completed.")

        # Update 4-Point Cards
        metrics = report_dict.get("metrics", {})
        checks_map = {c.get("check"): c for c in report_dict.get("checks", [])}

        # 1. Timeline alignment
        chk_align = checks_map.get("alignment", {})
        lag = metrics.get("alignment_lag_samples", 0)
        self.card_align.set_result(
            chk_align.get("passed", False),
            f"lag = {lag} samples",
        )

        # 2. Recombination
        chk_recomb = checks_map.get("recombination", {})
        recomb_db = metrics.get("recombination_db", 0.0)
        self.card_recomb.set_result(
            chk_recomb.get("passed", False),
            f"residual {recomb_db:.1f} dB below mix",
        )

        # 3. Artefacts
        chk_art = checks_map.get("transient_artefacts", {})
        jump = metrics.get("envelope_jump", 0.0)
        self.card_artefact.set_result(
            chk_art.get("passed", False),
            f"envelope jump = {jump:.2f}",
        )

        # 4. Stereo
        stem_keys = ["vocals", "drums", "bass", "other"]
        stereo_passes = [checks_map.get(f"stereo_{k}", {}).get("passed", True) for k in stem_keys]
        all_stereo_ok = all(stereo_passes)
        stereo_readouts = []
        for k in stem_keys:
            val = metrics.get(f"stereo_corr_{k}", None)
            if val is not None:
                stereo_readouts.append(f"{k[0].upper()}:{val:.2f}")
        self.card_stereo.set_result(
            all_stereo_ok,
            ", ".join(stereo_readouts) if stereo_readouts else "all checks valid",
        )

        # Update stem energy meters and stem row readouts
        if stems is not None:
            v_rms, v_pk = self._calc_rms_peak(stems.vocals[0], stems.vocals[1])
            d_rms, d_pk = self._calc_rms_peak(stems.drums[0], stems.drums[1])
            b_rms, b_pk = self._calc_rms_peak(stems.bass[0], stems.bass[1])
            o_rms, o_pk = self._calc_rms_peak(stems.other[0], stems.other[1])

            self.stem_rows["Vocals"].update_metrics(v_rms, v_pk)
            self.stem_rows["Drums"].update_metrics(d_rms, d_pk)
            self.stem_rows["Bass"].update_metrics(b_rms, b_pk)
            self.stem_rows["Other"].update_metrics(o_rms, o_pk)

            self.stem_energy_widget.set_levels(v_rms, d_rms, b_rms, o_rms)

            # Calculate residual metrics
            if self._source_audio is not None:
                mix_l, mix_r, _ = self._source_audio
                n_min = min(len(mix_l), len(stems.vocals[0]))
                sum_l = stems.vocals[0][:n_min] + stems.drums[0][:n_min] + stems.bass[0][:n_min] + stems.other[0][:n_min]
                sum_r = stems.vocals[1][:n_min] + stems.drums[1][:n_min] + stems.bass[1][:n_min] + stems.other[1][:n_min]
                res_l = mix_l[:n_min] - sum_l
                res_r = mix_r[:n_min] - sum_r
                r_rms, r_pk = self._calc_rms_peak(res_l, res_r)
                self.stem_rows["Residual"].update_metrics(r_rms, r_pk)

    def _calc_rms_peak(self, left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
        if len(left) == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(0.5 * (np.mean(left ** 2) + np.mean(right ** 2))))
        pk = float(max(np.max(np.abs(left)), np.max(np.abs(right))))
        return rms, pk

    def _on_worker_error(self, message: str) -> None:
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Error: {message}")
        self.banner_overall.setText("OVERALL STATUS: ERROR")
        self.banner_overall.setStyleSheet(
            """
            background-color: #3e262c;
            color: #e06c75;
            border: 1px solid #e06c75;
            border-radius: 4px;
            font-weight: 800;
            font-size: 12px;
            padding: 4px 12px;
            """
        )
        QMessageBox.critical(self, "AI Evaluation Error", message)

    def _toggle_stem_audition(self, stem_name: str) -> None:
        if self._currently_playing_stem == stem_name and self._audio_player.is_playing:
            self._stop_audition()
            return

        if self._stems_obj is None:
            QMessageBox.information(self, "No Stems", "Please run stem separation first.")
            return

        sr = 44100
        if self._source_audio:
            sr = self._source_audio[2]

        chans: np.ndarray | None = None
        if stem_name == "Vocals":
            chans = np.stack([self._stems_obj.vocals[0], self._stems_obj.vocals[1]], axis=0)
        elif stem_name == "Drums":
            chans = np.stack([self._stems_obj.drums[0], self._stems_obj.drums[1]], axis=0)
        elif stem_name == "Bass":
            chans = np.stack([self._stems_obj.bass[0], self._stems_obj.bass[1]], axis=0)
        elif stem_name == "Other":
            chans = np.stack([self._stems_obj.other[0], self._stems_obj.other[1]], axis=0)
        elif stem_name == "Residual" and self._source_audio:
            mix_l, mix_r, _ = self._source_audio
            n_min = min(len(mix_l), len(self._stems_obj.vocals[0]))
            sum_l = (
                self._stems_obj.vocals[0][:n_min]
                + self._stems_obj.drums[0][:n_min]
                + self._stems_obj.bass[0][:n_min]
                + self._stems_obj.other[0][:n_min]
            )
            sum_r = (
                self._stems_obj.vocals[1][:n_min]
                + self._stems_obj.drums[1][:n_min]
                + self._stems_obj.bass[1][:n_min]
                + self._stems_obj.other[1][:n_min]
            )
            res_l = mix_l[:n_min] - sum_l
            res_r = mix_r[:n_min] - sum_r
            chans = np.stack([res_l, res_r], axis=0)

        if chans is None:
            QMessageBox.warning(self, "Stem Unavailable", f"Cannot play stem: {stem_name}")
            return

        self._audio_player.stop()
        self._audio_player.load_audio(chans, sample_rate=sr, layout_name="stereo")
        self._audio_player.set_audition_mode("stereo")
        self._audio_player.set_volume(self.slider_vol.value() / 100.0)
        self._audio_player.play()

        self._currently_playing_stem = stem_name
        for name, row in self.stem_rows.items():
            row.set_active_playback(name == stem_name)

    def _stop_audition(self) -> None:
        self._audio_player.stop()
        self._currently_playing_stem = None
        for row in self.stem_rows.values():
            row.set_active_playback(False)

    def _on_player_state_changed(self, is_playing: bool) -> None:
        if not is_playing:
            self._currently_playing_stem = None
            for row in self.stem_rows.values():
                row.set_active_playback(False)

    def _on_player_error(self, message: str) -> None:
        self._stop_audition()
        QMessageBox.warning(self, "Audio Playback Error", message)

    def _on_volume_changed(self, val: int) -> None:
        self.lbl_vol.setText(f"{val}%")
        self._audio_player.set_volume(val / 100.0)

    def _show_raw_json(self) -> None:
        if not self._raw_report:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("AI Acceptance Harness - Raw JSON Report")
        dlg.resize(600, 500)
        dlg_layout = QVBoxLayout(dlg)

        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setFont(QFont("Courier New", 10))
        edit.setStyleSheet("background-color: #181a1f; color: #abb2bf; border: 1px solid #2d3139;")
        edit.setText(json.dumps(self._raw_report, indent=2))
        dlg_layout.addWidget(edit)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        dlg_layout.addWidget(btn_close)
        dlg.exec()
