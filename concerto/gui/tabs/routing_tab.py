"""Concerto GUI - Speaker Routing & Calibration Tone Generator Tab.

Provides interactive channel routing verification, calibration tone generation,
and visual 5.1 / 7.1 speaker geometry testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets.audio_player import AudioPlayer
from ..widgets.speaker_layout import SurroundLayoutWidget
from ..workers import RoutingTestWorker


class RoutingTab(QWidget):
    """Speaker Routing & Calibration Tone Studio."""

    CHANNEL_CONFIG = [
        ("FL", "Front Left", 1000.0, "#61afef"),
        ("FR", "Front Right", 2000.0, "#61afef"),
        ("FC", "Front Centre", 3000.0, "#98c379"),
        ("LFE", "Low Frequency Effects", 60.0, "#e5c07b"),
        ("SL", "Side Left", 4000.0, "#c678dd"),
        ("SR", "Side Right", 5000.0, "#c678dd"),
        ("BL", "Back Left", 6000.0, "#e06c75"),
        ("BR", "Back Right", 7000.0, "#e06c75"),
    ]

    SWEEP_ORDER = ["FL", "FC", "FR", "SR", "BR", "BL", "SL", "LFE"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: RoutingTestWorker | None = None
        self._audio_player = AudioPlayer(self)
        self._is_sweeping = False
        self._sweep_index = 0
        self._sweep_timer = QTimer(self)
        self._sweep_timer.setInterval(750)
        self._sweep_timer.timeout.connect(self._sweep_step)

        self._setup_ui()
        self._connect_signals()

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

        # Top Section: Two-Column Split (Left: Generator Controls, Right: Speaker Diagram)
        top_split = QHBoxLayout()
        top_split.setContentsMargins(0, 0, 0, 0)
        top_split.setSpacing(14)

        # 1. Generator Controls Card
        ctrl_box = QGroupBox("Calibration Signal Generator & Routing Verification")
        ctrl_layout = QVBoxLayout(ctrl_box)
        ctrl_layout.setContentsMargins(14, 16, 14, 14)
        ctrl_layout.setSpacing(10)

        # Output file path picker
        ctrl_layout.addWidget(QLabel("Output Calibration File:"))
        path_row = QHBoxLayout()
        self.txt_output_path = QLineEdit("routing_test.flac")
        path_row.addWidget(self.txt_output_path, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #abb2bf;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3e4451;
                color: #ffffff;
            }
            """
        )
        self.btn_browse.clicked.connect(self._browse_output)
        path_row.addWidget(self.btn_browse)
        ctrl_layout.addLayout(path_row)

        # Tone duration slider
        dur_header = QHBoxLayout()
        dur_header.addWidget(QLabel("Tone Duration Per Channel:"))
        self.lbl_duration = QLabel("1.0 s")
        self.lbl_duration.setStyleSheet("font-weight: bold; color: #61afef;")
        dur_header.addWidget(self.lbl_duration)
        ctrl_layout.addLayout(dur_header)

        self.slider_duration = QSlider(Qt.Orientation.Horizontal)
        self.slider_duration.setRange(5, 50)  # 0.5s to 5.0s
        self.slider_duration.setValue(10)     # 1.0s
        self.slider_duration.valueChanged.connect(self._on_duration_changed)
        ctrl_layout.addWidget(self.slider_duration)

        # Overall Status Badge
        self.lbl_overall_status = QLabel("ROUTING STATUS: NOT VERIFIED")
        self.lbl_overall_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_overall_status.setFixedHeight(28)
        self.lbl_overall_status.setStyleSheet(
            """
            background-color: #21252b;
            color: #5c6370;
            border: 1px solid #2d3139;
            border-radius: 4px;
            font-weight: 800;
            font-size: 11px;
            """
        )
        ctrl_layout.addWidget(self.lbl_overall_status)

        # Action Buttons
        self.btn_generate = QPushButton("⚡ Generate & Verify Test File")
        self.btn_generate.setFixedHeight(34)
        self.btn_generate.setStyleSheet(
            """
            QPushButton {
                background-color: #61afef;
                color: #181a1f;
                font-weight: 700;
                font-size: 12px;
                border-radius: 4px;
                padding: 6px;
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
        self.btn_generate.clicked.connect(self._run_generation)
        ctrl_layout.addWidget(self.btn_generate)

        sweep_row = QHBoxLayout()
        self.btn_sweep = QPushButton("🔄 Auto Sweep Channels")
        self.btn_sweep.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #98c379;
                border: 1px solid #98c379;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #283c2e;
            }
            """
        )
        self.btn_sweep.clicked.connect(self._toggle_sweep)
        sweep_row.addWidget(self.btn_sweep)

        self.btn_stop = QPushButton("■ Stop Playback")
        self.btn_stop.setStyleSheet(
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
        self.btn_stop.clicked.connect(self._stop_all_playback)
        sweep_row.addWidget(self.btn_stop)
        ctrl_layout.addLayout(sweep_row)

        ctrl_layout.addStretch()
        top_split.addWidget(ctrl_box, 1)

        # 2. Surround Speaker Layout Card
        layout_box = QGroupBox("Interactive 7.1 Speaker Geometry (Click Speaker to Audition)")
        layout_box_layout = QVBoxLayout(layout_box)
        layout_box_layout.setContentsMargins(14, 16, 14, 14)

        self.surround_layout_widget = SurroundLayoutWidget()
        self.surround_layout_widget.setMinimumHeight(280)
        self.surround_layout_widget.speaker_clicked.connect(self._on_speaker_clicked)
        layout_box_layout.addWidget(self.surround_layout_widget)

        top_split.addWidget(layout_box, 1)
        layout.addLayout(top_split)

        # Bottom Section: Channel Routing Table
        table_box = QGroupBox("Channel Routing Assignments & Signature Frequencies")
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(14, 16, 14, 14)
        table_layout.setSpacing(8)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Channel", "Speaker Position", "Signature Tone", "Detected Freq", "Routing Status", "Interactive Test"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setRowCount(len(self.CHANNEL_CONFIG))
        self.table.setFixedHeight(270)
        self.table.setStyleSheet(
            """
            QTableWidget {
                background-color: #181a1f;
                border: 1px solid #2d3139;
                gridline-color: #21252b;
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

        self.test_buttons: dict[str, QPushButton] = {}

        for row, (ch, name, freq, color) in enumerate(self.CHANNEL_CONFIG):
            # 0. Channel
            item_ch = QTableWidgetItem(ch)
            item_ch.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_ch.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            item_ch.setForeground(QColor(color))
            self.table.setItem(row, 0, item_ch)

            # 1. Position Name
            item_name = QTableWidgetItem(name)
            item_name.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table.setItem(row, 1, item_name)

            # 2. Signature Tone
            item_sig = QTableWidgetItem(f"{freq:.0f} Hz" if freq >= 100 else f"{freq:.1f} Hz")
            item_sig.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_sig.setFont(QFont("Courier New", 10))
            self.table.setItem(row, 2, item_sig)

            # 3. Detected Freq
            item_det = QTableWidgetItem("--")
            item_det.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_det.setFont(QFont("Courier New", 10))
            self.table.setItem(row, 3, item_det)

            # 4. Routing Status Badge
            badge = QLabel("PENDING")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                """
                background-color: #3e4451;
                color: #abb2bf;
                font-weight: bold;
                font-size: 10px;
                border-radius: 4px;
                margin: 4px;
                """
            )
            self.table.setCellWidget(row, 4, badge)

            # 5. Interactive Test Button
            btn_test = QPushButton("▶ Play Tone")
            btn_test.setStyleSheet(
                """
                QPushButton {
                    background-color: #2c313a;
                    color: #abb2bf;
                    border: 1px solid #3e4451;
                    border-radius: 4px;
                    padding: 2px 8px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #3e4451;
                    color: #ffffff;
                }
                """
            )
            # Use default argument capture in lambda
            btn_test.clicked.connect(lambda checked=False, ch_name=ch: self._test_single_channel(ch_name))
            self.test_buttons[ch] = btn_test
            self.table.setCellWidget(row, 5, btn_test)

        table_layout.addWidget(self.table)
        layout.addWidget(table_box)
        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _connect_signals(self) -> None:
        self._audio_player.error_signal.connect(self._on_player_error)

    def _on_duration_changed(self, val: int) -> None:
        sec = val / 10.0
        self.lbl_duration.setText(f"{sec:.1f} s")

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration Audio File",
            self.txt_output_path.text(),
            "FLAC Audio (*.flac);;All Files (*.*)",
        )
        if path:
            self.txt_output_path.setText(path)

    def _run_generation(self) -> None:
        out_path = self.txt_output_path.text().strip()
        if not out_path:
            QMessageBox.warning(self, "Invalid Path", "Please provide a valid output file path.")
            return

        self._stop_all_playback()
        self.btn_generate.setEnabled(False)
        self.lbl_overall_status.setText("GENERATING & VERIFYING...")
        self.lbl_overall_status.setStyleSheet(
            """
            background-color: #21252b;
            color: #61afef;
            border: 1px solid #61afef;
            border-radius: 4px;
            font-weight: 800;
            font-size: 11px;
            """
        )

        sec = self.slider_duration.value() / 10.0
        self._worker = RoutingTestWorker(
            output_path=out_path,
            seconds=sec,
            parent=self,
        )
        self._worker.finished_signal.connect(self._on_worker_finished)
        self._worker.error_signal.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_finished(self, output_path: str, all_passed: bool, checks: list) -> None:
        self.btn_generate.setEnabled(True)

        if all_passed:
            self.lbl_overall_status.setText("ROUTING STATUS: ALL CHANNELS VERIFIED ✓")
            self.lbl_overall_status.setStyleSheet(
                """
                background-color: #283c2e;
                color: #98c379;
                border: 1px solid #98c379;
                border-radius: 4px;
                font-weight: 800;
                font-size: 11px;
                """
            )
        else:
            self.lbl_overall_status.setText("ROUTING STATUS: ROUTING MISMATCH DETECTED ✗")
            self.lbl_overall_status.setStyleSheet(
                """
                background-color: #3e262c;
                color: #e06c75;
                border: 1px solid #e06c75;
                border-radius: 4px;
                font-weight: 800;
                font-size: 11px;
                """
            )

        # Update table rows
        check_dict = {c["channel"]: c for c in checks}
        for row, (ch, _, _, _) in enumerate(self.CHANNEL_CONFIG):
            chk = check_dict.get(ch)
            if chk:
                det_hz = chk["detected_hz"]
                passed = chk["passed"]

                item_det = self.table.item(row, 3)
                if item_det:
                    item_det.setText(f"{det_hz:.1f} Hz")

                badge = self.table.cellWidget(row, 4)
                if isinstance(badge, QLabel):
                    if passed:
                        badge.setText("PASS")
                        badge.setStyleSheet(
                            """
                            background-color: #283c2e;
                            color: #98c379;
                            border: 1px solid #98c379;
                            font-weight: bold;
                            font-size: 10px;
                            border-radius: 4px;
                            margin: 4px;
                            """
                        )
                    else:
                        badge.setText("FAIL")
                        badge.setStyleSheet(
                            """
                            background-color: #3e262c;
                            color: #e06c75;
                            border: 1px solid #e06c75;
                            font-weight: bold;
                            font-size: 10px;
                            border-radius: 4px;
                            margin: 4px;
                            """
                        )

    def _on_worker_error(self, message: str) -> None:
        self.btn_generate.setEnabled(True)
        self.lbl_overall_status.setText("ROUTING STATUS: GENERATION ERROR")
        self.lbl_overall_status.setStyleSheet(
            """
            background-color: #3e262c;
            color: #e06c75;
            border: 1px solid #e06c75;
            border-radius: 4px;
            font-weight: 800;
            font-size: 11px;
            """
        )
        QMessageBox.critical(self, "Routing Test Error", message)

    def _on_speaker_clicked(self, channel: str) -> None:
        self._test_single_channel(channel)

    def _test_single_channel(self, channel: str) -> None:
        # Find frequency
        freq = 1000.0
        for ch, _, f, _ in self.CHANNEL_CONFIG:
            if ch == channel:
                freq = f
                break

        self._play_tone_for_channel(channel, freq, duration_s=0.6)

    def _play_tone_for_channel(self, channel: str, freq: float, duration_s: float = 0.6) -> None:
        self._audio_player.stop()

        # Highlight in surround layout widget
        self.surround_layout_widget.clear_levels()
        self.surround_layout_widget.set_selected_speaker(channel)
        self.surround_layout_widget.set_channel_level(channel, 1.0)

        # Highlight in table
        for row, (ch, _, _, _) in enumerate(self.CHANNEL_CONFIG):
            if ch == channel:
                self.table.selectRow(row)
                break

        # Synthesize tone
        sr = 48000
        n_samples = int(sr * duration_s)
        t = np.arange(n_samples) / sr
        amp = 0.5 if channel != "LFE" else 0.8
        tone = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

        # Apply short smooth envelope to avoid clicks
        fade_len = min(int(sr * 0.02), n_samples // 4)
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            tone[:fade_len] *= fade_in
            tone[-fade_len:] *= fade_out

        # If auditioning through headphones (fold-down/stereo):
        # We can construct stereo channels folded down or direct
        # Channel layout mapping for 7.1: FL=0, FR=1, FC=2, LFE=3, SL=4, SR=5, BL=6, BR=7
        ch_idx_map = {"FL": 0, "FR": 1, "FC": 2, "LFE": 3, "SL": 4, "SR": 5, "BL": 6, "BR": 7}
        idx = ch_idx_map.get(channel, 0)

        chans = np.zeros((8, n_samples), dtype=np.float32)
        chans[idx] = tone

        # If LFE is played on headphones, standard fold-down ignores LFE.
        # Add audible reference to both channels so listener hears the 60 Hz tone:
        if channel == "LFE":
            chans[0] = tone * 0.5
            chans[1] = tone * 0.5

        self._audio_player.load_audio(chans, sample_rate=sr, layout_name="7.1")
        self._audio_player.set_audition_mode("fold_down")
        self._audio_player.play()

        # Schedule clearing the halo after duration
        QTimer.singleShot(int(duration_s * 1000) + 50, self._on_tone_finished)

    def _on_tone_finished(self) -> None:
        if not self._is_sweeping:
            self.surround_layout_widget.clear_levels()

    def _toggle_sweep(self) -> None:
        if self._is_sweeping:
            self._stop_sweep()
        else:
            self._start_sweep()

    def _start_sweep(self) -> None:
        self._is_sweeping = True
        self._sweep_index = 0
        self.btn_sweep.setText("■ Stop Sweep")
        self.btn_sweep.setStyleSheet(
            """
            QPushButton {
                background-color: #3e262c;
                color: #e06c75;
                border: 1px solid #e06c75;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            """
        )
        self._sweep_step()
        self._sweep_timer.start()

    def _sweep_step(self) -> None:
        if not self._is_sweeping:
            return

        if self._sweep_index >= len(self.SWEEP_ORDER):
            self._sweep_index = 0

        ch = self.SWEEP_ORDER[self._sweep_index]
        self._sweep_index += 1

        freq = 1000.0
        for name, _, f, _ in self.CHANNEL_CONFIG:
            if name == ch:
                freq = f
                break

        self._play_tone_for_channel(ch, freq, duration_s=0.6)

    def _stop_sweep(self) -> None:
        self._is_sweeping = False
        self._sweep_timer.stop()
        self.btn_sweep.setText("🔄 Auto Sweep Channels")
        self.btn_sweep.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #98c379;
                border: 1px solid #98c379;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #283c2e;
            }
            """
        )
        self.surround_layout_widget.clear_levels()
        self.table.clearSelection()

    def _stop_all_playback(self) -> None:
        self._stop_sweep()
        self._audio_player.stop()
        self.surround_layout_widget.clear_levels()
        self.table.clearSelection()

    def _on_player_error(self, message: str) -> None:
        self._stop_all_playback()
        QMessageBox.warning(self, "Audio Playback Notice", message)
