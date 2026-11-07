"""Acceptance & Quality Assurance Suite tab for Concerto GUI.

Features:
- Multichannel file selection (5.1 or 7.1 FLAC/WAV) with drag-and-drop.
- Optional Stereo Reference file input for front preservation, duration, and downmix SDR tests.
- Prominent visual PASS/FAIL status badge and acceptance summary metrics.
- Comprehensive QA checklist table displaying status, check name, target threshold, and measured values.
- Channel routing frequency detector verifying 1 kHz FL, 2 kHz FR, 3 kHz FC, 60 Hz LFE,
  4 kHz SL, 5 kHz SR, 6 kHz BL, 7 kHz BR tones.
- Formatted JSON inspector with one-click clipboard copying and export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QGuiApplication
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...validate.tests import detect_channel_frequencies
from ..workers import ValidationWorker


class ValidationTab(QWidget):
    """Multichannel acceptance test & validation suite tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._active_worker: ValidationWorker | None = None
        self._last_result_json: str = ""

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Scroll area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # 1. Inputs Section (Multichannel + Optional Reference)
        layout.addWidget(self._build_inputs_section())

        # 2. Results Header (Big Status Badge + Quick Stats)
        layout.addWidget(self._build_badge_section())

        # 3. QA Checklist Table
        layout.addWidget(self._build_table_section())

        # 4. Channel Routing Frequency Verification
        layout.addWidget(self._build_routing_section())

        # 5. Raw JSON Report Viewer
        layout.addWidget(self._build_json_section())

        layout.addStretch(1)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # 1. Inputs Section
    # -------------------------------------------------------------------------
    def _build_inputs_section(self) -> QGroupBox:
        box = QGroupBox("Validation Targets")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # Multichannel file row
        multi_row = QHBoxLayout()
        multi_row.setSpacing(8)
        lbl_multi = QLabel("Rendered Multichannel Audio:")
        lbl_multi.setMinimumWidth(180)
        multi_row.addWidget(lbl_multi)

        self.multi_edit = QLineEdit()
        self.multi_edit.setPlaceholderText("Select rendered 5.1 or 7.1 FLAC/WAV file...")
        multi_row.addWidget(self.multi_edit, 1)

        self.btn_browse_multi = QPushButton("Browse...")
        self.btn_browse_multi.clicked.connect(self._browse_multi)
        multi_row.addWidget(self.btn_browse_multi)
        layout.addLayout(multi_row)

        # Optional Stereo Reference file row
        ref_row = QHBoxLayout()
        ref_row.setSpacing(8)
        lbl_ref = QLabel("Original Stereo Reference (Optional):")
        lbl_ref.setMinimumWidth(180)
        ref_row.addWidget(lbl_ref)

        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText("Select original stereo file for bit-preservation, duration, and downmix SDR tests...")
        ref_row.addWidget(self.ref_edit, 1)

        self.btn_browse_ref = QPushButton("Browse...")
        self.btn_browse_ref.clicked.connect(self._browse_ref)
        ref_row.addWidget(self.btn_browse_ref)
        layout.addLayout(ref_row)

        # Action bar
        act_row = QHBoxLayout()
        act_row.setSpacing(10)

        self.btn_validate = QPushButton("Run Validation")
        self.btn_validate.setObjectName("primaryButton")
        self.btn_validate.setStyleSheet("QPushButton { font-weight: 600; font-size: 13px; padding: 6px 20px; }")
        self.btn_validate.clicked.connect(self.run_validation)
        act_row.addWidget(self.btn_validate)

        self.btn_check_routing = QPushButton("Detect Routing Frequencies")
        self.btn_check_routing.clicked.connect(self.check_routing_frequencies)
        act_row.addWidget(self.btn_check_routing)

        act_row.addStretch(1)

        self.lbl_file_info = QLabel("No file loaded")
        self.lbl_file_info.setStyleSheet("color: #7f848e; font-size: 11px;")
        act_row.addWidget(self.lbl_file_info)

        layout.addLayout(act_row)
        return box

    # -------------------------------------------------------------------------
    # 2. Big Visual PASS/FAIL Badge Section
    # -------------------------------------------------------------------------
    def _build_badge_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #21252b; border: 1px solid #2d3139; border-radius: 6px; padding: 12px; }"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        # Badge pill
        self.badge_lbl = QLabel("READY")
        self.badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_lbl.setMinimumSize(180, 52)
        self.badge_lbl.setStyleSheet(
            "QLabel { background-color: #282c34; color: #abb2bf; border-radius: 6px; "
            "font-size: 18px; font-weight: bold; letter-spacing: 2px; padding: 6px 16px; }"
        )
        layout.addWidget(self.badge_lbl)

        # Status summary text
        sum_vbox = QVBoxLayout()
        sum_vbox.setSpacing(3)

        self.lbl_summary_title = QLabel("Acceptance Test Suite")
        self.lbl_summary_title.setStyleSheet("color: #d7dae0; font-size: 14px; font-weight: 600;")
        sum_vbox.addWidget(self.lbl_summary_title)

        self.lbl_summary_desc = QLabel("Load a multichannel audio file and run validation to verify DSP safety criteria.")
        self.lbl_summary_desc.setStyleSheet("color: #abb2bf; font-size: 12px;")
        sum_vbox.addWidget(self.lbl_summary_desc)

        layout.addLayout(sum_vbox, 1)
        return frame

    # -------------------------------------------------------------------------
    # 3. QA Checklist Table
    # -------------------------------------------------------------------------
    def _build_table_section(self) -> QGroupBox:
        box = QGroupBox("Deterministic Acceptance Criteria")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        self.checks_table = QTableWidget(0, 4)
        self.checks_table.setHorizontalHeaderLabels([
            "Status",
            "Verification Check",
            "Target Specification",
            "Observed Result / Measurement",
        ])
        self.checks_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.checks_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.checks_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.checks_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.checks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.checks_table.setAlternatingRowColors(True)
        self.checks_table.setMinimumHeight(180)
        layout.addWidget(self.checks_table)

        return box

    # -------------------------------------------------------------------------
    # 4. Channel Routing Frequencies
    # -------------------------------------------------------------------------
    def _build_routing_section(self) -> QGroupBox:
        self.routing_box = QGroupBox("Channel Routing Frequencies (Calibration Tone Verification)")
        layout = QVBoxLayout(self.routing_box)
        layout.setSpacing(6)

        self.routing_table = QTableWidget(0, 4)
        self.routing_table.setHorizontalHeaderLabels([
            "Channel",
            "Standard Calibration Frequency",
            "Detected Dominant Frequency",
            "Routing Match",
        ])
        self.routing_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.routing_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.routing_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.routing_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.routing_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.routing_table.setAlternatingRowColors(True)
        self.routing_table.setFixedHeight(120)
        layout.addWidget(self.routing_table)

        return self.routing_box

    # -------------------------------------------------------------------------
    # 5. Raw JSON Report Viewer
    # -------------------------------------------------------------------------
    def _build_json_section(self) -> QGroupBox:
        box = QGroupBox("Raw Validation JSON Report")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        # Toolbar
        tb = QHBoxLayout()
        tb.setSpacing(8)

        self.btn_copy_json = QPushButton("Copy JSON")
        self.btn_copy_json.clicked.connect(self._copy_json)
        tb.addWidget(self.btn_copy_json)

        self.btn_save_json = QPushButton("Save JSON Report...")
        self.btn_save_json.clicked.connect(self._save_json)
        tb.addWidget(self.btn_save_json)

        tb.addStretch(1)
        layout.addLayout(tb)

        self.json_view = QTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setFont(QFont("Consolas", 10))
        self.json_view.setStyleSheet(
            "QTextEdit { background-color: #1e2227; color: #abb2bf; border: 1px solid #2d3139; border-radius: 4px; }"
        )
        self.json_view.setFixedHeight(150)
        layout.addWidget(self.json_view)

        return box

    # -------------------------------------------------------------------------
    # File Browsing & Drag & Drop
    # -------------------------------------------------------------------------
    def load_multichannel_file(self, file_path: str) -> None:
        path = Path(file_path).resolve()
        if not path.is_file():
            return
        self.multi_edit.setText(str(path))
        try:
            info = sf.info(str(path))
            self.lbl_file_info.setText(f"{info.channels} ch | {info.samplerate:,} Hz | {info.duration:.1f}s | {info.format}")
        except Exception:
            pass

    def _browse_multi(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Rendered Multichannel File",
            "",
            "Audio Files (*.flac *.wav);;All Files (*.*)",
        )
        if file_path:
            self.load_multichannel_file(file_path)

    def _browse_ref(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Original Stereo Reference File",
            "",
            "Audio Files (*.flac *.wav *.aiff *.aif *.mp3);;All Files (*.*)",
        )
        if file_path:
            self.ref_edit.setText(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith((".flac", ".wav")) for url in urls):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                self.load_multichannel_file(file_path)
                event.acceptProposedAction()
                break

    # -------------------------------------------------------------------------
    # Validation Execution
    # -------------------------------------------------------------------------
    def run_validation(self) -> None:
        in_path = self.multi_edit.text().strip()
        if not in_path or not os.path.isfile(in_path):
            self.badge_lbl.setText("ERROR")
            self.badge_lbl.setStyleSheet(
                "QLabel { background-color: #3e1b1e; color: #e06c75; border-radius: 6px; "
                "font-size: 18px; font-weight: bold; padding: 6px 16px; }"
            )
            self.lbl_summary_desc.setText("Please select a valid multichannel audio file to validate.")
            return

        ref_path = self.ref_edit.text().strip() or None
        if ref_path and not os.path.isfile(ref_path):
            ref_path = None

        self.btn_validate.setEnabled(False)
        self.badge_lbl.setText("RUNNING")
        self.badge_lbl.setStyleSheet(
            "QLabel { background-color: #2c384a; color: #61afef; border-radius: 6px; "
            "font-size: 18px; font-weight: bold; padding: 6px 16px; }"
        )
        self.lbl_summary_desc.setText("Executing deterministic DSP acceptance tests...")

        self._active_worker = ValidationWorker(
            input_path=in_path,
            reference_path=ref_path,
            parent=self,
        )
        self._active_worker.finished_signal.connect(self._on_validation_finished)
        self._active_worker.error_signal.connect(self._on_validation_error)
        self._active_worker.start()

    def _on_validation_finished(self, res_dict: dict[str, Any]) -> None:
        self.btn_validate.setEnabled(True)
        passed = bool(res_dict.get("passed", False))
        checks = res_dict.get("checks", [])
        raw_json = res_dict.get("json", "{}")

        self._last_result_json = raw_json
        self.json_view.setPlainText(raw_json)

        # Update Badge
        if passed:
            self.badge_lbl.setText("PASSED")
            self.badge_lbl.setStyleSheet(
                "QLabel { background-color: #1e3a29; color: #98c379; border: 1px solid #2ea043; border-radius: 6px; "
                "font-size: 18px; font-weight: bold; letter-spacing: 2px; padding: 6px 16px; }"
            )
            self.lbl_summary_title.setText("All Acceptance Tests Passed Successfully")
            self.lbl_summary_desc.setText(
                f"{len(checks)} / {len(checks)} checks satisfied bit-preservation, fold-down SDR, stability, and clipping limits."
            )
        else:
            self.badge_lbl.setText("FAILED")
            self.badge_lbl.setStyleSheet(
                "QLabel { background-color: #3e1b1e; color: #e06c75; border: 1px solid #da3633; border-radius: 6px; "
                "font-size: 18px; font-weight: bold; letter-spacing: 2px; padding: 6px 16px; }"
            )
            self.lbl_summary_title.setText("Acceptance Criteria Violation Detected")
            n_fail = sum(1 for c in checks if not c.get("passed"))
            self.lbl_summary_desc.setText(f"{n_fail} test(s) failed strict verification standards. Inspect details below.")

        # Checklist targets map
        targets_map = {
            "channel_count": "6 (5.1) or 8 (7.1) channels",
            "clipping": "Peak amplitude ≤ 1.0 (0 dBFS)",
            "duration": "Frame count & sample rate match reference exactly",
            "flac_integrity": "FLAC container decoding consistency",
            "lfe_silent": "LFE peak amplitude == 0.0 (silent in upmix)",
            "front_preservation": "E_L = 0.0 and E_R = 0.0 (bitwise preservation)",
            "alignment_FL": "FL delay relative to reference ≤ 1 sample",
            "alignment_FR": "FR delay relative to reference ≤ 1 sample",
            "stability": "All output samples finite (no NaN or Inf)",
            "fold_down": "Stereo fold-down correlation < 0.999 or RMS > 1e-4",
            "downmix_sdr": "Fold-down reconstruction SDR > 10.0 dB",
        }

        # Populate Checks Table
        self.checks_table.setRowCount(len(checks))
        for r, chk in enumerate(checks):
            name = chk.get("check", "")
            chk_ok = bool(chk.get("passed", False))
            detail = chk.get("detail", "")
            target = targets_map.get(name, "Acceptance standard")

            # Status column badge
            status_item = QTableWidgetItem("PASS" if chk_ok else "FAIL")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if chk_ok:
                status_item.setForeground(QColor("#98c379"))
            else:
                status_item.setForeground(QColor("#e06c75"))

            self.checks_table.setItem(r, 0, status_item)
            self.checks_table.setItem(r, 1, QTableWidgetItem(name))
            self.checks_table.setItem(r, 2, QTableWidgetItem(target))
            self.checks_table.setItem(r, 3, QTableWidgetItem(detail))

    def _on_validation_error(self, message: str) -> None:
        self.btn_validate.setEnabled(True)
        self.badge_lbl.setText("ERROR")
        self.badge_lbl.setStyleSheet(
            "QLabel { background-color: #3e1b1e; color: #e06c75; border-radius: 6px; font-size: 18px; font-weight: bold; padding: 6px 16px; }"
        )
        self.lbl_summary_desc.setText(f"Validation encountered an error: {message}")

    # -------------------------------------------------------------------------
    # Channel Routing Verification
    # -------------------------------------------------------------------------
    def check_routing_frequencies(self) -> None:
        in_path = self.multi_edit.text().strip()
        if not in_path or not os.path.isfile(in_path):
            return

        expected_map = {
            "FL": 1000.0,
            "FR": 2000.0,
            "FC": 3000.0,
            "LFE": 60.0,
            "SL": 4000.0,
            "SR": 5000.0,
            "BL": 6000.0,
            "BR": 7000.0,
        }

        try:
            detected = detect_channel_frequencies(in_path)
            self.routing_table.setRowCount(len(detected))

            for r, (ch, f_det) in enumerate(detected.items()):
                f_exp = expected_map.get(ch, 0.0)
                # Match tolerance +/- 50 Hz
                is_match = abs(f_det - f_exp) < 50.0

                self.routing_table.setItem(r, 0, QTableWidgetItem(ch))
                self.routing_table.setItem(r, 1, QTableWidgetItem(f"{f_exp:.0f} Hz"))
                self.routing_table.setItem(r, 2, QTableWidgetItem(f"{f_det:.1f} Hz"))

                match_item = QTableWidgetItem("MATCH" if is_match else f"DIFF ({f_det - f_exp:+.1f} Hz)")
                match_item.setForeground(QColor("#98c379" if is_match else "#e5c07b"))
                self.routing_table.setItem(r, 3, match_item)

        except Exception as e:
            self.lbl_summary_desc.setText(f"Channel routing detection note: {e}")

    # -------------------------------------------------------------------------
    # JSON Handlers
    # -------------------------------------------------------------------------
    def _copy_json(self) -> None:
        if self._last_result_json:
            clipboard = QGuiApplication.clipboard()
            if clipboard:
                clipboard.setText(self._last_result_json)

    def _save_json(self) -> None:
        if not self._last_result_json:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Validation JSON Report",
            "validation_report.json",
            "JSON Files (*.json);;All Files (*.*)",
        )
        if dest:
            Path(dest).write_text(self._last_result_json)
