"""Concerto GUI - DSP Preset & Parameter Studio Tab.

Provides visual configuration and editing of Concerto TOML DSP profiles,
all-pass decorrelation coefficients, adaptive controller parameters,
and real-time TOML generation.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def serialize_toml_val(val: Any) -> str:
    """Format Python values into valid TOML literals."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        s = f"{float(val):.4f}".rstrip("0")
        if s.endswith("."):
            s += "0"
        return s
    if isinstance(val, list):
        items = [serialize_toml_val(x) for x in val]
        return f"[{', '.join(items)}]"
    return f'"{val}"'


def dict_to_toml_str(cfg: dict[str, Any]) -> str:
    """Convert config dictionary to structured TOML text representation."""
    lines: list[str] = ["# Concerto DSP Configuration Profile", ""]
    section_order = ["output", "surround", "decorrelation", "centre", "ai", "adaptive", "lfe", "analysis"]

    for sec in section_order:
        data = cfg.get(sec)
        if isinstance(data, dict):
            lines.append(f"[{sec}]")
            for k, v in data.items():
                lines.append(f"{k} = {serialize_toml_val(v)}")
            lines.append("")

    for sec, data in cfg.items():
        if sec not in section_order and isinstance(data, dict):
            lines.append(f"[{sec}]")
            for k, v in data.items():
                lines.append(f"{k} = {serialize_toml_val(v)}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


class ConfigTab(QWidget):
    """Visual DSP Preset & Parameter Studio."""

    config_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating_ui = False
        self._current_preset_name = "default.toml"
        self._configs_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs"

        self._setup_ui()
        self.load_preset(self._configs_dir / "default.toml")

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header toolbar: Preset selection and file actions
        toolbar_box = QFrame()
        toolbar_box.setStyleSheet(
            """
            QFrame {
                background-color: #21252b;
                border: 1px solid #2d3139;
                border-radius: 6px;
                padding: 6px;
            }
            """
        )
        tb_layout = QHBoxLayout(toolbar_box)
        tb_layout.setContentsMargins(10, 6, 10, 6)
        tb_layout.setSpacing(10)

        tb_layout.addWidget(QLabel("Active Preset:"))

        self.combo_presets = QComboBox()
        self.combo_presets.setFixedWidth(180)
        self.combo_presets.addItems([
            "default.toml",
            "adaptive.toml",
            "adaptive-ai.toml",
            "Custom File...",
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_dropdown_changed)
        tb_layout.addWidget(self.combo_presets)

        self.btn_reload = QPushButton("Reload")
        self.btn_reload.clicked.connect(self._reload_current_preset)
        tb_layout.addWidget(self.btn_reload)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save_current_preset)
        tb_layout.addWidget(self.btn_save)

        self.btn_save_as = QPushButton("Save As...")
        self.btn_save_as.clicked.connect(self._save_as_preset)
        tb_layout.addWidget(self.btn_save_as)

        self.btn_export = QPushButton("Export TOML...")
        self.btn_export.clicked.connect(self._export_toml)
        tb_layout.addWidget(self.btn_export)

        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.setStyleSheet("color: #e06c75;")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        tb_layout.addWidget(self.btn_reset)

        tb_layout.addStretch()
        main_layout.addWidget(toolbar_box)

        # Main splitter: Left side is parameter category tabs, Right side is live TOML preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Container: Sub-Tabs for Categories
        self.category_tabs = QTabWidget()
        self.category_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #2d3139;
                background-color: #181a1f;
            }
            QTabBar::tab {
                background-color: #21252b;
                color: #abb2bf;
                padding: 8px 14px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #181a1f;
                color: #61afef;
                border: 1px solid #2d3139;
                border-bottom: 1px solid #181a1f;
                font-weight: bold;
            }
            """
        )

        # Build Category Sub-Tabs
        self._build_surround_tab()
        self._build_filters_tab()
        self._build_decorrelation_tab()
        self._build_adaptive_tab()
        self._build_centre_ai_tab()
        self._build_output_tab()

        splitter.addWidget(self.category_tabs)

        # Right Container: Live TOML Preview and Editor
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(6)

        header_preview = QHBoxLayout()
        lbl_preview = QLabel("Live TOML Configuration")
        lbl_preview.setStyleSheet("font-weight: bold; color: #d7dae0; font-size: 12px;")
        header_preview.addWidget(lbl_preview)
        header_preview.addStretch()

        self.btn_apply_toml = QPushButton("Apply TOML Changes")
        self.btn_apply_toml.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #61afef;
                border: 1px solid #61afef;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #212b38;
            }
            """
        )
        self.btn_apply_toml.clicked.connect(self._apply_toml_text)
        header_preview.addWidget(self.btn_apply_toml)
        right_layout.addLayout(header_preview)

        self.toml_editor = QTextEdit()
        self.toml_editor.setFont(QFont("Courier New", 10))
        self.toml_editor.setStyleSheet(
            """
            QTextEdit {
                background-color: #181a1f;
                color: #abb2bf;
                border: 1px solid #2d3139;
                border-radius: 4px;
                padding: 6px;
            }
            """
        )
        right_layout.addWidget(self.toml_editor)

        splitter.addWidget(right_panel)
        splitter.setSizes([550, 450])
        main_layout.addWidget(splitter, 1)

    # -------------------------------------------------------------------------
    # Category Tab Builders
    # -------------------------------------------------------------------------

    def _build_surround_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box = QGroupBox("Surround Extraction & AGC Envelope")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        # Extraction mode
        grid.addWidget(QLabel("Extraction Mode:"), 0, 0)
        self.combo_extraction = QComboBox()
        self.combo_extraction.addItems(["stereo_difference", "mono_difference"])
        self.combo_extraction.currentIndexChanged.connect(self._on_ui_changed)
        grid.addWidget(self.combo_extraction, 0, 1, 1, 2)

        # Base Gain
        grid.addWidget(QLabel("Base Gain (g_s):"), 1, 0)
        self.spin_sur_gain = QDoubleSpinBox()
        self.spin_sur_gain.setRange(0.0, 1.0)
        self.spin_sur_gain.setSingleStep(0.01)
        self.spin_sur_gain.setValue(0.25)
        self.spin_sur_gain.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_sur_gain, 1, 1)

        # Stereo Beta
        grid.addWidget(QLabel("Stereo Beta (Crossfeed):"), 2, 0)
        self.spin_sur_beta = QDoubleSpinBox()
        self.spin_sur_beta.setRange(0.05, 0.98)
        self.spin_sur_beta.setSingleStep(0.01)
        self.spin_sur_beta.setValue(0.25)
        self.spin_sur_beta.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_sur_beta, 2, 1)

        # Target RMS Ratio
        grid.addWidget(QLabel("Target RMS Ratio:"), 3, 0)
        self.spin_rms_ratio = QDoubleSpinBox()
        self.spin_rms_ratio.setRange(0.05, 1.0)
        self.spin_rms_ratio.setSingleStep(0.01)
        self.spin_rms_ratio.setValue(0.35)
        self.spin_rms_ratio.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_rms_ratio, 3, 1)

        # RMS Window (s)
        grid.addWidget(QLabel("RMS Window (seconds):"), 4, 0)
        self.spin_rms_win = QDoubleSpinBox()
        self.spin_rms_win.setRange(0.05, 2.0)
        self.spin_rms_win.setSingleStep(0.05)
        self.spin_rms_win.setValue(0.400)
        self.spin_rms_win.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_rms_win, 4, 1)

        # Gain Tau (s)
        grid.addWidget(QLabel("Gain Envelope Tau (s):"), 5, 0)
        self.spin_gain_tau = QDoubleSpinBox()
        self.spin_gain_tau.setRange(0.1, 10.0)
        self.spin_gain_tau.setSingleStep(0.1)
        self.spin_gain_tau.setValue(2.0)
        self.spin_gain_tau.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_gain_tau, 5, 1)

        # Gain Min & Max
        grid.addWidget(QLabel("Gain Clamp [Min, Max]:"), 6, 0)
        clamp_row = QHBoxLayout()
        self.spin_gain_min = QDoubleSpinBox()
        self.spin_gain_min.setRange(0.0, 2.0)
        self.spin_gain_min.setValue(0.0)
        self.spin_gain_min.valueChanged.connect(self._on_ui_changed)
        clamp_row.addWidget(self.spin_gain_min)

        self.spin_gain_max = QDoubleSpinBox()
        self.spin_gain_max.setRange(0.5, 5.0)
        self.spin_gain_max.setValue(2.0)
        self.spin_gain_max.valueChanged.connect(self._on_ui_changed)
        clamp_row.addWidget(self.spin_gain_max)
        grid.addLayout(clamp_row, 6, 1, 1, 2)

        layout.addWidget(box)
        layout.addStretch()
        scroll.setWidget(content)
        self.category_tabs.addTab(scroll, "Surround")

    def _build_filters_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box = QGroupBox("Band-Limit Filter Specification (Butterworth SOS)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        # Highpass Cutoff
        grid.addWidget(QLabel("Highpass Cutoff (Hz):"), 0, 0)
        self.spin_highpass = QDoubleSpinBox()
        self.spin_highpass.setRange(20.0, 1000.0)
        self.spin_highpass.setSingleStep(10.0)
        self.spin_highpass.setValue(150.0)
        self.spin_highpass.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_highpass, 0, 1)

        # Lowpass Cutoff
        grid.addWidget(QLabel("Lowpass Cutoff (Hz):"), 1, 0)
        self.spin_lowpass = QDoubleSpinBox()
        self.spin_lowpass.setRange(1000.0, 20000.0)
        self.spin_lowpass.setSingleStep(250.0)
        self.spin_lowpass.setValue(8000.0)
        self.spin_lowpass.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_lowpass, 1, 1)

        # Filter Order
        grid.addWidget(QLabel("Filter Section Order (per edge):"), 2, 0)
        self.spin_filter_order = QSpinBox()
        self.spin_filter_order.setRange(1, 8)
        self.spin_filter_order.setValue(4)
        self.spin_filter_order.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_filter_order, 2, 1)

        layout.addWidget(box)
        layout.addStretch()
        scroll.setWidget(content)
        self.category_tabs.addTab(scroll, "Filters")

    def _build_decorrelation_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box = QGroupBox("First-Order All-Pass Decorrelation Engine")
        vbox = QVBoxLayout(box)
        vbox.setSpacing(10)

        top_row = QHBoxLayout()
        self.chk_decorr_enabled = QCheckBox("Enable Decorrelation")
        self.chk_decorr_enabled.setChecked(True)
        self.chk_decorr_enabled.toggled.connect(self._on_ui_changed)
        top_row.addWidget(self.chk_decorr_enabled)

        top_row.addSpacing(20)
        top_row.addWidget(QLabel("Cascaded Sections:"))
        self.spin_decorr_sections = QSpinBox()
        self.spin_decorr_sections.setRange(1, 6)
        self.spin_decorr_sections.setValue(3)
        self.spin_decorr_sections.valueChanged.connect(self._on_ui_changed)
        top_row.addWidget(self.spin_decorr_sections)
        top_row.addStretch()
        vbox.addLayout(top_row)

        lbl_desc = QLabel("Independent Per-Path All-Pass Coefficients (|a| < 1.0 required for stability):")
        lbl_desc.setStyleSheet("color: #5c6370; font-size: 11px;")
        vbox.addWidget(lbl_desc)

        # Coefficients table: 4 rows (SL, SR, BL, BR), 3 columns
        self.table_coeffs = QTableWidget(4, 3)
        self.table_coeffs.setVerticalHeaderLabels(["Side Left (SL)", "Side Right (SR)", "Back Left (BL)", "Back Right (BR)"])
        self.table_coeffs.setHorizontalHeaderLabels(["Section 1 (a0)", "Section 2 (a1)", "Section 3 (a2)"])
        self.table_coeffs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_coeffs.setFixedHeight(160)
        self.table_coeffs.cellChanged.connect(self._on_table_cell_changed)
        vbox.addWidget(self.table_coeffs)

        layout.addWidget(box)
        layout.addStretch()
        scroll.setWidget(content)
        self.category_tabs.addTab(scroll, "Decorrelation")

    def _build_adaptive_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box = QGroupBox("Signal-Adaptive Controller (Milestone 3)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        # Adaptive Enable
        self.chk_adaptive_enabled = QCheckBox("Enable Adaptive Controller")
        self.chk_adaptive_enabled.toggled.connect(self._on_ui_changed)
        grid.addWidget(self.chk_adaptive_enabled, 0, 0, 1, 2)

        # Beta Adaptive Enable
        self.chk_beta_adaptive = QCheckBox("Enable Signal-Adaptive Beta (rho-driven)")
        self.chk_beta_adaptive.toggled.connect(self._on_ui_changed)
        grid.addWidget(self.chk_beta_adaptive, 1, 0, 1, 2)

        # Beta Min & Max
        grid.addWidget(QLabel("Adaptive Beta Range [Min, Max]:"), 2, 0)
        beta_range_row = QHBoxLayout()
        self.spin_beta_min = QDoubleSpinBox()
        self.spin_beta_min.setRange(0.05, 0.50)
        self.spin_beta_min.setSingleStep(0.01)
        self.spin_beta_min.setValue(0.15)
        self.spin_beta_min.valueChanged.connect(self._on_ui_changed)
        beta_range_row.addWidget(self.spin_beta_min)

        self.spin_beta_max = QDoubleSpinBox()
        self.spin_beta_max.setRange(0.50, 0.98)
        self.spin_beta_max.setSingleStep(0.01)
        self.spin_beta_max.setValue(0.90)
        self.spin_beta_max.valueChanged.connect(self._on_ui_changed)
        beta_range_row.addWidget(self.spin_beta_max)
        grid.addLayout(beta_range_row, 2, 1)

        # Beta Tau
        grid.addWidget(QLabel("Beta Smoothing Tau (s):"), 3, 0)
        self.spin_beta_tau = QDoubleSpinBox()
        self.spin_beta_tau.setRange(0.1, 5.0)
        self.spin_beta_tau.setSingleStep(0.1)
        self.spin_beta_tau.setValue(1.5)
        self.spin_beta_tau.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_beta_tau, 3, 1)

        # Transition Tau
        grid.addWidget(QLabel("Transition Smoothing Tau (s):"), 4, 0)
        self.spin_trans_tau = QDoubleSpinBox()
        self.spin_trans_tau.setRange(0.1, 5.0)
        self.spin_trans_tau.setSingleStep(0.1)
        self.spin_trans_tau.setValue(2.0)
        self.spin_trans_tau.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_trans_tau, 4, 1)

        # Transient Tau
        grid.addWidget(QLabel("Transient Attack Tau (s):"), 5, 0)
        self.spin_transient_tau = QDoubleSpinBox()
        self.spin_transient_tau.setRange(0.005, 0.200)
        self.spin_transient_tau.setSingleStep(0.005)
        self.spin_transient_tau.setDecimals(3)
        self.spin_transient_tau.setValue(0.050)
        self.spin_transient_tau.valueChanged.connect(self._on_ui_changed)
        grid.addWidget(self.spin_transient_tau, 5, 1)

        layout.addWidget(box)
        layout.addStretch()
        scroll.setWidget(content)
        self.category_tabs.addTab(scroll, "Adaptive")

    def _build_centre_ai_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box_centre = QGroupBox("Centre Channel Derivation")
        grid_c = QGridLayout(box_centre)
        grid_c.setHorizontalSpacing(12)
        grid_c.setVerticalSpacing(10)

        grid_c.addWidget(QLabel("Centre Channel Mode:"), 0, 0)
        self.combo_centre_mode = QComboBox()
        self.combo_centre_mode.addItems(["disabled", "mid_fill", "vocal_mask"])
        self.combo_centre_mode.currentIndexChanged.connect(self._on_ui_changed)
        grid_c.addWidget(self.combo_centre_mode, 0, 1)

        grid_c.addWidget(QLabel("Mid Fill Gain:"), 1, 0)
        self.spin_mid_fill_gain = QDoubleSpinBox()
        self.spin_mid_fill_gain.setRange(0.0, 1.0)
        self.spin_mid_fill_gain.setSingleStep(0.05)
        self.spin_mid_fill_gain.setValue(0.3548)
        self.spin_mid_fill_gain.valueChanged.connect(self._on_ui_changed)
        grid_c.addWidget(self.spin_mid_fill_gain, 1, 1)

        grid_c.addWidget(QLabel("Vocal Gain (dB):"), 2, 0)
        self.spin_vocal_gain = QDoubleSpinBox()
        self.spin_vocal_gain.setRange(-24.0, 6.0)
        self.spin_vocal_gain.setSingleStep(0.5)
        self.spin_vocal_gain.setValue(-6.0)
        self.spin_vocal_gain.valueChanged.connect(self._on_ui_changed)
        grid_c.addWidget(self.spin_vocal_gain, 2, 1)
        layout.addWidget(box_centre)

        box_ai = QGroupBox("Neural Stem Separation Engine")
        grid_a = QGridLayout(box_ai)
        grid_a.setHorizontalSpacing(12)
        grid_a.setVerticalSpacing(10)

        self.chk_ai_enabled = QCheckBox("Enable AI Stem Separation")
        self.chk_ai_enabled.toggled.connect(self._on_ui_changed)
        grid_a.addWidget(self.chk_ai_enabled, 0, 0, 1, 2)

        grid_a.addWidget(QLabel("Default Model:"), 1, 0)
        self.combo_ai_model = QComboBox()
        self.combo_ai_model.addItems(["htdemucs", "BS-Roformer", "melband"])
        self.combo_ai_model.currentIndexChanged.connect(self._on_ui_changed)
        grid_a.addWidget(self.combo_ai_model, 1, 1)
        layout.addWidget(box_ai)

        layout.addStretch()
        scroll.setWidget(content)
        self.category_tabs.addTab(scroll, "Centre & AI")

    def _build_output_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box = QGroupBox("Output Format & LFE Routing")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("Output Layout:"), 0, 0)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["7.1", "5.1"])
        self.combo_layout.currentIndexChanged.connect(self._on_ui_changed)
        grid.addWidget(self.combo_layout, 0, 1)

        grid.addWidget(QLabel("FLAC Bit Depth:"), 1, 0)
        self.combo_bit_depth = QComboBox()
        self.combo_bit_depth.addItems(["16", "24", "32"])
        self.combo_bit_depth.setCurrentText("24")
        self.combo_bit_depth.currentIndexChanged.connect(self._on_ui_changed)
        grid.addWidget(self.combo_bit_depth, 1, 1)

        self.chk_preserve_front = QCheckBox("Preserve Front Channels (Bit-Identical FL/FR)")
        self.chk_preserve_front.setChecked(True)
        self.chk_preserve_front.toggled.connect(self._on_ui_changed)
        grid.addWidget(self.chk_preserve_front, 2, 0, 1, 2)

        self.chk_lfe_enabled = QCheckBox("Enable LFE Subwoofer Channel Routing")
        self.chk_lfe_enabled.setChecked(False)
        self.chk_lfe_enabled.toggled.connect(self._on_ui_changed)
        grid.addWidget(self.chk_lfe_enabled, 3, 0, 1, 2)

        layout.addWidget(box)
        layout.addStretch()
        scroll.setWidget(content)
        self.category_tabs.addTab(scroll, "Output & LFE")

    # -------------------------------------------------------------------------
    # Configuration Data Binding
    # -------------------------------------------------------------------------

    def get_current_config(self) -> dict[str, Any]:
        """Collect current parameter values into a complete config dictionary."""
        # Decorrelation coefficients from table
        coeffs: dict[str, list[float]] = {"a_left": [], "a_right": [], "a_back_l": [], "a_back_r": []}
        key_map = [("a_left", 0), ("a_right", 1), ("a_back_l", 2), ("a_back_r", 3)]
        for k, row in key_map:
            vals: list[float] = []
            for col in range(3):
                item = self.table_coeffs.item(row, col)
                try:
                    vals.append(float(item.text()) if item else 0.0)
                except ValueError:
                    vals.append(0.0)
            coeffs[k] = vals

        return {
            "output": {
                "layout": self.combo_layout.currentText(),
                "bit_depth": int(self.combo_bit_depth.currentText()),
                "preserve_front": self.chk_preserve_front.isChecked(),
            },
            "surround": {
                "gain": round(float(self.spin_sur_gain.value()), 4),
                "extraction": self.combo_extraction.currentText(),
                "stereo_beta": round(float(self.spin_sur_beta.value()), 4),
                "highpass_hz": round(float(self.spin_highpass.value()), 1),
                "lowpass_hz": round(float(self.spin_lowpass.value()), 1),
                "filter_order": int(self.spin_filter_order.value()),
                "target_rms_ratio": round(float(self.spin_rms_ratio.value()), 4),
                "rms_window_s": round(float(self.spin_rms_win.value()), 4),
                "gain_tau_s": round(float(self.spin_gain_tau.value()), 2),
                "gain_min": round(float(self.spin_gain_min.value()), 2),
                "gain_max": round(float(self.spin_gain_max.value()), 2),
            },
            "decorrelation": {
                "enabled": self.chk_decorr_enabled.isChecked(),
                "sections": int(self.spin_decorr_sections.value()),
                "a_left": coeffs["a_left"],
                "a_right": coeffs["a_right"],
                "a_back_l": coeffs["a_back_l"],
                "a_back_r": coeffs["a_back_r"],
            },
            "centre": {
                "mode": self.combo_centre_mode.currentText(),
                "mid_fill_gain": round(float(self.spin_mid_fill_gain.value()), 4),
                "vocal_gain_db": round(float(self.spin_vocal_gain.value()), 2),
            },
            "ai": {
                "enabled": self.chk_ai_enabled.isChecked(),
                "model": self.combo_ai_model.currentText(),
            },
            "adaptive": {
                "enabled": self.chk_adaptive_enabled.isChecked(),
                "transition_tau_s": round(float(self.spin_trans_tau.value()), 2),
                "transient_tau_s": round(float(self.spin_transient_tau.value()), 3),
                "beta_adaptive": self.chk_beta_adaptive.isChecked(),
                "beta_min": round(float(self.spin_beta_min.value()), 4),
                "beta_max": round(float(self.spin_beta_max.value()), 4),
                "beta_tau_s": round(float(self.spin_beta_tau.value()), 2),
            },
            "lfe": {
                "enabled": self.chk_lfe_enabled.isChecked(),
            },
            "analysis": {
                "fft_size": 4096,
                "hop_size": 1024,
                "window": "hann",
            },
        }

    def load_config_dict(self, cfg: dict[str, Any]) -> None:
        """Populate UI controls from a configuration dictionary."""
        self._updating_ui = True
        try:
            out = cfg.get("output", {})
            self.combo_layout.setCurrentText(str(out.get("layout", "7.1")))
            self.combo_bit_depth.setCurrentText(str(out.get("bit_depth", 24)))
            self.chk_preserve_front.setChecked(bool(out.get("preserve_front", True)))

            sur = cfg.get("surround", {})
            self.spin_sur_gain.setValue(float(sur.get("gain", 0.25)))
            self.combo_extraction.setCurrentText(str(sur.get("extraction", "stereo_difference")))
            self.spin_sur_beta.setValue(float(sur.get("stereo_beta", 0.25)))
            self.spin_highpass.setValue(float(sur.get("highpass_hz", 150.0)))
            self.spin_lowpass.setValue(float(sur.get("lowpass_hz", 8000.0)))
            self.spin_filter_order.setValue(int(sur.get("filter_order", 4)))
            self.spin_rms_ratio.setValue(float(sur.get("target_rms_ratio", 0.35)))
            self.spin_rms_win.setValue(float(sur.get("rms_window_s", 0.400)))
            self.spin_gain_tau.setValue(float(sur.get("gain_tau_s", 2.0)))
            self.spin_gain_min.setValue(float(sur.get("gain_min", 0.0)))
            self.spin_gain_max.setValue(float(sur.get("gain_max", 2.0)))

            dec = cfg.get("decorrelation", {})
            self.chk_decorr_enabled.setChecked(bool(dec.get("enabled", True)))
            self.spin_decorr_sections.setValue(int(dec.get("sections", 3)))

            table_map = [
                ("a_left", 0, [0.68, -0.41, 0.23]),
                ("a_right", 1, [-0.55, 0.37, -0.19]),
                ("a_back_l", 2, [0.31, -0.62, 0.44]),
                ("a_back_r", 3, [-0.29, 0.48, -0.57]),
            ]
            for key, row, fallback in table_map:
                vals = dec.get(key, fallback)
                for col in range(3):
                    v = vals[col] if col < len(vals) else 0.0
                    item = QTableWidgetItem(f"{float(v):.2f}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table_coeffs.setItem(row, col, item)

            cen = cfg.get("centre", {})
            self.combo_centre_mode.setCurrentText(str(cen.get("mode", "disabled")))
            self.spin_mid_fill_gain.setValue(float(cen.get("mid_fill_gain", 0.3548)))
            self.spin_vocal_gain.setValue(float(cen.get("vocal_gain_db", -6.0)))

            ai = cfg.get("ai", {})
            self.chk_ai_enabled.setChecked(bool(ai.get("enabled", False)))
            self.combo_ai_model.setCurrentText(str(ai.get("model", "htdemucs")))

            adv = cfg.get("adaptive", {})
            self.chk_adaptive_enabled.setChecked(bool(adv.get("enabled", False)))
            self.spin_trans_tau.setValue(float(adv.get("transition_tau_s", 2.0)))
            self.spin_transient_tau.setValue(float(adv.get("transient_tau_s", 0.050)))
            self.chk_beta_adaptive.setChecked(bool(adv.get("beta_adaptive", False)))
            self.spin_beta_min.setValue(float(adv.get("beta_min", 0.15)))
            self.spin_beta_max.setValue(float(adv.get("beta_max", 0.90)))
            self.spin_beta_tau.setValue(float(adv.get("beta_tau_s", 1.5)))

            lfe = cfg.get("lfe", {})
            self.chk_lfe_enabled.setChecked(bool(lfe.get("enabled", False)))
        finally:
            self._updating_ui = False

        self._refresh_toml_preview()

    def _refresh_toml_preview(self) -> None:
        cfg = self.get_current_config()
        text = dict_to_toml_str(cfg)
        self.toml_editor.blockSignals(True)
        self.toml_editor.setPlainText(text)
        self.toml_editor.blockSignals(False)
        self.config_changed.emit(cfg)

    def _on_ui_changed(self) -> None:
        if not self._updating_ui:
            self._refresh_toml_preview()

    def _on_table_cell_changed(self, row: int, col: int) -> None:
        if not self._updating_ui:
            self._refresh_toml_preview()

    # -------------------------------------------------------------------------
    # Preset Actions
    # -------------------------------------------------------------------------

    def load_preset(self, path: Path | str) -> None:
        """Load configuration from a TOML file."""
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "rb") as f:
                data = tomllib.load(f)
            self._current_preset_name = p.name
            self.load_config_dict(data)
        except Exception as e:
            QMessageBox.critical(self, "Load Preset Error", f"Failed to load {p.name}:\n{e}")

    def _on_preset_dropdown_changed(self, index: int) -> None:
        if index == 0:
            self.load_preset(self._configs_dir / "default.toml")
        elif index == 1:
            self.load_preset(self._configs_dir / "adaptive.toml")
        elif index == 2:
            self.load_preset(self._configs_dir / "adaptive-ai.toml")
        elif index == 3:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select TOML Configuration File",
                str(self._configs_dir),
                "TOML Files (*.toml);;All Files (*.*)",
            )
            if path:
                self.load_preset(path)

    def _reload_current_preset(self) -> None:
        self.load_preset(self._configs_dir / self._current_preset_name)

    def _save_current_preset(self) -> None:
        target = self._configs_dir / self._current_preset_name
        try:
            cfg = self.get_current_config()
            toml_text = dict_to_toml_str(cfg)
            with open(target, "w", encoding="utf-8") as f:
                f.write(toml_text)
            QMessageBox.information(self, "Preset Saved", f"Successfully saved to {target.name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save preset:\n{e}")

    def _save_as_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preset As",
            str(self._configs_dir / self._current_preset_name),
            "TOML Files (*.toml);;All Files (*.*)",
        )
        if path:
            try:
                cfg = self.get_current_config()
                toml_text = dict_to_toml_str(cfg)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(toml_text)
                self._current_preset_name = Path(path).name
                QMessageBox.information(self, "Preset Saved", f"Saved configuration to {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Save As Error", f"Failed to save:\n{e}")

    def _export_toml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export TOML Profile",
            "concerto_custom.toml",
            "TOML Files (*.toml);;All Files (*.*)",
        )
        if path:
            try:
                cfg = self.get_current_config()
                toml_text = dict_to_toml_str(cfg)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(toml_text)
                QMessageBox.information(self, "Export Complete", f"Exported TOML to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export TOML:\n{e}")

    def _reset_to_defaults(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Reset Parameters",
            "Are you sure you want to reset all parameters to default.toml?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.combo_presets.setCurrentIndex(0)
            self.load_preset(self._configs_dir / "default.toml")

    def _apply_toml_text(self) -> None:
        raw_text = self.toml_editor.toPlainText()
        try:
            data = tomllib.loads(raw_text)
            self.load_config_dict(data)
            QMessageBox.information(self, "TOML Applied", "Parameters successfully parsed and updated in UI.")
        except Exception as e:
            QMessageBox.critical(self, "TOML Syntax Error", f"Invalid TOML format:\n{e}")
