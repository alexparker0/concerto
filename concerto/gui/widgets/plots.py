"""Spectral correlation, coherence, goniometer, and stem energy visualization widgets.

Provides high-performance pyqtgraph and custom QPainter widgets for
analyzing multichannel audio, stereo field geometry, and stem energy distribution.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Configure pyqtgraph for dark DAW workstation theme
pg.setConfigOptions(antialias=True, background="#181a1f", foreground="#abb2bf")


class SpectralCorrelationPlot(QWidget):
    """Plot widget for frequency-dependent Pearson correlation rho(f), coherence gamma^2(f), and directness d(f).

    Supports logarithmic and linear frequency display (20 Hz - 20 kHz) with
    custom acoustic octave ticks, crosshair coordinate inspection, and interactive controls.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._log_mode: bool = True
        self._freqs: np.ndarray | None = None
        self._rho: np.ndarray | None = None
        self._coherence: np.ndarray | None = None
        self._directness: np.ndarray | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header toolbar
        toolbar = QWidget(self)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(4, 2, 4, 2)
        tb_layout.setSpacing(8)

        self._title_label = QLabel("Spectral Correlation & Coherence", toolbar)
        title_font = QFont()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet("color: #d7dae0; font-size: 12px;")
        tb_layout.addWidget(self._title_label)

        tb_layout.addStretch(1)

        # Inspector readout
        self._inspect_label = QLabel("Hover to inspect bins", toolbar)
        self._inspect_label.setStyleSheet(
            "color: #abb2bf; font-family: monospace; font-size: 11px; "
            "background-color: #21252b; padding: 2px 8px; border-radius: 3px; border: 1px solid #2d3139;"
        )
        tb_layout.addWidget(self._inspect_label)

        # Log/linear toggle
        self._scale_btn = QPushButton("Scale: Log", toolbar)
        self._scale_btn.setCheckable(True)
        self._scale_btn.setChecked(True)
        self._scale_btn.setStyleSheet(
            "QPushButton { background-color: #282c34; color: #abb2bf; border: 1px solid #2d3139; "
            "padding: 2px 10px; border-radius: 3px; font-size: 11px; } "
            "QPushButton:hover { background-color: #353b45; color: #ffffff; } "
            "QPushButton:checked { background-color: #2c384a; color: #61afef; border-color: #61afef; }"
        )
        self._scale_btn.toggled.connect(self._on_scale_toggled)
        tb_layout.addWidget(self._scale_btn)

        # Clear button
        self._clear_btn = QPushButton("Clear", toolbar)
        self._clear_btn.setStyleSheet(
            "QPushButton { background-color: #282c34; color: #abb2bf; border: 1px solid #2d3139; "
            "padding: 2px 10px; border-radius: 3px; font-size: 11px; } "
            "QPushButton:hover { background-color: #353b45; color: #ffffff; }"
        )
        self._clear_btn.clicked.connect(self.clear)
        tb_layout.addWidget(self._clear_btn)

        layout.addWidget(toolbar)

        # Main PlotWidget
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=True, y=False)

        # Style axes
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel("bottom", "Frequency", units="Hz")
        plot_item.setLabel("left", "Value", units="")
        plot_item.showAxis("right", True)
        plot_item.showAxis("top", True)
        plot_item.getAxis("right").setStyle(showValues=False)
        plot_item.getAxis("top").setStyle(showValues=False)

        # Zero reference line
        self._zero_line = pg.InfiniteLine(
            angle=0,
            pos=0.0,
            pen=pg.mkPen(color="#5c6370", width=1, style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self.plot_widget.addItem(self._zero_line)

        # +1.0 reference line
        self._one_line = pg.InfiniteLine(
            angle=0,
            pos=1.0,
            pen=pg.mkPen(color="#3e4451", width=1, style=Qt.PenStyle.DotLine),
            movable=False,
        )
        self.plot_widget.addItem(self._one_line)

        # Legend
        self._legend = self.plot_widget.addLegend(
            offset=(12, 12),
            labelTextSize="10pt",
            labelTextColor="#abb2bf",
            brush=pg.mkBrush("#1b1e23e0"),
            pen=pg.mkPen("#2d3139", width=1),
        )

        # Curves
        # Correlation rho(f): cyan #61afef
        self._curve_rho = self.plot_widget.plot(
            name="Correlation ρ(f)",
            pen=pg.mkPen("#61afef", width=2.0),
        )

        # Coherence gamma^2(f): green #98c379
        self._curve_coherence = self.plot_widget.plot(
            name="Coherence γ²(f)",
            pen=pg.mkPen("#98c379", width=2.0),
        )

        # Directness d(f): amber #e5c07b
        self._curve_directness = self.plot_widget.plot(
            name="Directness d(f)",
            pen=pg.mkPen("#e5c07b", width=2.0, style=Qt.PenStyle.DashLine),
        )

        # Crosshair cursor
        self._v_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#61afef99", width=1, style=Qt.PenStyle.DashLine),
        )
        self._h_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#61afef99", width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self._v_line, ignoreBounds=True)
        self.plot_widget.addItem(self._h_line, ignoreBounds=True)
        self._v_line.hide()
        self._h_line.hide()

        # Connect mouse hover tracking
        self._mouse_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        layout.addWidget(self.plot_widget)

        # Initial axes configuration
        self._apply_axis_scale()

    def _apply_axis_scale(self) -> None:
        """Apply logarithmic or linear scale and set custom ticks."""
        bottom_axis = self.plot_widget.getAxis("bottom")
        if self._log_mode:
            self.plot_widget.setLogMode(x=True, y=False)
            major_ticks = [
                (np.log10(100.0), "100 Hz"),
                (np.log10(1000.0), "1 kHz"),
                (np.log10(10000.0), "10 kHz"),
            ]
            minor_ticks = [
                (np.log10(20.0), "20"),
                (np.log10(50.0), "50"),
                (np.log10(200.0), "200"),
                (np.log10(500.0), "500"),
                (np.log10(2000.0), "2k"),
                (np.log10(5000.0), "5k"),
                (np.log10(20000.0), "20k"),
            ]
            bottom_axis.setTicks([major_ticks, minor_ticks])
            self.plot_widget.setXRange(np.log10(20.0), np.log10(20000.0), padding=0.01)
        else:
            self.plot_widget.setLogMode(x=False, y=False)
            ticks = [
                (20.0, "20 Hz"),
                (1000.0, "1 kHz"),
                (5000.0, "5 kHz"),
                (10000.0, "10 kHz"),
                (15000.0, "15 kHz"),
                (20000.0, "20 kHz"),
            ]
            bottom_axis.setTicks([ticks])
            self.plot_widget.setXRange(20.0, 20000.0, padding=0.01)

        self.plot_widget.setYRange(-1.05, 1.05, padding=0.02)

    def _on_scale_toggled(self, checked: bool) -> None:
        self.set_log_mode(checked)

    def set_log_mode(self, enabled: bool) -> None:
        """Switch between logarithmic and linear frequency display."""
        self._log_mode = bool(enabled)
        self._scale_btn.blockSignals(True)
        self._scale_btn.setChecked(self._log_mode)
        self._scale_btn.setText("Scale: Log" if self._log_mode else "Scale: Linear")
        self._scale_btn.blockSignals(False)

        self._apply_axis_scale()
        self._replot()

    def set_data(
        self,
        freqs: np.ndarray,
        rho: np.ndarray,
        coherence: np.ndarray,
        directness: np.ndarray | None = None,
    ) -> None:
        """Update plot curves with new spectral correlation data.

        Parameters
        ----------
        freqs: np.ndarray
            Frequencies in Hz (1D array, e.g. 20 Hz to 20,000 Hz).
        rho: np.ndarray
            Pearson correlation coefficient per frequency bin (-1.0 to +1.0).
        coherence: np.ndarray
            Magnitude squared coherence per frequency bin (0.0 to 1.0).
        directness: np.ndarray | None
            Signal directness ratio d(f) (0.0 to 1.0).
        """
        self._freqs = np.asarray(freqs, dtype=np.float64)
        self._rho = np.asarray(rho, dtype=np.float64)
        self._coherence = np.asarray(coherence, dtype=np.float64)
        self._directness = (
            np.asarray(directness, dtype=np.float64) if directness is not None else None
        )

        self._replot()

    def _replot(self) -> None:
        if self._freqs is None or len(self._freqs) == 0:
            return

        # Ensure frequencies are positive for log mode
        valid_mask = self._freqs > 0.0
        f = self._freqs[valid_mask]
        if len(f) == 0:
            return

        if self._rho is not None:
            r = np.clip(self._rho[valid_mask], -1.0, 1.0)
            self._curve_rho.setData(f, r)

        if self._coherence is not None:
            c = np.clip(self._coherence[valid_mask], 0.0, 1.0)
            self._curve_coherence.setData(f, c)

        if self._directness is not None:
            d = np.clip(self._directness[valid_mask], 0.0, 1.0)
            self._curve_directness.setData(f, d)
            self._curve_directness.show()
        else:
            self._curve_directness.clear()
            self._curve_directness.hide()

    def clear(self) -> None:
        """Clear all curves and reset inspector readout."""
        self._freqs = None
        self._rho = None
        self._coherence = None
        self._directness = None

        self._curve_rho.clear()
        self._curve_coherence.clear()
        self._curve_directness.clear()
        self._v_line.hide()
        self._h_line.hide()
        self._inspect_label.setText("Hover to inspect bins")

    def _on_mouse_moved(self, evt: tuple) -> None:
        pos = evt[0]
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            self._v_line.hide()
            self._h_line.hide()
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_val = mouse_point.x()
        y_val = mouse_point.y()

        # Compute actual frequency in Hz
        freq_hz = 10.0**x_val if self._log_mode else x_val
        if freq_hz < 15.0 or freq_hz > 24000.0 or y_val < -1.1 or y_val > 1.1:
            self._v_line.hide()
            self._h_line.hide()
            return

        self._v_line.setPos(x_val)
        self._h_line.setPos(y_val)
        self._v_line.show()
        self._h_line.show()

        # Interpolate values if data is loaded
        if self._freqs is not None and len(self._freqs) > 1:
            idx = np.argmin(np.abs(self._freqs - freq_hz))
            r_val = self._rho[idx] if self._rho is not None else 0.0
            c_val = self._coherence[idx] if self._coherence is not None else 0.0
            d_str = (
                f" | d: {self._directness[idx]:.2f}"
                if self._directness is not None
                else ""
            )
            self._inspect_label.setText(
                f"{freq_hz:.0f} Hz | ρ: {r_val:+.2f} | γ²: {c_val:.2f}{d_str}"
            )
        else:
            self._inspect_label.setText(f"{freq_hz:.0f} Hz | y: {y_val:+.2f}")


class PhaseGoniometerWidget(QWidget):
    """Lissajous stereo vectorscope and horizontal phase correlation meter.

    Displays Left vs Right (rotated 45 degrees as Mid vs Side) real-time stereo field
    and a Pearson correlation meter ranging from -1.0 (anti-phase red) to 0.0 (wide amber)
    to +1.0 (mono green).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._points: list[QPointF] = []
        self._correlation: float = 0.0
        self._left_rms: float = 0.0
        self._right_rms: float = 0.0

    def set_audio(self, left: np.ndarray, right: np.ndarray) -> None:
        """Sample or average stereo chunks to update vectorscope and correlation bar.

        Parameters
        ----------
        left: np.ndarray
            Left channel PCM audio samples (float32 or float64).
        right: np.ndarray
            Right channel PCM audio samples (float32 or float64).
        """
        l_arr = np.asarray(left, dtype=np.float32).ravel()
        r_arr = np.asarray(right, dtype=np.float32).ravel()

        n_samples = min(len(l_arr), len(r_arr))
        if n_samples == 0:
            self.clear()
            return

        l_arr = l_arr[:n_samples]
        r_arr = r_arr[:n_samples]

        # Calculate RMS
        self._left_rms = float(np.sqrt(np.mean(l_arr**2)))
        self._right_rms = float(np.sqrt(np.mean(r_arr**2)))

        # Calculate Pearson phase correlation: sum(L*R) / sqrt(sum(L^2)*sum(R^2))
        dot_product = float(np.dot(l_arr, r_arr))
        norm_l = float(np.dot(l_arr, l_arr))
        norm_r = float(np.dot(r_arr, r_arr))
        denom = np.sqrt(norm_l * norm_r)

        if denom > 1e-12:
            self._correlation = float(np.clip(dot_product / denom, -1.0, 1.0))
        else:
            self._correlation = 0.0

        # Downsample for smooth 60fps rendering (up to 1500 points)
        max_points = 1500
        if n_samples > max_points:
            indices = np.linspace(0, n_samples - 1, max_points, dtype=int)
            l_sub = l_arr[indices]
            r_sub = r_arr[indices]
        else:
            l_sub = l_arr
            r_sub = r_arr

        # Rotate 45 degrees into M/S space:
        # Mid  = (L + R) / sqrt(2) -> vertical axis (+M up)
        # Side = (R - L) / sqrt(2) -> horizontal axis (+S right)
        inv_sqrt2 = 0.70710678
        m_vals = (l_sub + r_sub) * inv_sqrt2
        s_vals = (r_sub - l_sub) * inv_sqrt2

        # Convert to QPointF list (normalized to [-1.0, 1.0])
        self._points = [QPointF(float(s), float(m)) for s, m in zip(s_vals, m_vals)]
        self.update()

    def set_correlation(self, correlation: float) -> None:
        """Set correlation directly (-1.0 to 1.0)."""
        self._correlation = float(np.clip(correlation, -1.0, 1.0))
        self.update()

    def clear(self) -> None:
        """Clear vectorscope points and reset correlation meter."""
        self._points = []
        self._correlation = 0.0
        self._left_rms = 0.0
        self._right_rms = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = self.width()
        height = self.height()

        # Background fill
        painter.fillRect(0, 0, width, height, QColor("#181a1f"))

        # Dimensions: reserve 50px at the bottom for the correlation meter
        meter_height = 46
        scope_height = max(10, height - meter_height - 10)
        center_x = width / 2.0
        center_y = 10 + scope_height / 2.0
        radius = min(width - 24, scope_height - 10) / 2.0

        if radius > 15:
            self._draw_scope(painter, center_x, center_y, radius)

        # Draw correlation bar
        self._draw_correlation_meter(painter, width, height, meter_height)

    def _draw_scope(
        self, painter: QPainter, cx: float, cy: float, radius: float
    ) -> None:
        # Scope outer background circle
        painter.setPen(QPen(QColor("#2d3139"), 1.5))
        painter.setBrush(QBrush(QColor("#131519")))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Graticule concentric circles: 0.25, 0.5, 0.75
        dash_pen = QPen(QColor("#252930"), 1, Qt.PenStyle.DashLine)
        painter.setPen(dash_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for scale in (0.25, 0.5, 0.75):
            painter.drawEllipse(QPointF(cx, cy), radius * scale, radius * scale)

        # Crosshair axes: Mid (vertical), Side (horizontal)
        axis_pen = QPen(QColor("#2d3139"), 1.0)
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(cx, cy - radius), QPointF(cx, cy + radius))
        painter.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))

        # 45-degree diagonal reference lines (pure Left and pure Right)
        diag_pen = QPen(QColor("#24272f"), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(diag_pen)
        diag_offset = radius * 0.7071
        painter.drawLine(
            QPointF(cx - diag_offset, cy - diag_offset),
            QPointF(cx + diag_offset, cy + diag_offset),
        )
        painter.drawLine(
            QPointF(cx - diag_offset, cy + diag_offset),
            QPointF(cx + diag_offset, cy - diag_offset),
        )

        # Graticule labels: +M, -M, L, R, +S
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#5c6370"))

        painter.drawText(
            QRectF(cx - 20, cy - radius + 2, 40, 16),
            Qt.AlignmentFlag.AlignCenter,
            "+M",
        )
        painter.drawText(
            QRectF(cx - 20, cy + radius - 18, 40, 16),
            Qt.AlignmentFlag.AlignCenter,
            "-M",
        )
        painter.drawText(
            QRectF(cx - radius + 4, cy - 8, 20, 16),
            Qt.AlignmentFlag.AlignCenter,
            "+S",
        )
        painter.drawText(
            QRectF(cx + radius - 24, cy - 8, 20, 16),
            Qt.AlignmentFlag.AlignCenter,
            "-S",
        )
        painter.drawText(
            QRectF(cx - diag_offset - 10, cy - diag_offset - 14, 20, 16),
            Qt.AlignmentFlag.AlignCenter,
            "L",
        )
        painter.drawText(
            QRectF(cx + diag_offset - 10, cy - diag_offset - 14, 20, 16),
            Qt.AlignmentFlag.AlignCenter,
            "R",
        )

        # Draw point cloud
        if self._points:
            # Subtle phosphor dots in cyan
            painter.setPen(QPen(QColor(97, 175, 239, 140), 2.0))
            pts_to_draw = []
            for pt in self._points:
                px = cx + pt.x() * radius * 0.95
                py = cy - pt.y() * radius * 0.95
                pts_to_draw.append(QPointF(px, py))
            painter.drawPoints(pts_to_draw)

    def _draw_correlation_meter(
        self, painter: QPainter, width: int, height: int, meter_height: int
    ) -> None:
        bar_y = height - meter_height + 4
        bar_h = 16
        bar_margin = 16
        bar_w = width - (bar_margin * 2)

        if bar_w <= 20:
            return

        # Header label
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor("#abb2bf"))
        painter.drawText(
            QRectF(bar_margin, bar_y - 18, bar_w * 0.6, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Phase Correlation",
        )

        # Correlation value string and badge color
        corr = self._correlation
        if corr >= 0.2:
            status_text = f"{corr:+.2f} (In Phase)"
            badge_color = QColor("#98c379")
        elif corr <= -0.2:
            status_text = f"{corr:+.2f} (Anti-Phase)"
            badge_color = QColor("#e06c75")
        else:
            status_text = f"{corr:+.2f} (Wide/Diff)"
            badge_color = QColor("#e5c07b")

        painter.setPen(badge_color)
        painter.drawText(
            QRectF(bar_margin + bar_w * 0.4, bar_y - 18, bar_w * 0.6, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            status_text,
        )

        # Track background
        track_rect = QRectF(bar_margin, bar_y, bar_w, bar_h)
        painter.setPen(QPen(QColor("#2d3139"), 1))
        painter.setBrush(QBrush(QColor("#1e2227")))
        painter.drawRoundedRect(track_rect, 4, 4)

        # Center zero line
        mid_x = bar_margin + bar_w / 2.0
        painter.setPen(QPen(QColor("#5c6370"), 1))
        painter.drawLine(QPointF(mid_x, bar_y), QPointF(mid_x, bar_y + bar_h))

        # Filled active correlation bar from center
        half_w = bar_w / 2.0
        bar_len = abs(corr) * half_w

        if bar_len > 1.0:
            fill_rect = (
                QRectF(mid_x, bar_y + 1, bar_len, bar_h - 2)
                if corr > 0
                else QRectF(mid_x - bar_len, bar_y + 1, bar_len, bar_h - 2)
            )
            painter.setPen(Qt.PenStyle.NoPen)
            fill_color = QColor(badge_color)
            fill_color.setAlpha(180)
            painter.setBrush(QBrush(fill_color))
            painter.drawRoundedRect(fill_rect, 2, 2)

        # Cursor needle
        needle_x = mid_x + corr * half_w
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(
            QPointF(needle_x, bar_y - 2), QPointF(needle_x, bar_y + bar_h + 2)
        )

        # Legend marks: -1, 0, +1
        font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(QColor("#5c6370"))
        painter.drawText(
            QRectF(bar_margin, bar_y + bar_h + 1, 20, 12),
            Qt.AlignmentFlag.AlignLeft,
            "-1",
        )
        painter.drawText(
            QRectF(mid_x - 10, bar_y + bar_h + 1, 20, 12),
            Qt.AlignmentFlag.AlignCenter,
            "0",
        )
        painter.drawText(
            QRectF(bar_margin + bar_w - 20, bar_y + bar_h + 1, 20, 12),
            Qt.AlignmentFlag.AlignRight,
            "+1",
        )


class StemEnergyWidget(QWidget):
    """Multi-bar energy level meter for decomposed stems (Vocals, Drums, Bass, Other).

    Features color-coded stem indicators, graduated dB scales, peak ticks, and numeric readouts.
    """

    STEM_COLORS = {
        "Vocals": QColor("#61afef"),  # Cyan
        "Drums": QColor("#e06c75"),  # Coral
        "Bass": QColor("#c678dd"),  # Purple
        "Other": QColor("#98c379"),  # Green
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._levels: dict[str, float] = {
            "Vocals": 0.0,
            "Drums": 0.0,
            "Bass": 0.0,
            "Other": 0.0,
        }
        self._peaks: dict[str, float] = {
            "Vocals": 0.0,
            "Drums": 0.0,
            "Bass": 0.0,
            "Other": 0.0,
        }

    def set_levels(
        self,
        vocals_rms: float,
        drums_rms: float,
        bass_rms: float,
        other_rms: float,
    ) -> None:
        """Update stem RMS energy levels (linear 0.0 to 1.0 or amplitude)."""
        self._set_stem_val("Vocals", vocals_rms)
        self._set_stem_val("Drums", drums_rms)
        self._set_stem_val("Bass", bass_rms)
        self._set_stem_val("Other", other_rms)
        self.update()

    def set_stem_levels(self, levels: dict[str, float]) -> None:
        """Update levels via dictionary (case-insensitive keys)."""
        for k, v in levels.items():
            cap = k.capitalize()
            if cap in self._levels:
                self._set_stem_val(cap, v)
        self.update()

    def _set_stem_val(self, stem: str, val: float) -> None:
        v = float(np.clip(val, 0.0, 1.0))
        self._levels[stem] = v
        if v > self._peaks[stem]:
            self._peaks[stem] = v

    def clear(self) -> None:
        """Reset all stem energy levels and peaks to zero."""
        for k in self._levels:
            self._levels[k] = 0.0
            self._peaks[k] = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = self.width()
        height = self.height()

        # Canvas background
        painter.fillRect(0, 0, width, height, QColor("#181a1f"))

        stems = ["Vocals", "Drums", "Bass", "Other"]
        n_stems = len(stems)

        margin_x = 12
        margin_top = 10
        margin_bottom = 26
        meter_area_h = height - margin_top - margin_bottom
        total_w = width - (margin_x * 2)
        col_w = total_w / n_stems

        bar_w = min(28.0, col_w * 0.6)

        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        for i, stem in enumerate(stems):
            cx = margin_x + i * col_w + col_w / 2.0
            bar_x = cx - bar_w / 2.0
            bar_y = margin_top + 16
            bar_h = meter_area_h - 22

            if bar_h <= 10:
                continue

            # Stem label at top
            color = self.STEM_COLORS[stem]
            painter.setPen(color)
            painter.drawText(
                QRectF(cx - col_w / 2.0, margin_top, col_w, 14),
                Qt.AlignmentFlag.AlignCenter,
                stem,
            )

            # Meter trough background
            trough_rect = QRectF(bar_x, bar_y, bar_w, bar_h)
            painter.setPen(QPen(QColor("#2d3139"), 1))
            painter.setBrush(QBrush(QColor("#16181d")))
            painter.drawRoundedRect(trough_rect, 3, 3)

            # Draw level fill
            lvl = self._levels[stem]
            if lvl > 0.001:
                fill_h = bar_h * lvl
                fill_rect = QRectF(bar_x + 1, bar_y + bar_h - fill_h, bar_w - 2, fill_h)

                gradient = QLinearGradient(
                    fill_rect.topLeft(), fill_rect.bottomLeft()
                )
                gradient.setColorAt(0.0, color)
                darker_color = QColor(color)
                darker_color.setAlpha(120)
                gradient.setColorAt(1.0, darker_color)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(gradient))
                painter.drawRoundedRect(fill_rect, 2, 2)

            # Peak tick line
            pk = self._peaks[stem]
            if pk > 0.01:
                pk_y = bar_y + bar_h - (bar_h * pk)
                painter.setPen(QPen(QColor("#ffffff"), 1.5))
                painter.drawLine(
                    QPointF(bar_x + 1, pk_y), QPointF(bar_x + bar_w - 1, pk_y)
                )

            # Readout value at bottom
            val_font = QFont()
            val_font.setPointSize(7)
            painter.setFont(val_font)
            painter.setPen(QColor("#abb2bf"))

            if lvl > 0.0001:
                db = 20.0 * np.log10(max(lvl, 1e-4))
                val_text = f"{db:.1f} dB"
            else:
                val_text = "-inf"

            painter.drawText(
                QRectF(cx - col_w / 2.0, bar_y + bar_h + 3, col_w, 14),
                Qt.AlignmentFlag.AlignCenter,
                val_text,
            )
