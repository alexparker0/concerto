"""Surround sound speaker layout visualization widget.

Provides an interactive top-down acoustic room diagram for 5.1 and 7.1
surround layouts, displaying speaker geometry, real-time energy halos,
and channel solo/selection click events.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget


class SurroundLayoutWidget(QWidget):
    """Interactive top-down 5.1 / 7.1 surround loudspeaker geometry widget.

    Features:
    - Acoustic sweet spot listener orientation at center (0, 0).
    - Standard ITU-R angular positions for 7.1 (FL, FR, FC, LFE, SL, SR, BL, BR)
      and 5.1 (dimming BL and BR, angling surrounds).
    - Dynamic energy halo and speaker cone illumination based on channel RMS level.
    - Clickable speakers emitting channel identification signals for auditioning / soloing.
    """

    speaker_clicked = Signal(str)

    # Standard channel metadata
    CHANNEL_ORDER_71 = ("FL", "FR", "FC", "LFE", "SL", "SR", "BL", "BR")
    CHANNEL_ORDER_51 = ("FL", "FR", "FC", "LFE", "SL", "SR")

    CHANNEL_FULL_NAMES = {
        "FL": "Front Left",
        "FR": "Front Right",
        "FC": "Front Centre",
        "LFE": "Low Frequency Effects",
        "SL": "Side Left",
        "SR": "Side Right",
        "BL": "Back Left",
        "BR": "Back Right",
    }

    # Base speaker box dimensions
    SPEAKER_WIDTH = 48.0
    SPEAKER_HEIGHT = 38.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._layout: str = "7.1"
        self._levels: dict[str, float] = {ch: 0.0 for ch in self.CHANNEL_ORDER_71}
        self._hovered_speaker: str | None = None
        self._selected_speaker: str | None = None

        # Cache of current speaker bounding boxes for hit testing
        self._speaker_rects: dict[str, QRectF] = {}

    def set_layout(self, layout_name: str) -> None:
        """Set active layout ("5.1" or "7.1")."""
        norm = str(layout_name).strip().lower()
        if "5.1" in norm or "6ch" in norm or norm == "5":
            self._layout = "5.1"
        else:
            self._layout = "7.1"
        self.update()

    def get_layout(self) -> str:
        """Return active layout string ("5.1" or "7.1")."""
        return self._layout

    def set_channel_levels(self, levels: dict[str, float]) -> None:
        """Set RMS or peak levels for all channels (0.0 to 1.0).

        Parameters
        ----------
        levels: dict[str, float]
            Mapping of channel labels (e.g. 'FL', 'FC', 'SL') to levels in [0.0, 1.0].
        """
        for k, v in levels.items():
            upper = k.upper()
            if upper in self._levels:
                self._levels[upper] = float(np.clip(v, 0.0, 1.0))
        self.update()

    def set_channel_level(self, channel: str, level: float) -> None:
        """Set level for a single channel."""
        ch_upper = channel.upper()
        if ch_upper in self._levels:
            self._levels[ch_upper] = float(np.clip(level, 0.0, 1.0))
            self.update()

    def clear_levels(self) -> None:
        """Reset all channel levels to zero."""
        for ch in self._levels:
            self._levels[ch] = 0.0
        self.update()

    def get_selected_speaker(self) -> str | None:
        """Return currently selected/soloed speaker channel label."""
        return self._selected_speaker

    def set_selected_speaker(self, channel: str | None) -> None:
        """Set currently selected speaker channel label."""
        if channel is not None and channel.upper() in self.CHANNEL_ORDER_71:
            self._selected_speaker = channel.upper()
        else:
            self._selected_speaker = None
        self.update()

    def _get_speaker_angles(self) -> dict[str, tuple[float, float]]:
        """Return dict of {channel: (angle_degrees_from_front, radius_ratio)}."""
        # 0 deg is front (top), 90 deg is right, -90 deg is left
        if self._layout == "5.1":
            return {
                "FC": (0.0, 1.0),
                "FL": (-30.0, 1.0),
                "FR": (30.0, 1.0),
                "LFE": (20.0, 0.65),
                "SL": (-110.0, 1.0),
                "SR": (110.0, 1.0),
                "BL": (-145.0, 1.0),  # Dimmed/inactive
                "BR": (145.0, 1.0),  # Dimmed/inactive
            }
        else:
            return {
                "FC": (0.0, 1.0),
                "FL": (-30.0, 1.0),
                "FR": (30.0, 1.0),
                "LFE": (20.0, 0.65),
                "SL": (-95.0, 1.0),
                "SR": (95.0, 1.0),
                "BL": (-145.0, 1.0),
                "BR": (145.0, 1.0),
            }

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()

        # Canvas background
        painter.fillRect(0, 0, w, h, QColor("#181a1f"))

        cx = w / 2.0
        cy = h / 2.0
        # Compute layout radius leaving room for speaker boxes
        radius = min(w - 70, h - 70) / 2.0

        if radius < 30:
            return

        # Draw room acoustic boundary and distance rings
        self._draw_room_grid(painter, cx, cy, radius)

        # Draw listener head in center
        self._draw_listener_head(painter, cx, cy)

        # Compute and draw all speakers
        angles_map = self._get_speaker_angles()
        self._speaker_rects.clear()

        # Scale speaker box slightly if widget is small
        scale_factor = min(1.0, max(0.75, radius / 120.0))
        box_w = self.SPEAKER_WIDTH * scale_factor
        box_h = self.SPEAKER_HEIGHT * scale_factor

        for ch in self.CHANNEL_ORDER_71:
            deg, r_ratio = angles_map[ch]
            rad = np.radians(deg)
            r = radius * r_ratio

            sx = cx + r * np.sin(rad)
            sy = cy - r * np.cos(rad)

            rect = QRectF(sx - box_w / 2.0, sy - box_h / 2.0, box_w, box_h)
            self._speaker_rects[ch] = rect

            is_dimmed = self._layout == "5.1" and ch in ("BL", "BR")
            self._draw_speaker(painter, ch, rect, is_dimmed)

    def _draw_room_grid(
        self, painter: QPainter, cx: float, cy: float, radius: float
    ) -> None:
        """Draw circular acoustic boundary, distance rings, and reference axes."""
        # Outer room boundary
        painter.setPen(QPen(QColor("#24272e"), 1.5))
        painter.setBrush(QBrush(QColor("#15171b")))
        painter.drawEllipse(QPointF(cx, cy), radius + 22, radius + 22)

        # 50% distance ring
        dash_pen = QPen(QColor("#202329"), 1, Qt.PenStyle.DashLine)
        painter.setPen(dash_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius * 0.5, radius * 0.5)

        # Speaker distance circle
        outer_pen = QPen(QColor("#282c34"), 1, Qt.PenStyle.DotLine)
        painter.setPen(outer_pen)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Subtle crosshair
        axis_pen = QPen(QColor("#1f2227"), 1)
        painter.setPen(axis_pen)
        painter.drawLine(
            QPointF(cx, cy - radius - 15), QPointF(cx, cy + radius + 15)
        )
        painter.drawLine(
            QPointF(cx - radius - 15, cy), QPointF(cx + radius + 15, cy)
        )

        # Layout watermark label at top
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#5c6370"))
        painter.drawText(
            QRectF(cx - 50, cy - radius - 20, 100, 14),
            Qt.AlignmentFlag.AlignCenter,
            f"{self._layout} SURROUND",
        )

    def _draw_listener_head(self, painter: QPainter, cx: float, cy: float) -> None:
        """Draw top-down listener head with nose and ear orientation indicators."""
        head_radius = 16.0

        # Sweet spot outer acoustic halo
        painter.setPen(QPen(QColor("#61afef22"), 1))
        painter.setBrush(QBrush(QColor("#61afef0c")))
        painter.drawEllipse(QPointF(cx, cy), head_radius + 8, head_radius + 8)

        # Head circle
        painter.setPen(QPen(QColor("#abb2bf"), 1.5))
        painter.setBrush(QBrush(QColor("#21252b")))
        painter.drawEllipse(QPointF(cx, cy), head_radius, head_radius)

        # Left Ear
        painter.setBrush(QBrush(QColor("#282c34")))
        painter.drawRoundedRect(
            QRectF(cx - head_radius - 4, cy - 5, 4, 10), 2, 2
        )

        # Right Ear
        painter.drawRoundedRect(QRectF(cx + head_radius, cy - 5, 4, 10), 2, 2)

        # Nose pointer pointing straight forward towards Center channel (0 degrees)
        nose_path = QPainterPath()
        nose_path.moveTo(cx - 4.5, cy - head_radius + 2)
        nose_path.lineTo(cx, cy - head_radius - 7)
        nose_path.lineTo(cx + 4.5, cy - head_radius + 2)
        nose_path.closeSubpath()
        painter.setBrush(QBrush(QColor("#abb2bf")))
        painter.drawPath(nose_path)

        # Sweet spot label
        font = QFont()
        font.setPointSize(6)
        painter.setFont(font)
        painter.setPen(QColor("#5c6370"))
        painter.drawText(
            QRectF(cx - 20, cy - 4, 40, 10),
            Qt.AlignmentFlag.AlignCenter,
            "LISTENER",
        )

    def _draw_speaker(
        self,
        painter: QPainter,
        ch: str,
        rect: QRectF,
        is_dimmed: bool,
    ) -> None:
        """Render a single speaker box with cone rings, energy halo, and labels."""
        level = self._levels.get(ch, 0.0)
        is_hovered = self._hovered_speaker == ch and not is_dimmed
        is_selected = self._selected_speaker == ch and not is_dimmed

        cx = rect.center().x()
        cy = rect.center().y()
        w = rect.width()
        h = rect.height()

        # Dimmed mode opacity
        if is_dimmed:
            painter.setOpacity(0.20)
        else:
            painter.setOpacity(1.0)

        # 1. Energy Halo behind speaker when active
        if level > 0.01 and not is_dimmed:
            halo_size = min(w, h) * (0.4 + 0.6 * level)
            halo_rect = rect.adjusted(
                -halo_size, -halo_size, halo_size, halo_size
            )

            # Color shift: cyan -> green -> amber (near peak)
            if level > 0.8:
                halo_color = QColor("#e5c07b")
            elif ch == "LFE":
                halo_color = QColor("#c678dd")
            elif ch in ("FL", "FR", "FC"):
                halo_color = QColor("#61afef")
            else:
                halo_color = QColor("#98c379")

            halo_alpha = int(np.clip(160 * level, 20, 200))
            radial = QRadialGradient(rect.center(), halo_rect.width() / 2.0)
            c_center = QColor(halo_color)
            c_center.setAlpha(halo_alpha)
            c_outer = QColor(halo_color)
            c_outer.setAlpha(0)
            radial.setColorAt(0.0, c_center)
            radial.setColorAt(1.0, c_outer)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(radial))
            painter.drawEllipse(halo_rect)

        # 2. Speaker Box Enclosure
        box_color = QColor("#1e2227")
        if is_selected:
            border_pen = QPen(QColor("#61afef"), 2.0)
            box_color = QColor("#222831")
        elif is_hovered:
            border_pen = QPen(QColor("#abb2bf"), 1.5)
            box_color = QColor("#282c34")
        elif level > 0.01:
            active_color = (
                QColor("#c678dd") if ch == "LFE" else QColor("#61afef")
            )
            border_pen = QPen(active_color, 1.2)
        else:
            border_pen = QPen(QColor("#2d3139"), 1.0)

        painter.setPen(border_pen)
        painter.setBrush(QBrush(box_color))
        painter.drawRoundedRect(rect, 5.0, 5.0)

        # 3. Speaker Driver Cone / Subwoofer styling
        cone_radius = min(w, h) * 0.28
        cone_center = QPointF(cx - w * 0.20, cy)

        if ch == "LFE":
            # Subwoofer driver: heavier concentric circles
            painter.setPen(QPen(QColor("#3e4451"), 1.0))
            painter.setBrush(QBrush(QColor("#181a1f")))
            painter.drawEllipse(cone_center, cone_radius * 1.1, cone_radius * 1.1)

            # Dust cap / active woofer core
            core_color = (
                QColor("#c678dd") if level > 0.05 else QColor("#282c34")
            )
            painter.setBrush(QBrush(core_color))
            painter.drawEllipse(cone_center, cone_radius * 0.5, cone_radius * 0.5)
        else:
            # Standard speaker cone
            painter.setPen(QPen(QColor("#353a45"), 1.0))
            painter.setBrush(QBrush(QColor("#16181d")))
            painter.drawEllipse(cone_center, cone_radius, cone_radius)

            # Cone ring
            ring_color = (
                QColor("#61afef") if level > 0.05 else QColor("#2a2e37")
            )
            painter.setPen(QPen(ring_color, 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                cone_center, cone_radius * 0.65, cone_radius * 0.65
            )

            # Center dust cap
            cap_color = (
                QColor("#ffffff") if level > 0.05 else QColor("#3e4451")
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(cap_color))
            painter.drawEllipse(
                cone_center, cone_radius * 0.28, cone_radius * 0.28
            )

        # 4. Text labels: Channel name + Level readout
        text_rect = QRectF(cx - w * 0.05, cy - h * 0.45, w * 0.52, h * 0.90)

        font_label = QFont()
        font_label.setPointSize(int(min(w, h) * 0.21))
        font_label.setBold(True)
        painter.setFont(font_label)

        if is_selected:
            painter.setPen(QColor("#61afef"))
        elif level > 0.05:
            painter.setPen(QColor("#ffffff"))
        else:
            painter.setPen(QColor("#abb2bf"))

        # Channel label
        painter.drawText(
            QRectF(text_rect.left(), text_rect.top(), text_rect.width(), h * 0.5),
            Qt.AlignmentFlag.AlignCenter,
            ch,
        )

        # Level / status readout under channel name
        font_sub = QFont()
        font_sub.setPointSize(int(min(w, h) * 0.16))
        painter.setFont(font_sub)

        if is_dimmed:
            painter.setPen(QColor("#5c6370"))
            painter.drawText(
                QRectF(
                    text_rect.left(),
                    text_rect.top() + h * 0.45,
                    text_rect.width(),
                    h * 0.45,
                ),
                Qt.AlignmentFlag.AlignCenter,
                "off",
            )
        elif level > 0.01:
            db = 20.0 * np.log10(max(level, 1e-4))
            painter.setPen(
                QColor("#e5c07b") if level > 0.8 else QColor("#98c379")
            )
            painter.drawText(
                QRectF(
                    text_rect.left(),
                    text_rect.top() + h * 0.45,
                    text_rect.width(),
                    h * 0.45,
                ),
                Qt.AlignmentFlag.AlignCenter,
                f"{db:.0f}dB",
            )
        else:
            painter.setPen(QColor("#5c6370"))
            painter.drawText(
                QRectF(
                    text_rect.left(),
                    text_rect.top() + h * 0.45,
                    text_rect.width(),
                    h * 0.45,
                ),
                Qt.AlignmentFlag.AlignCenter,
                "---",
            )

        painter.setOpacity(1.0)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        hovered_ch: str | None = None

        for ch, rect in self._speaker_rects.items():
            if rect.contains(pos):
                if not (self._layout == "5.1" and ch in ("BL", "BR")):
                    hovered_ch = ch
                    break

        if hovered_ch != self._hovered_speaker:
            self._hovered_speaker = hovered_ch
            if hovered_ch is not None:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                full_name = self.CHANNEL_FULL_NAMES.get(hovered_ch, hovered_ch)
                lvl = self._levels.get(hovered_ch, 0.0)
                self.setToolTip(f"{full_name} ({hovered_ch})\nLevel: {lvl * 100:.1f}%")
            else:
                self.unsetCursor()
                self.setToolTip("")
            self.update()

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            clicked_ch: str | None = None

            for ch, rect in self._speaker_rects.items():
                if rect.contains(pos):
                    if not (self._layout == "5.1" and ch in ("BL", "BR")):
                        clicked_ch = ch
                        break

            if clicked_ch is not None:
                # Toggle selection or set clicked
                self._selected_speaker = clicked_ch
                self.speaker_clicked.emit(clicked_ch)
                self.update()
            else:
                # Clicked empty space clears selection
                if self._selected_speaker is not None:
                    self._selected_speaker = None
                    self.update()

        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hovered_speaker is not None:
            self._hovered_speaker = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)
