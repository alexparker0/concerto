"""Concerto GUI - Styled Log Viewer Widget.

Provides a dark, monospace console widget with color-coded log levels,
auto-scroll locking, clipboard copy, and file export.
"""

from __future__ import annotations

from datetime import datetime
import html
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


class StyledLogViewer(QtWidgets.QWidget):
    """Monospace activity log console with color-coded levels.

    Supported levels: 'info', 'warning', 'error', 'success'.
    Emits logs safely across threads using Qt queued signals.
    """

    _signal_append = QtCore.Signal(str, str)

    # Color palette
    LEVEL_COLORS = {
        "info": "#abb2bf",
        "warning": "#e5c07b",
        "warn": "#e5c07b",
        "error": "#e06c75",
        "err": "#e06c75",
        "success": "#98c379",
        "ok": "#98c379",
    }

    BADGE_COLORS = {
        "info": "#61afef",
        "warning": "#e5c07b",
        "warn": "#e5c07b",
        "error": "#e06c75",
        "err": "#e06c75",
        "success": "#98c379",
        "ok": "#98c379",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._signal_append.connect(self._append_entry)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header toolbar
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        toolbar.setSpacing(8)

        lbl_title = QtWidgets.QLabel("Console Output")
        lbl_title.setStyleSheet("font-weight: 600; color: #abb2bf; font-size: 11px;")
        toolbar.addWidget(lbl_title)

        toolbar.addStretch(1)

        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setStyleSheet(
            "QCheckBox { color: #828997; font-size: 11px; } "
            "QCheckBox::indicator { width: 13px; height: 13px; border-radius: 2px; } "
            "QCheckBox::indicator:checked { background-color: #61afef; border: 1px solid #61afef; } "
            "QCheckBox::indicator:unchecked { background-color: #181a1f; border: 1px solid #3e4451; }"
        )
        toolbar.addWidget(self.chk_autoscroll)

        self.btn_copy = QtWidgets.QPushButton("Copy")
        self.btn_copy.setToolTip("Copy log text to clipboard")
        self.btn_copy.setFixedHeight(22)
        self.btn_copy.setStyleSheet(self._button_style())
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        toolbar.addWidget(self.btn_copy)

        self.btn_export = QtWidgets.QPushButton("Export...")
        self.btn_export.setToolTip("Export log text to a file")
        self.btn_export.setFixedHeight(22)
        self.btn_export.setStyleSheet(self._button_style())
        self.btn_export.clicked.connect(self.export_to_file)
        toolbar.addWidget(self.btn_export)

        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.setToolTip("Clear all console output")
        self.btn_clear.setFixedHeight(22)
        self.btn_clear.setStyleSheet(self._button_style())
        self.btn_clear.clicked.connect(self.clear)
        toolbar.addWidget(self.btn_clear)

        layout.addLayout(toolbar)

        # Main text viewer
        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setReadOnly(True)
        self.editor.setMaximumBlockCount(10000)

        # Monospace font with fallbacks
        font = QtGui.QFont("Consolas", 10)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setFamilies(["Consolas", "Monaco", "Courier New", "Liberation Mono", "monospace"])
        self.editor.setFont(font)

        self.editor.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #181a1f;
                color: #abb2bf;
                border: 1px solid #2d3139;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #3e4451;
                selection-color: #ffffff;
            }
            QScrollBar:vertical {
                border: none;
                background: #181a1f;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3e4451;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #5c6370;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

        layout.addWidget(self.editor, 1)

    @staticmethod
    def _button_style() -> str:
        return (
            "QPushButton { "
            "background-color: #21252b; "
            "color: #abb2bf; "
            "border: 1px solid #2d3139; "
            "border-radius: 3px; "
            "padding: 2px 8px; "
            "font-size: 11px; "
            "} "
            "QPushButton:hover { "
            "background-color: #2c313a; "
            "border-color: #3e4451; "
            "color: #ffffff; "
            "} "
            "QPushButton:pressed { "
            "background-color: #181a1f; "
            "}"
        )

    # -------------------------------------------------------------------------
    # Public Logging API
    # -------------------------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        """Log a message with a specific severity level.

        Thread-safe: safely called from any background thread or main thread.
        Levels: 'info', 'warning', 'error', 'success'.
        """
        self._signal_append.emit(str(message), str(level))

    def log_info(self, msg: str) -> None:
        """Log an informational message (#abb2bf)."""
        self.log(msg, "info")

    def log_warning(self, msg: str) -> None:
        """Log a warning message (#e5c07b)."""
        self.log(msg, "warning")

    def log_error(self, msg: str) -> None:
        """Log an error message (#e06c75)."""
        self.log(msg, "error")

    def log_success(self, msg: str) -> None:
        """Log a success message (#98c379)."""
        self.log(msg, "success")

    def clear(self) -> None:
        """Clear all messages from the log viewer."""
        self.editor.clear()

    def toPlainText(self) -> str:
        """Return raw unformatted log contents as plain text."""
        return self.editor.toPlainText()

    def get_text(self) -> str:
        """Alias for toPlainText()."""
        return self.toPlainText()

    def set_auto_scroll(self, enabled: bool) -> None:
        """Enable or disable auto-scroll to latest log entries."""
        self.chk_autoscroll.setChecked(bool(enabled))

    def is_auto_scroll(self) -> bool:
        """Return True if auto-scroll is enabled."""
        return self.chk_autoscroll.isChecked()

    def copy_to_clipboard(self) -> None:
        """Copy current log content to the system clipboard."""
        text = self.editor.toPlainText()
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)

    def export_to_file(self, filepath: str = "") -> bool:
        """Export log contents to a file. Prompts file picker if path empty."""
        if not filepath:
            filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export Log Output",
                "concerto.log",
                "Log Files (*.log *.txt);;All Files (*)",
            )
        if not filepath:
            return False

        try:
            Path(filepath).write_text(self.editor.toPlainText(), encoding="utf-8")
            return True
        except Exception as err:
            self.log_error(f"Failed to export log file: {err}")
            return False

    # -------------------------------------------------------------------------
    # Internal Slots & Formatting
    # -------------------------------------------------------------------------

    @QtCore.Slot(str, str)
    def _append_entry(self, message: str, level: str) -> None:
        lvl_key = level.lower().strip()
        text_color = self.LEVEL_COLORS.get(lvl_key, "#abb2bf")
        badge_color = self.BADGE_COLORS.get(lvl_key, "#abb2bf")

        tag_name = lvl_key.upper()
        if tag_name == "WARN":
            tag_name = "WARNING"
        elif tag_name == "ERR":
            tag_name = "ERROR"
        elif tag_name == "OK":
            tag_name = "SUCCESS"

        time_str = datetime.now().strftime("%H:%M:%S")

        lines = str(message).splitlines()
        if not lines:
            lines = [""]

        # Format first line with timestamp and level badge
        first_line = html.escape(lines[0])
        html_chunks = [
            f'<span style="color: #5c6370;">[{time_str}]</span> '
            f'<span style="color: {badge_color}; font-weight: bold;">[{tag_name}]</span> '
            f'<span style="color: {text_color};">{first_line}</span>'
        ]

        # Multi-line indent alignment
        for subline in lines[1:]:
            escaped_sub = html.escape(subline)
            html_chunks.append(
                f'<span style="color: #5c6370;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> '
                f'<span style="color: {text_color};">{escaped_sub}</span>'
            )

        formatted_html = "<br/>".join(html_chunks)
        self.editor.appendHtml(formatted_html)

        if self.chk_autoscroll.isChecked():
            sb = self.editor.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
