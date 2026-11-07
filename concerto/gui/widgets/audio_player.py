"""Concerto GUI - Audio Auditioning Player and Transport Bar.

Provides an interactive audio playback engine using sounddevice.OutputStream,
supporting original stereo, rendered multichannel (5.1/7.1), and real-time
MPEG fold-down headphone auditioning.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import threading
from typing import Sequence

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayer(QtCore.QObject):
    """Background audio playback engine using sounddevice.OutputStream.

    Supports:
    - Stereo, 5.1, and 7.1 audio buffers or files.
    - Audition modes:
      - 'stereo': first 2 channels.
      - 'multichannel': direct multichannel routing (if hardware supports it).
      - 'fold_down': real-time MPEG/ITU fold-down matrix to stereo headphones:
        L_down = FL + 0.707 * FC + 0.707 * SL + 0.5 * BL
        R_down = FR + 0.707 * FC + 0.707 * SR + 0.5 * BR
    """

    # Signals
    position_changed = QtCore.Signal(float, float)  # (current_time_s, total_duration_s)
    state_changed = QtCore.Signal(bool)            # is_playing
    error_signal = QtCore.Signal(str)             # error message string

    # Fold-down mixing coefficients
    G_CENTRE: float = 0.70710678
    G_SURROUND: float = 0.70710678
    G_BACK: float = 0.5

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)

        self._lock = threading.Lock()
        self._channels: np.ndarray | None = None  # shape: (n_channels, n_frames), float32
        self._sample_rate: int = 48000
        self._layout_name: str = "7.1"
        self._total_frames: int = 0
        self._duration_s: float = 0.0
        self._current_frame: int = 0

        self._is_playing: bool = False
        self._volume: float = 1.0
        self._audition_mode: str = "stereo"  # "stereo", "multichannel", "fold_down"

        self._stream: sd.OutputStream | None = None
        self._active_stream_channels: int = 0

        # UI position updater timer (~30 Hz)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_timer)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        """Return True if playback is currently active."""
        return self._is_playing

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self._duration_s

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            return float(self._current_frame) / float(self._sample_rate)

    @property
    def volume(self) -> float:
        """Playback volume multiplier in [0.0, 2.0]."""
        return self._volume

    @property
    def audition_mode(self) -> str:
        """Active audition mode ('stereo', 'multichannel', 'fold_down')."""
        return self._audition_mode

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    @property
    def channels_count(self) -> int:
        """Number of channels in the loaded audio buffer."""
        return self._channels.shape[0] if self._channels is not None else 0

    @property
    def layout_name(self) -> str:
        """Layout identifier string ('stereo', '5.1', '7.1')."""
        return self._layout_name

    # -------------------------------------------------------------------------
    # Audio Loading
    # -------------------------------------------------------------------------

    def load_audio(
        self,
        channels: Sequence[np.ndarray] | np.ndarray,
        sample_rate: int = 48000,
        layout_name: str = "7.1",
    ) -> None:
        """Load audio buffer into the player.

        Parameters
        ----------
        channels:
            List of 1D numpy arrays per channel, or 2D array [channels, frames]
            or [frames, channels].
        sample_rate:
            Sampling rate in Hz (default 48000).
        layout_name:
            Layout label ('stereo', '5.1', '7.1').
        """
        was_playing = self._is_playing
        if was_playing:
            self.stop()

        if isinstance(channels, (list, tuple)):
            if len(channels) == 0:
                data = np.zeros((2, 0), dtype=np.float32)
            else:
                arrs = [np.asarray(c, dtype=np.float32).ravel() for c in channels]
                max_len = max(len(c) for c in arrs)
                data = np.zeros((len(arrs), max_len), dtype=np.float32)
                for idx, c in enumerate(arrs):
                    data[idx, : len(c)] = c
        elif isinstance(channels, np.ndarray):
            arr = np.asarray(channels, dtype=np.float32)
            if arr.ndim == 1:
                data = arr[np.newaxis, :]  # (1, frames)
            elif arr.ndim == 2:
                # Disambiguate (channels, frames) vs (frames, channels)
                if arr.shape[0] <= 8 and arr.shape[1] > 8:
                    data = arr
                elif arr.shape[1] <= 8 and arr.shape[0] > 8:
                    data = arr.T
                else:
                    data = arr if arr.shape[0] <= arr.shape[1] else arr.T
            else:
                raise ValueError(f"Audio array must be 1D or 2D, got shape {arr.shape}")
        else:
            raise TypeError(f"Unsupported audio type: {type(channels)}")

        # Ensure contiguous float32 data
        data = np.ascontiguousarray(data, dtype=np.float32)

        with self._lock:
            self._channels = data
            self._sample_rate = int(sample_rate) if sample_rate > 0 else 48000
            self._layout_name = str(layout_name)
            self._total_frames = data.shape[1]
            self._duration_s = (
                float(self._total_frames) / float(self._sample_rate)
                if self._sample_rate > 0
                else 0.0
            )
            self._current_frame = 0

        # Close existing stream if sample rate or channel configuration changed
        self._close_stream()

        self.position_changed.emit(0.0, self._duration_s)

    def load_file(self, path: str | Path) -> bool:
        """Load an audio file (WAV, FLAC, AIFF, etc.) from disk.

        Returns True on success, False on error (emitting error_signal).
        """
        p = Path(path)
        if not p.is_file():
            self.error_signal.emit(f"Audio file not found: {p}")
            return False

        try:
            import soundfile as sf

            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            # data has shape (frames, channels)
            channels_t = data.T  # (channels, frames)
            num_channels = channels_t.shape[0]

            if num_channels == 8:
                layout = "7.1"
            elif num_channels == 6:
                layout = "5.1"
            elif num_channels == 2:
                layout = "stereo"
            else:
                layout = f"{num_channels}ch"

            self.load_audio(channels_t, sr, layout)
            return True
        except Exception as err:
            self.error_signal.emit(f"Failed to load audio file '{p}': {err}")
            return False

    # -------------------------------------------------------------------------
    # Playback Controls
    # -------------------------------------------------------------------------

    def play(self) -> None:
        """Start or resume playback."""
        if self._channels is None or self._total_frames == 0:
            return

        with self._lock:
            if self._current_frame >= self._total_frames:
                self._current_frame = 0

        # Determine output channels needed for current mode
        target_channels = self._determine_output_channels()

        # Open or re-open sounddevice stream if necessary
        if (
            self._stream is None
            or self._stream.closed
            or self._active_stream_channels != target_channels
            or self._stream.samplerate != self._sample_rate
        ):
            self._close_stream()
            try:
                self._stream = sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=target_channels,
                    callback=self._audio_callback,
                    blocksize=2048,
                    dtype="float32",
                )
                self._active_stream_channels = target_channels
            except Exception as err:
                self.error_signal.emit(f"Failed to open audio stream: {err}")
                self._is_playing = False
                self.state_changed.emit(False)
                return

        try:
            if not self._stream.active:
                self._stream.start()
            self._is_playing = True
            self._timer.start()
            self.state_changed.emit(True)
        except Exception as err:
            self.error_signal.emit(f"Failed to start audio playback: {err}")
            self._is_playing = False
            self.state_changed.emit(False)

    def pause(self) -> None:
        """Pause playback without resetting position."""
        with self._lock:
            self._is_playing = False

        if self._stream is not None and self._stream.active:
            try:
                self._stream.stop()
            except Exception:
                pass

        self._timer.stop()
        self.state_changed.emit(False)

    def stop(self) -> None:
        """Stop playback and reset position to 00:00."""
        with self._lock:
            self._is_playing = False
            self._current_frame = 0

        if self._stream is not None and self._stream.active:
            try:
                self._stream.stop()
            except Exception:
                pass

        self._timer.stop()
        self.position_changed.emit(0.0, self._duration_s)
        self.state_changed.emit(False)

    def seek(self, position_s: float) -> None:
        """Seek to position in seconds."""
        if self._sample_rate <= 0 or self._total_frames == 0:
            return

        pos_clamped = max(0.0, min(float(position_s), self._duration_s))
        target_frame = int(round(pos_clamped * self._sample_rate))
        target_frame = max(0, min(target_frame, self._total_frames))

        with self._lock:
            self._current_frame = target_frame

        self.position_changed.emit(pos_clamped, self._duration_s)

    def set_volume(self, volume: float) -> None:
        """Set output volume multiplier in range [0.0, 2.0]."""
        vol = max(0.0, min(float(volume), 2.0))
        with self._lock:
            self._volume = vol

    def set_audition_mode(self, mode: str) -> None:
        """Switch audition mode ('stereo', 'multichannel', 'fold_down')."""
        norm = mode.lower().strip().replace("-", "_").replace(" ", "_")
        if norm in ("stereo", "original_stereo"):
            norm = "stereo"
        elif norm in ("multichannel", "surround", "direct_multichannel"):
            norm = "multichannel"
        elif norm in ("fold_down", "folddown", "headphone_fold_down", "downmix"):
            norm = "fold_down"
        else:
            norm = "stereo"

        if norm == self._audition_mode:
            return

        with self._lock:
            self._audition_mode = norm

        # If switching requires a different number of output channels,
        # restart the stream cleanly
        target_channels = self._determine_output_channels()
        if self._stream is not None and self._active_stream_channels != target_channels:
            was_playing = self._is_playing
            if was_playing:
                self.pause()
            self._close_stream()
            if was_playing:
                self.play()

    def close(self) -> None:
        """Release audio stream and timer resources."""
        self.stop()
        self._close_stream()

    # -------------------------------------------------------------------------
    # Internal Audio Stream & Processing
    # -------------------------------------------------------------------------

    def _determine_output_channels(self) -> int:
        """Calculate necessary hardware output channels for current mode."""
        if self._audition_mode in ("stereo", "fold_down"):
            return 2

        if self._audition_mode == "multichannel":
            if self._channels is None:
                return 2

            requested = self._channels.shape[0]
            try:
                dev = sd.query_devices(kind="output")
                max_dev_channels = int(dev.get("max_output_channels", 2))
            except Exception:
                max_dev_channels = 2

            if max_dev_channels < requested:
                self.error_signal.emit(
                    f"Hardware output device only supports {max_dev_channels} channels "
                    f"({requested} required for multichannel). Falling back to fold-down stereo."
                )
                self._audition_mode = "fold_down"
                return 2

            return requested

        return 2

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._active_stream_channels = 0

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Real-time sounddevice callback executing on high-priority audio thread."""
        if not self._is_playing or self._channels is None:
            outdata.fill(0)
            return

        with self._lock:
            cur = self._current_frame
            total = self._total_frames
            if cur >= total:
                self._is_playing = False
                outdata.fill(0)
                return

            avail = total - cur
            n_frames = min(frames, avail)
            block = self._channels[:, cur : cur + n_frames]
            self._current_frame = cur + n_frames
            mode = self._audition_mode
            vol = self._volume

        out_ch = outdata.shape[1]
        num_ch = block.shape[0]

        if mode == "fold_down":
            # MPEG/ITU fold-down matrix:
            # L_down = FL + 0.707 * FC + 0.707 * SL + 0.5 * BL
            # R_down = FR + 0.707 * FC + 0.707 * SR + 0.5 * BR
            if num_ch >= 8:
                # 7.1: FL=0, FR=1, FC=2, LFE=3, SL=4, SR=5, BL=6, BR=7
                fl = block[0]
                fr = block[1]
                fc = block[2]
                sl = block[4]
                sr = block[5]
                bl = block[6]
                br = block[7]
                left = fl + self.G_CENTRE * fc + self.G_SURROUND * sl + self.G_BACK * bl
                right = fr + self.G_CENTRE * fc + self.G_SURROUND * sr + self.G_BACK * br
            elif num_ch >= 6:
                # 5.1: FL=0, FR=1, FC=2, LFE=3, SL=4, SR=5
                fl = block[0]
                fr = block[1]
                fc = block[2]
                sl = block[4]
                sr = block[5]
                left = fl + self.G_CENTRE * fc + self.G_SURROUND * sl
                right = fr + self.G_CENTRE * fc + self.G_SURROUND * sr
            elif num_ch >= 2:
                left = block[0]
                right = block[1]
            else:
                left = block[0]
                right = block[0]

            if out_ch >= 2:
                outdata[:n_frames, 0] = left * vol
                outdata[:n_frames, 1] = right * vol
                if out_ch > 2:
                    outdata[:n_frames, 2:].fill(0)
            elif out_ch == 1:
                outdata[:n_frames, 0] = 0.5 * (left + right) * vol

        elif mode == "multichannel":
            copy_ch = min(num_ch, out_ch)
            outdata[:n_frames, :copy_ch] = (block[:copy_ch, :].T) * vol
            if copy_ch < out_ch:
                outdata[:n_frames, copy_ch:].fill(0)

        else:  # "stereo" or default
            if num_ch >= 2:
                left = block[0]
                right = block[1]
            elif num_ch == 1:
                left = block[0]
                right = block[0]
            else:
                left = np.zeros(n_frames, dtype=np.float32)
                right = np.zeros(n_frames, dtype=np.float32)

            if out_ch >= 2:
                outdata[:n_frames, 0] = left * vol
                outdata[:n_frames, 1] = right * vol
                if out_ch > 2:
                    outdata[:n_frames, 2:].fill(0)
            elif out_ch == 1:
                outdata[:n_frames, 0] = 0.5 * (left + right) * vol

        # Pad remaining buffer with silence if near end of track
        if n_frames < frames:
            outdata[n_frames:, :].fill(0)

    @QtCore.Slot()
    def _on_timer(self) -> None:
        """Periodic GUI updater for playback position."""
        with self._lock:
            is_playing = self._is_playing
            cur = self._current_frame
            sr = self._sample_rate
            dur = self._duration_s

        cur_s = float(cur) / float(sr) if sr > 0 else 0.0
        self.position_changed.emit(cur_s, dur)

        if not is_playing:
            self.stop()


# -----------------------------------------------------------------------------
# JumpSlider Component
# -----------------------------------------------------------------------------

class JumpSlider(QtWidgets.QSlider):
    """QSlider subclass that jumps immediately to clicked mouse position."""

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            x_pos = (
                int(event.position().x())
                if hasattr(event, "position")
                else event.x()
            )
            val = QtWidgets.QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                x_pos,
                self.width(),
            )
            self.setValue(val)
        super().mousePressEvent(event)


# -----------------------------------------------------------------------------
# AudioTransportBar Widget
# -----------------------------------------------------------------------------

class AudioTransportBar(QtWidgets.QWidget):
    """Audition transport bar with playback controls, scrub slider, and audition mode."""

    def __init__(
        self,
        player: AudioPlayer | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if isinstance(player, QtWidgets.QWidget) and parent is None:
            parent = player
            player = None
        super().__init__(parent)

        self._player = player if player is not None else AudioPlayer(self)
        self._user_seeking = False

        self._setup_ui()
        self._connect_signals()

    @property
    def player(self) -> AudioPlayer:
        """Return the underlying AudioPlayer engine."""
        return self._player

    # -------------------------------------------------------------------------
    # Public Delegation API
    # -------------------------------------------------------------------------

    def load_audio(
        self,
        channels: Sequence[np.ndarray] | np.ndarray,
        sample_rate: int = 48000,
        layout_name: str = "7.1",
    ) -> None:
        """Delegate audio loading to the internal player."""
        self._player.load_audio(channels, sample_rate, layout_name)

    def load_file(self, path: str | Path) -> bool:
        """Delegate file loading to the internal player."""
        return self._player.load_file(path)

    def play(self) -> None:
        """Start audio playback."""
        self._player.play()

    def pause(self) -> None:
        """Pause audio playback."""
        self._player.pause()

    def stop(self) -> None:
        """Stop audio playback and reset scrub bar."""
        self._player.stop()

    def seek(self, position_s: float) -> None:
        """Seek audio position in seconds."""
        self._player.seek(position_s)

    def set_volume(self, volume: float) -> None:
        """Set volume in [0.0, 2.0]."""
        self._player.set_volume(volume)
        slider_val = int(round(min(1.0, max(0.0, volume)) * 100))
        self.slider_vol.blockSignals(True)
        self.slider_vol.setValue(slider_val)
        self.slider_vol.blockSignals(False)
        self.lbl_vol.setText(f"{slider_val}%")

    def set_audition_mode(self, mode: str) -> None:
        """Set audition mode ('stereo', 'multichannel', 'fold_down')."""
        self._player.set_audition_mode(mode)
        # Sync combo box
        for i in range(self.cmb_mode.count()):
            if self.cmb_mode.itemData(i) == self._player.audition_mode:
                self.cmb_mode.blockSignals(True)
                self.cmb_mode.setCurrentIndex(i)
                self.cmb_mode.blockSignals(False)
                break

    # -------------------------------------------------------------------------
    # UI Setup & Styling
    # -------------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setFixedHeight(54)
        self.setStyleSheet(
            """
            AudioTransportBar {
                background-color: #21252b;
                border-top: 1px solid #2d3139;
                border-bottom: 1px solid #2d3139;
            }
            """
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        # 1. Play / Pause Button
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.setFixedWidth(78)
        self.btn_play.setFixedHeight(30)
        self.btn_play.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #98c379;
                font-weight: bold;
                border: 1px solid #3e4451;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #353b45;
                color: #b0e38d;
                border-color: #98c379;
            }
            QPushButton:pressed {
                background-color: #21252b;
            }
            """
        )
        layout.addWidget(self.btn_play)

        # 2. Stop Button
        self.btn_stop = QtWidgets.QPushButton("⏹ Stop")
        self.btn_stop.setFixedWidth(68)
        self.btn_stop.setFixedHeight(30)
        self.btn_stop.setStyleSheet(
            """
            QPushButton {
                background-color: #2c313a;
                color: #e06c75;
                font-weight: bold;
                border: 1px solid #3e4451;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #353b45;
                color: #ff808a;
                border-color: #e06c75;
            }
            QPushButton:pressed {
                background-color: #21252b;
            }
            """
        )
        layout.addWidget(self.btn_stop)

        layout.addWidget(self._make_separator())

        # 3. Timestamp readout (00:00 / 03:45)
        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_time.setFixedWidth(104)
        self.lbl_time.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_time.setStyleSheet(
            "font-family: Consolas, Monaco, monospace; color: #abb2bf; font-size: 11px;"
        )
        layout.addWidget(self.lbl_time)

        # 4. Scrub / Seek Slider
        self.slider_seek = JumpSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.setValue(0)
        self.slider_seek.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #2d3139;
                height: 6px;
                background: #181a1f;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #61afef;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #abb2bf;
                border: 1px solid #5c6370;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #ffffff;
                border-color: #61afef;
            }
            """
        )
        layout.addWidget(self.slider_seek, 1)

        layout.addWidget(self._make_separator())

        # 5. Audition Mode Dropdown
        lbl_mode = QtWidgets.QLabel("Audition:")
        lbl_mode.setStyleSheet("color: #828997; font-size: 11px;")
        layout.addWidget(lbl_mode)

        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItem("Original Stereo", "stereo")
        self.cmb_mode.addItem("Rendered Surround", "multichannel")
        self.cmb_mode.addItem("Headphone Fold-Down", "fold_down")
        self.cmb_mode.setFixedHeight(28)
        self.cmb_mode.setMinimumWidth(150)
        self.cmb_mode.setStyleSheet(
            """
            QComboBox {
                background-color: #181a1f;
                color: #abb2bf;
                border: 1px solid #2d3139;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QComboBox:hover {
                border-color: #61afef;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
            QComboBox QAbstractItemView {
                background-color: #21252b;
                color: #abb2bf;
                selection-background-color: #3e4451;
                border: 1px solid #2d3139;
            }
            """
        )
        layout.addWidget(self.cmb_mode)

        layout.addWidget(self._make_separator())

        # 6. Volume Control
        self.lbl_vol_icon = QtWidgets.QLabel("🔊")
        self.lbl_vol_icon.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.lbl_vol_icon)

        self.slider_vol = JumpSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(80)
        self.slider_vol.setFixedWidth(80)
        self.slider_vol.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #2d3139;
                height: 5px;
                background: #181a1f;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #98c379;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #abb2bf;
                border: 1px solid #5c6370;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #ffffff;
                border-color: #98c379;
            }
            """
        )
        layout.addWidget(self.slider_vol)

        self.lbl_vol = QtWidgets.QLabel("80%")
        self.lbl_vol.setFixedWidth(34)
        self.lbl_vol.setStyleSheet(
            "font-family: Consolas, monospace; color: #828997; font-size: 10px;"
        )
        layout.addWidget(self.lbl_vol)

        # Set initial volume
        self._player.set_volume(0.8)

    @staticmethod
    def _make_separator() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #2d3139;")
        return line

    # -------------------------------------------------------------------------
    # Signals & Interactions
    # -------------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Player signals
        self._player.position_changed.connect(self._on_player_position_changed)
        self._player.state_changed.connect(self._on_player_state_changed)
        self._player.error_signal.connect(self._on_player_error)

        # Button signals
        self.btn_play.clicked.connect(self._on_toggle_play)
        self.btn_stop.clicked.connect(self.stop)

        # Seek slider signals
        self.slider_seek.sliderPressed.connect(self._on_seek_pressed)
        self.slider_seek.sliderMoved.connect(self._on_seek_moved)
        self.slider_seek.sliderReleased.connect(self._on_seek_released)

        # Combo mode signal
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_combo_changed)

        # Volume slider signal
        self.slider_vol.valueChanged.connect(self._on_volume_changed)

    @QtCore.Slot()
    def _on_toggle_play(self) -> None:
        if self._player.is_playing:
            self._player.pause()
        else:
            self._player.play()

    @QtCore.Slot(bool)
    def _on_player_state_changed(self, is_playing: bool) -> None:
        if is_playing:
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet(
                """
                QPushButton {
                    background-color: #2c313a;
                    color: #e5c07b;
                    font-weight: bold;
                    border: 1px solid #e5c07b;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #353b45;
                    color: #ffd88b;
                }
                """
            )
        else:
            self.btn_play.setText("▶ Play")
            self.btn_play.setStyleSheet(
                """
                QPushButton {
                    background-color: #2c313a;
                    color: #98c379;
                    font-weight: bold;
                    border: 1px solid #3e4451;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #353b45;
                    color: #b0e38d;
                    border-color: #98c379;
                }
                """
            )

    @QtCore.Slot(float, float)
    def _on_player_position_changed(self, current_s: float, total_s: float) -> None:
        if not self._user_seeking:
            ratio = (current_s / total_s) if total_s > 0 else 0.0
            slider_pos = int(round(ratio * 1000))
            self.slider_seek.blockSignals(True)
            self.slider_seek.setValue(slider_pos)
            self.slider_seek.blockSignals(False)
            self.lbl_time.setText(f"{self._format_time(current_s)} / {self._format_time(total_s)}")

    @QtCore.Slot()
    def _on_seek_pressed(self) -> None:
        self._user_seeking = True

    @QtCore.Slot(int)
    def _on_seek_moved(self, value: int) -> None:
        dur = self._player.duration
        target_s = (float(value) / 1000.0) * dur
        self.lbl_time.setText(f"{self._format_time(target_s)} / {self._format_time(dur)}")

    @QtCore.Slot()
    def _on_seek_released(self) -> None:
        dur = self._player.duration
        target_s = (float(self.slider_seek.value()) / 1000.0) * dur
        self._player.seek(target_s)
        self._user_seeking = False

    @QtCore.Slot(int)
    def _on_mode_combo_changed(self, index: int) -> None:
        mode_data = self.cmb_mode.itemData(index)
        if mode_data:
            self._player.set_audition_mode(mode_data)

    @QtCore.Slot(int)
    def _on_volume_changed(self, value: int) -> None:
        vol = float(value) / 100.0
        self._player.set_volume(vol)
        self.lbl_vol.setText(f"{value}%")
        if value == 0:
            self.lbl_vol_icon.setText("🔈")
        elif value < 50:
            self.lbl_vol_icon.setText("🔉")
        else:
            self.lbl_vol_icon.setText("🔊")

    @QtCore.Slot(str)
    def _on_player_error(self, message: str) -> None:
        logger.warning("AudioPlayer error: %s", message)
        # Sync combo mode if player automatically fell back to fold_down
        current_mode = self._player.audition_mode
        for i in range(self.cmb_mode.count()):
            if self.cmb_mode.itemData(i) == current_mode:
                self.cmb_mode.blockSignals(True)
                self.cmb_mode.setCurrentIndex(i)
                self.cmb_mode.blockSignals(False)
                break

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into MM:SS or HH:MM:SS string."""
        if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
            seconds = 0.0
        tot = int(round(seconds))
        hours = tot // 3600
        mins = (tot % 3600) // 60
        secs = tot % 60
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"
