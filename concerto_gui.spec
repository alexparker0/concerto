# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for Concerto Standalone GUI Application."""

import os
import sys
from pathlib import Path

block_cipher = None

repo_root = Path(SPECPATH).resolve()

datas = [
    (str(repo_root / "configs"), "configs"),
]

binaries = []
dll_path = repo_root / "native" / "build" / "bin" / "libconcerto_core.dll"
if dll_path.exists():
    binaries.append((str(dll_path), "."))
    binaries.append((str(dll_path), "native/build/bin"))

hidden_imports = [
    "concerto",
    "concerto.cli",
    "concerto.native",
    "concerto.adaptive",
    "concerto.progress",
    "concerto.dsp.upmix",
    "concerto.dsp.correlation",
    "concerto.dsp.midside",
    "concerto.dsp.allpass",
    "concerto.dsp.rms",
    "concerto.audio.buffer",
    "concerto.audio.decode",
    "concerto.audio.encode",
    "concerto.audio.layouts",
    "concerto.audio.ffmpeg_io",
    "concerto.ai.separation",
    "concerto.ai.centre",
    "concerto.ai.harness",
    "concerto.validate.tests",
    "concerto.analysis.report",
    "concerto.gui",
    "concerto.gui.app",
    "concerto.gui.main_window",
    "concerto.gui.theme",
    "concerto.gui.workers",
    "concerto.gui.widgets",
    "concerto.gui.widgets.plots",
    "concerto.gui.widgets.speaker_layout",
    "concerto.gui.widgets.audio_player",
    "concerto.gui.widgets.log_viewer",
    "concerto.gui.tabs",
    "concerto.gui.tabs.render_tab",
    "concerto.gui.tabs.analysis_tab",
    "concerto.gui.tabs.validation_tab",
    "concerto.gui.tabs.benchmark_tab",
    "concerto.gui.tabs.ai_tab",
    "concerto.gui.tabs.routing_tab",
    "concerto.gui.tabs.config_tab",
    "concerto.gui.tabs.diagnostics_tab",
    "scipy",
    "scipy.signal",
    "scipy.special",
    "soundfile",
    "sounddevice",
    "pyqtgraph",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

a = Analysis(
    [str(repo_root / "run_gui.py")],
    pathex=[str(repo_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

onefile_mode = os.environ.get("CONCERTO_ONEFILE", "1") == "1"

if onefile_mode:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="concerto-gui",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,  # Windowed GUI application
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="concerto-gui",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="concerto-gui",
    )
