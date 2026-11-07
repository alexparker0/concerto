#!/usr/bin/env python3
"""Build script for compiling Concerto Standalone GUI Executable on Windows."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_prerequisites() -> bool:
    print("[1/4] Checking build prerequisites...")
    required_modules = ["PySide6", "pyqtgraph", "sounddevice", "PyInstaller", "numpy", "scipy", "soundfile"]
    missing = []
    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print(f"ERROR: Missing required Python modules: {', '.join(missing)}")
        print("Run: pip install -r requirements-gui.txt")
        return False

    print("  All Python GUI & DSP dependencies present.")
    return True


def ensure_native_core() -> bool:
    print("[2/4] Checking native C++20 DSP core...")
    dll_path = REPO_ROOT / "native" / "build" / "bin" / "libconcerto_core.dll"
    if dll_path.exists():
        print(f"  Native DSP core located at: {dll_path}")
        return True

    print("  Native DSP core not found. Attempting to build with CMake...")
    cmake = shutil.which("cmake")
    if not cmake:
        print("  WARNING: cmake not found in PATH. Proceeding with Python DSP fallback.")
        return True

    build_dir = REPO_ROOT / "native" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(["cmake", "-S", "native", "-B", "native/build", "-DCMAKE_BUILD_TYPE=Release"], cwd=str(REPO_ROOT))
        subprocess.check_call(["cmake", "--build", "native/build"], cwd=str(REPO_ROOT))
        if dll_path.exists():
            print(f"  Successfully built native DSP core: {dll_path}")
            return True
        else:
            print("  WARNING: Native core build succeeded but DLL not found at expected path.")
    except Exception as e:
        print(f"  WARNING: Failed to build native core: {e}")
        print("  Proceeding with Python DSP fallback.")

    return True


def build_pyinstaller() -> bool:
    print("[3/4] Compiling standalone executable with PyInstaller...")
    spec_file = REPO_ROOT / "concerto_gui.spec"
    if not spec_file.exists():
        print(f"ERROR: PyInstaller spec not found: {spec_file}")
        return False

    cmd = [sys.executable, "-m", "PyInstaller", str(spec_file), "--noconfirm"]
    print(f"  Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"ERROR: PyInstaller compilation failed with code {result.returncode}")
        return False

    print("  PyInstaller compilation finished.")
    return True


def verify_dist() -> bool:
    print("[4/4] Verifying standalone distribution...")
    dist_dir = REPO_ROOT / "dist" / "concerto-gui"
    exe_path = dist_dir / "concerto-gui.exe"

    if not exe_path.exists():
        print(f"ERROR: Expected executable not found: {exe_path}")
        return False

    print(f"  SUCCESS! Standalone application built successfully:")
    print(f"    Executable: {exe_path}")
    print(f"    Distribution Folder: {dist_dir}")
    print()
    print("To launch the standalone app:")
    print(f"  {exe_path}")
    return True


def main() -> int:
    print("=" * 60)
    print(" Conceto Desktop GUI Standalone Packaging Builder")
    print("=" * 60)

    if not check_prerequisites():
        return 1
    if not ensure_native_core():
        return 1
    if not build_pyinstaller():
        return 1
    if not verify_dist():
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
