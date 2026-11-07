"""Concerto Standalone Desktop GUI Entrypoint."""

import sys
from concerto.gui.app import run_gui

if __name__ == "__main__":
    initial_file = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_gui(initial_file))
