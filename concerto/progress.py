"""Terminal progress reporting.

Uses tqdm when available; falls back to plain percentage printing every ~5%.
Each stage creates a bar with a known total and reports via a callback:

    update, close = make_bar("render", total_chunks)
    for i in ...:
        ...
        update()
    close()
"""

from __future__ import annotations

import sys
from typing import Callable


def _noop() -> None:
    pass


def make_bar(desc: str, total: int) -> tuple[Callable[[], None], Callable[[], None]]:
    """Return (update, close) for a progress bar with `total` units."""
    try:
        from tqdm import tqdm
    except ImportError:
        pass
    else:
        bar = tqdm(total=total, desc=desc, unit="blk", dynamic_ncols=True,
                   leave=True)
        return bar.update, bar.close

    # Plain fallback: print every 5 %.
    state = {"done": 0, "printed": -5}

    def update() -> None:
        state["done"] += 1
        pct = int(100 * state["done"] / max(total, 1))
        if pct >= state["printed"] + 5 or state["done"] >= total:
            state["printed"] = pct
            end = "\n" if state["done"] >= total else ""
            print(f"  {desc}: {min(pct, 100):3d}%", end=end,
                  file=sys.stderr, flush=True)

    return update, _noop
