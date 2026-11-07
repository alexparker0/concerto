"""Channel layout definitions.

The internal channel order below is the single source of truth for the whole
application. Every file-format mapping is declared here explicitly and all
encode/decode paths go through :func:`to_internal` / :func:`from_internal`.
Never copy raw channels between a file and a processing buffer without this
mapping — file-format native orders differ (notably FLAC 6ch and FFmpeg 7.1).
"""

from __future__ import annotations

from dataclasses import dataclass

# Application-internal channel labels, in fixed order.
INTERNAL_51 = ("FL", "FR", "FC", "LFE", "SL", "SR")
INTERNAL_71 = ("FL", "FR", "FC", "LFE", "SL", "SR", "BL", "BR")


@dataclass(frozen=True)
class Layout:
    name: str
    order: tuple[str, ...]

    @property
    def channels(self) -> int:
        return len(self.order)

    def index(self, label: str) -> int:
        return self.order.index(label)


INTERNAL_5_1 = Layout("internal-5.1", INTERNAL_51)
INTERNAL_7_1 = Layout("internal-7.1", INTERNAL_71)

# Native channel orders of the file formats we write/read.
FLAC_6CH = Layout("flac-6ch", ("FL", "FR", "FC", "LFE", "BL", "BR"))
FLAC_8CH = Layout("flac-8ch", ("FL", "FR", "FC", "LFE", "SL", "SR", "BL", "BR"))

# WAVE (RBA) multichannel orders. WAVE 7.1 places the BACK pair BEFORE the
# SIDE pair — the opposite of FLAC 8ch. Per WAVE_FORMAT_EXTENSIBLE:
WAV_6CH = Layout("wav-6ch", ("FL", "FR", "FC", "LFE", "BL", "BR"))
WAV_8CH = Layout("wav-8ch", ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"))

# Positional view of a 6-channel FLAC file when written from internal 5.1:
# the file's surround pair (labelled BL/BR by the FLAC spec) carries our SL/SR.
FLAC_6CH_AS_51 = Layout("flac-6ch-as-5.1", ("FL", "FR", "FC", "LFE", "SL", "SR"))
# FFmpeg 7.1 = FL+FR+FC+LFE+BL+BR+SL+SR (back pair before side pair).
FFMPEG_71 = Layout("ffmpeg-7.1", ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"))
# FFmpeg/libav "5.1(side)" = FL+FR+FC+LFE+SL+SR (matches internal 5.1).
FFMPEG_51_SIDE = Layout("ffmpeg-5.1(side)", INTERNAL_51)


def internal_layout(name: str) -> Layout:
    """Resolve "5.1" / "7.1" to the internal Layout."""
    if name == "5.1":
        return INTERNAL_5_1
    if name == "7.1":
        return INTERNAL_7_1
    raise ValueError(f"unsupported layout: {name!r} (expected '5.1' or '7.1')")


def permutation(from_layout: Layout, to_layout: Layout) -> list[int]:
    """Index map so that ``to[k] = from[permutation[k]]``.

    Every label in ``to_layout`` must exist in ``from_layout``.
    """
    try:
        return [from_layout.index(ch) for ch in to_layout.order]
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"cannot map {from_layout.name} -> {to_layout.name}: {exc}"
        ) from exc


def reorder(buffer: "list[list[float]] | object", from_layout: Layout, to_layout: Layout):
    """Reorder channel-major buffer [ch][sample] between layouts."""
    perm = permutation(from_layout, to_layout)
    return [buffer[i] for i in perm]
