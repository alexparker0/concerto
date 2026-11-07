"""Input/output format support tests (WAV, Opus via FFmpeg fallback)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from concerto.audio.decode import decode_stereo
from concerto.audio.encode import encode_output
from concerto.audio.ffmpeg_io import ffmpeg_available
from concerto.audio.layouts import INTERNAL_5_1, INTERNAL_7_1
from concerto.validate.tests import detect_channel_frequencies, validate_file

SR = 48000
N = SR  # 1 s


@pytest.fixture(scope="module")
def stereo_src(tmp_path_factory):
    rng = np.random.default_rng(0)
    t = np.arange(N) / SR
    l = (0.3 * np.sin(2 * np.pi * 440 * t)
         + 0.05 * rng.standard_normal(N)).astype(np.float32)
    r = (0.3 * np.sin(2 * np.pi * 440 * t + 0.1)
         + 0.05 * rng.standard_normal(N)).astype(np.float32)
    base = tmp_path_factory.mktemp("fmt")
    src = base / "src.flac"
    sf.write(src, np.stack([l, r], 1), SR, format="FLAC", subtype="PCM_24")
    return str(src), l, r


def test_wav_input_decode(stereo_src):
    src, l, r = stereo_src
    wav = str(Path(src).parent / "src.wav")
    sf.write(wav, np.stack([l, r], 1), SR, format="WAV", subtype="PCM_24")
    buf, meta = decode_stereo(wav)
    assert meta.format == "WAV"
    assert buf.frame_count == N
    assert buf.sample_rate == SR
    assert np.allclose(buf.left, l, atol=1e-6)


def test_wav_output_channel_order(tmp_path, stereo_src):
    """WAVE 8ch places the back pair before the side pair — verify mapping."""
    import soundfile as _sf
    from concerto.audio.encode import encode_wav
    from concerto.validate.tests import routing_test_signal
    chans, labels = routing_test_signal(SR, 0.5)
    path = str(tmp_path / "routes.wav")
    encode_wav(chans, SR, INTERNAL_7_1.name, path, bit_depth=24)
    detected = detect_channel_frequencies(path)
    expected = {"FL": 1000, "FR": 2000, "FC": 3000, "LFE": 60,
                "SL": 4000, "SR": 5000, "BL": 6000, "BR": 7000}
    for label in labels:
        assert abs(detected[label] - expected[label]) < expected[label] * 0.05, label


def test_wav_output_validation(tmp_path, stereo_src):
    src, _, _ = stereo_src
    from concerto.dsp.upmix import UpmixRenderer, load_config
    buf, _ = decode_stereo(src)
    out = UpmixRenderer(SR, load_config("configs/default.toml")).render(
        buf.left, buf.right)
    path = str(tmp_path / "render.wav")
    encode_output(out.channels, SR, INTERNAL_7_1.name, path, bit_depth=24)
    result = validate_file(path, reference=src)
    assert result.passed, [c for c in result.checks if not c["passed"]]


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not on PATH")
def test_opus_input_decode(tmp_path, stereo_src):
    """Opus files decode through the FFmpeg fallback path."""
    src, l, r = stereo_src
    opus = tmp_path / "src.opus"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                    "-c:a", "libopus", "-b:a", "128k", str(opus)], check=True)
    buf, meta = decode_stereo(str(opus))
    # Opus decodes at 48 kHz natively; frame count ~1 s within resampler slop.
    # libsndfile >= 1.2 reads Opus natively (format "OGG"); the FFmpeg
    # fallback would report codec_name "opus" — both are valid paths.
    assert meta.format.lower() in ("opus", "ogg")
    assert abs(buf.frame_count / buf.sample_rate - 1.0) < 0.05
    assert buf.left.dtype == np.float32
    # Content correlation with the source (lossy but recognizable).
    n = min(buf.frame_count, N)
    src_l, _ = sf.read(src, dtype="float64", always_2d=True)
    src_l = src_l[:, 0]
    from scipy.signal import resample_poly
    import math
    g = math.gcd(buf.sample_rate, SR)
    ref = resample_poly(src_l, buf.sample_rate // g, SR // g)[:n]
    c = np.corrcoef(buf.left[:n].astype(np.float64), ref)[0, 1]
    assert c > 0.9


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not on PATH")
def test_mp3_input_decode(tmp_path, stereo_src):
    src, _, _ = stereo_src
    mp3 = tmp_path / "src.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                    "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)], check=True)
    buf, meta = decode_stereo(str(mp3))
    assert meta.format.lower() in ("mp3",)
    assert abs(buf.frame_count / buf.sample_rate - 1.0) < 0.1


def test_unsupported_output_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported output format"):
        encode_output([np.zeros(10, np.float32)] * 8, SR,
                      INTERNAL_7_1.name, str(tmp_path / "out.xyz"))


def test_tpdf_dither():
    from concerto.audio.encode import _dither_channels
    # Channels: FL (0), FR (1), FC (2, silent), LFE (3, silent), SL (4, active)
    chans = [
        np.ones(10000, dtype=np.float32) * 0.5,
        np.ones(10000, dtype=np.float32) * 0.5,
        np.zeros(10000, dtype=np.float32),
        np.zeros(10000, dtype=np.float32),
        np.ones(10000, dtype=np.float32) * 0.2,
    ]
    dithered = _dither_channels(chans, bit_depth=16)
    lsb = 1.0 / (1 << 15)
    # Front channels and silent LFE must remain untouched
    assert np.array_equal(dithered[0], chans[0])
    assert np.array_equal(dithered[1], chans[1])
    assert np.all(dithered[3] == 0.0)
    # Active surround channel must have dither noise bounded by 1 LSB
    diff = dithered[4] - chans[4]
    assert np.max(np.abs(diff)) <= lsb * 1.01
    assert abs(np.mean(diff)) < lsb * 0.1
