"""Tests for streaming ring-buffer audio I/O and streaming rendering."""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from concerto.audio.buffer import StereoRingBuffer
from concerto.audio.decode import decode_stereo, open_stereo_stream
from concerto.audio.encode import open_multichannel_stream, read_channels
from concerto.audio.layouts import INTERNAL_7_1, INTERNAL_5_1
from concerto.dsp.upmix import CHUNK, UpmixRenderer, load_config
from concerto.native import NativeUpmixRenderer, library_available
from concerto.adaptive import (
    AdaptiveBetaConfig,
    PolicyConfig,
    compute_adaptive_schedules,
    compute_adaptive_schedules_stream,
)


SR = 48000
N = SR * 2  # 2 seconds


@pytest.fixture
def synthetic_stereo_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1234)
    t = np.arange(N) / SR
    l = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.standard_normal(N)).astype(np.float32)
    r = (0.5 * np.sin(2 * np.pi * 880 * t) + 0.1 * rng.standard_normal(N)).astype(np.float32)
    p = tmp_path / "stereo_test.flac"
    sf.write(p, np.stack([l, r], axis=1), SR, format="FLAC", subtype="PCM_24")
    return p


def test_stereo_ring_buffer_wrap_around():
    capacity = 100
    rb = StereoRingBuffer(capacity)
    assert rb.capacity == 100
    assert rb.available == 0
    assert rb.space == 100

    # Write 60 samples
    l1 = np.arange(60, dtype=np.float32)
    r1 = np.arange(60, dtype=np.float32) + 100.0
    w = rb.write(l1, r1)
    assert w == 60
    assert rb.available == 60
    assert rb.space == 40

    # Read 40 samples
    out_l, out_r = rb.read(40)
    assert len(out_l) == 40
    assert np.array_equal(out_l, l1[:40])
    assert np.array_equal(out_r, r1[:40])
    assert rb.available == 20
    assert rb.space == 80

    # Write 70 samples (will wrap around end of buffer)
    l2 = np.arange(70, dtype=np.float32) + 200.0
    r2 = np.arange(70, dtype=np.float32) + 300.0
    w2 = rb.write(l2, r2)
    assert w2 == 70
    assert rb.available == 90
    assert rb.space == 10

    # Peek 30 samples (pointer must not advance)
    peek_l, peek_r = rb.peek(30)
    assert len(peek_l) == 30
    assert rb.available == 90

    # Read all remaining 90 samples
    all_l, all_r = rb.read(100)
    assert len(all_l) == 90
    assert rb.available == 0
    assert rb.space == 100
    # First 20 are the remainder of l1, next 70 are l2
    expected_l = np.concatenate([l1[40:], l2])
    expected_r = np.concatenate([r1[40:], r2])
    assert np.allclose(all_l, expected_l)
    assert np.allclose(all_r, expected_r)


def test_stereo_stream_reader_matches_decode_stereo(synthetic_stereo_file: Path):
    buf, meta = decode_stereo(str(synthetic_stereo_file))

    with open_stereo_stream(str(synthetic_stereo_file), chunk_size=4096) as reader:
        assert reader.sample_rate == buf.sample_rate
        assert reader.frame_count == buf.frame_count

        blocks_l = []
        blocks_r = []
        for l_chunk, r_chunk in reader.blocks():
            assert len(l_chunk) == len(r_chunk)
            assert len(l_chunk) <= 4096
            blocks_l.append(l_chunk)
            blocks_r.append(r_chunk)

        full_l = np.concatenate(blocks_l)
        full_r = np.concatenate(blocks_r)
        assert np.array_equal(full_l, buf.left)
        assert np.array_equal(full_r, buf.right)

        # Test rewind
        reader.seek(0)
        rewound_l = []
        for l_c, _ in reader.blocks():
            rewound_l.append(l_c)
        assert np.array_equal(np.concatenate(rewound_l), buf.left)


def test_multichannel_stream_writer(tmp_path: Path):
    out_flac = str(tmp_path / "stream_out.flac")
    sr = 48000
    n_frames = 8192
    chans = [np.ones(n_frames, dtype=np.float32) * (i + 1) * 0.1 for i in range(8)]

    with open_multichannel_stream(out_flac, sr, INTERNAL_7_1.name, bit_depth=24) as writer:
        writer.write_block(chans)
        assert writer.frames_written == n_frames
        peak = writer.close()

    assert peak > 0.0
    data, read_sr, n_ch = read_channels(out_flac)
    assert read_sr == sr
    assert n_ch == 8
    assert data.shape[0] == n_frames


def test_streaming_render_equivalence(synthetic_stereo_file: Path, tmp_path: Path):
    buf, _ = decode_stereo(str(synthetic_stereo_file))
    cfg = load_config("configs/default.toml")

    # In-memory render
    renderer_mem = UpmixRenderer(buf.sample_rate, cfg)
    out_mem = renderer_mem.render(buf.left, buf.right)

    # Streaming render
    out_stream_path = str(tmp_path / "stream_rendered.flac")
    renderer_stream = UpmixRenderer(buf.sample_rate, cfg)

    with open_stereo_stream(str(synthetic_stereo_file), chunk_size=CHUNK) as reader:
        with open_multichannel_stream(out_stream_path, reader.sample_rate, INTERNAL_7_1.name, bit_depth=24) as writer:
            for l_chunk, r_chunk in reader.blocks():
                blk = renderer_stream.render_block(l_chunk, r_chunk)
                writer.write_block(blk)

    # Read back and compare
    data, _, _ = read_channels(out_stream_path)
    # Compare with out_mem channels in file order
    from concerto.audio.layouts import FLAC_8CH, permutation
    perm = permutation(INTERNAL_7_1, FLAC_8CH)
    mem_file_order = np.stack([out_mem.channels[i] for i in perm], axis=1)

    # Within 24-bit PCM quantization slop
    diff = np.max(np.abs(data.astype(np.float64) - mem_file_order.astype(np.float64)))
    assert diff < 1e-4, f"Streaming render diverged: diff={diff}"


def test_adaptive_schedules_stream_equivalence(synthetic_stereo_file: Path):
    buf, _ = decode_stereo(str(synthetic_stereo_file))
    bcfg = AdaptiveBetaConfig()
    pcfg = PolicyConfig()

    # In-memory
    mem_sched = compute_adaptive_schedules(buf.left, buf.right, buf.sample_rate, bcfg, pcfg)

    # Streaming
    with open_stereo_stream(str(synthetic_stereo_file), chunk_size=CHUNK) as reader:
        stream_sched = compute_adaptive_schedules_stream(reader, buf.sample_rate, bcfg, pcfg)

    for k in ("beta", "gain", "decorr"):
        diff = np.max(np.abs(mem_sched[k] - stream_sched[k]))
        assert diff < 1e-6, f"Adaptive schedule {k} mismatch: {diff}"

def test_cli_render_stream(synthetic_stereo_file: Path, tmp_path: Path):
    from concerto.cli import main
    out_flac = str(tmp_path / "cli_stream_out.flac")
    rc = main(["render", str(synthetic_stereo_file), "--stream", "--mode", "adaptive", "--out", out_flac])
    assert rc == 0
    assert os.path.exists(out_flac)
    data, sr, n_ch = read_channels(out_flac)
    assert n_ch == 8
    assert sr == SR
