"""Concerto CLI: analyse, render, validate, benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .analysis.report import write_reports
from .audio.buffer import measure
from .audio.decode import decode_stereo, open_stereo_stream
from .audio.encode import encode_flac, encode_output, open_multichannel_stream
from .audio.layouts import INTERNAL_5_1, INTERNAL_7_1
from .dsp.correlation import AnalysisSettings, band_average, correlation_report
from .dsp.upmix import CHUNK, derive_5_1_from_7_1, load_config
from .native import make_renderer
from .progress import make_bar  # noqa: F401 (used in cmd_render/cmd_benchmark)
from .validate.tests import detect_channel_frequencies, validate_file

DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "default.toml"


def cmd_analyse(args: argparse.Namespace) -> int:
    buf, meta = decode_stereo(args.input)
    settings = AnalysisSettings()
    csv_path, json_path = write_reports(buf.left, buf.right, meta, args.out, settings)
    print(f"CSV report:  {csv_path}")
    print(f"JSON report: {json_path}")
    return 0


def _maybe_separate(buf, args) -> "object | None":
    """Run AI separation when enabled. High cost - strictly flag-gated."""
    if args.ai != "on":
        return None
    try:
        from .ai.separation import separate_stems
    except Exception as e:
        print(f"AI unavailable: {e}", file=sys.stderr)
        return None
    t0 = time.perf_counter()
    onnx_path = getattr(args, "onnx_model", None)
    use_cache = not getattr(args, "no_cache", False)
    stems = separate_stems(buf.left, buf.right, buf.sample_rate, progress=True,
                           use_cache=use_cache, onnx_model_path=onnx_path)
    dt = time.perf_counter() - t0
    print(f"AI separation: {dt:.1f} s "
          f"({buf.frame_count / buf.sample_rate / dt:.2f}x realtime)")
    return stems


def cmd_render(args: argparse.Namespace) -> int:
    if args.config:
        cfg = load_config(args.config)
    elif args.mode == "adaptive":
        # --ai on implies the AI profile (vocal centre + adaptive beta) when
        # the user did not choose a config explicitly. Falls back to the
        # plain adaptive profile if the AI one is absent.
        ai_cfg = Path(__file__).resolve().parent.parent / "configs" / "adaptive-ai.toml"
        base_cfg = Path(__file__).resolve().parent.parent / "configs" / "adaptive.toml"
        cfg = load_config(ai_cfg if args.ai == "on" and ai_cfg.exists() else base_cfg)
    else:
        cfg = load_config(DEFAULT_CFG)
    if args.layout:
        cfg["layout"] = args.layout
    if args.mode == "baseline":
        cfg["adaptive"] = False
        cfg["adaptive_beta"] = False
    if args.ai != "auto":
        pass  # AI is post-Milestone-1; accepted for CLI compatibility, no-op

    buf, meta = decode_stereo(args.input)
    renderer = make_renderer(buf.sample_rate, cfg)

    if getattr(args, "stream", False) and args.ai == "off":
        reader = open_stereo_stream(args.input, chunk_size=CHUNK)
        renderer = make_renderer(reader.sample_rate, cfg)
        layout_name = INTERNAL_7_1.name if cfg["layout"] == "7.1" else INTERNAL_5_1.name
        is_51 = cfg["layout"] != "7.1"
        dither = getattr(args, "dither", False) or bool(cfg.get("dither", False))

        schedules = None
        if args.mode == "adaptive" and cfg.get("adaptive_beta"):
            from .adaptive import (AdaptiveBetaConfig, PolicyConfig,
                                   compute_adaptive_schedules_stream)
            bcfg = AdaptiveBetaConfig(
                enabled=True,
                beta_min=float(cfg.get("beta_min", 0.15)),
                beta_max=float(cfg.get("beta_max", 0.90)),
                tau_s=float(cfg.get("beta_tau_s", 1.5)),
            )
            pcfg = PolicyConfig(tau_s=float(cfg.get("transition_tau_s", 1.5)),
                                transient_tau_s=float(cfg.get("transient_tau_s", 0.05)))
            n_chunks = (reader.frame_count + CHUNK - 1) // CHUNK if reader.frame_count > 0 else 1
            upd, close = make_bar("schedules", n_chunks)
            schedules = compute_adaptive_schedules_stream(
                reader, reader.sample_rate, bcfg, pcfg, progress=lambda total: upd)
            close()
            b, g, dc = schedules["beta"], schedules["gain"], schedules["decorr"]
            print(f"adaptive schedules over {len(b)} blocks: "
                  f"beta {b.min():.3f}-{b.max():.3f} (mean {b.mean():.3f}), "
                  f"gain x{g.min():.2f}-{g.max():.2f} (mean {g.mean():.2f}), "
                  f"decorr {dc.min():.2f}-{dc.max():.2f} (mean {dc.mean():.2f})")

        with open_multichannel_stream(args.out, reader.sample_rate, layout_name,
                                      bit_depth=cfg["bit_depth"], dither=dither) as writer:
            reader.seek(0)
            n_chunks = (reader.frame_count + CHUNK - 1) // CHUNK if reader.frame_count > 0 else 1
            upd, close = make_bar("render (stream)", n_chunks)
            ci = 0
            for l_chunk, r_chunk in reader.blocks():
                b = schedules["beta"][ci] if schedules is not None and ci < len(schedules["beta"]) else None
                g = schedules["gain"][ci] if schedules is not None and ci < len(schedules["gain"]) else None
                d = schedules["decorr"][ci] if schedules is not None and ci < len(schedules["decorr"]) else None
                blk = renderer.render_block(l_chunk, r_chunk, beta=b, gain=g, decorr=d)
                if is_51:
                    from .dsp.upmix import derive_5_1_from_7_1_block
                    blk = derive_5_1_from_7_1_block(blk)
                writer.write_block(blk)
                ci += 1
                upd()
            close()
            peak = writer.peak
            frames = writer.frames_written
        reader.close()
        print(f"Wrote {args.out}: {layout_name}, {renderer.sr} Hz, {frames} frames, peak={peak:.4f}")
        if peak > 1.0:
            print("WARNING: output peak exceeds 1.0 — reduce surround gain", file=sys.stderr)
            return 2
        return 0

    stems = _maybe_separate(buf, args)
    if stems is not None:
        from .ai.harness import HarnessConfig, evaluate_stems
        rep = evaluate_stems(buf.left, buf.right, stems, buf.sample_rate)
        if not rep.passed:
            # Advisory only: the render continues. Stem quality problems
            # affect the vocal-centre contribution, not the DSP mix.
            print("WARNING: AI acceptance criteria not met - stems kept:",
                  file=sys.stderr)
            for c in rep.checks:
                if not c["passed"]:
                    print(f"  {c['check']}: {c['detail']}", file=sys.stderr)
        else:
            print("AI acceptance criteria passed:",
                  rep.metrics.get("recombination_db"), "dB recombination")
    schedules = None
    if args.mode == "adaptive" and cfg.get("adaptive_beta"):
        from .adaptive import (AdaptiveBetaConfig, PolicyConfig,
                               compute_adaptive_schedules)
        bcfg = AdaptiveBetaConfig(
            enabled=True,
            beta_min=float(cfg.get("beta_min", 0.15)),
            beta_max=float(cfg.get("beta_max", 0.90)),
            tau_s=float(cfg.get("beta_tau_s", 1.5)),
        )
        pcfg = PolicyConfig(tau_s=float(cfg.get("transition_tau_s", 1.5)),
                            transient_tau_s=float(cfg.get("transient_tau_s", 0.05)))
        upd, close = make_bar("schedules", (buf.frame_count + CHUNK - 1) // CHUNK)
        schedules = compute_adaptive_schedules(
            buf.left, buf.right, buf.sample_rate, bcfg, pcfg,
            progress=lambda total: upd)
        close()
        b, g, dc = schedules["beta"], schedules["gain"], schedules["decorr"]
        print(f"adaptive schedules over {len(b)} blocks: "
              f"beta {b.min():.3f}-{b.max():.3f} (mean {b.mean():.3f}), "
              f"gain x{g.min():.2f}-{g.max():.2f} (mean {g.mean():.2f}), "
              f"decorr {dc.min():.2f}-{dc.max():.2f} (mean {dc.mean():.2f})")

    v_schedule = None
    if stems is not None and cfg.get("centre_mode") == "vocal_mask":
        from .ai.harness import v_schedule_from_stems
        v_schedule = v_schedule_from_stems(
            stems.vocals[0], stems.vocals[1], buf.left, buf.right,
            buf.sample_rate, CHUNK)
        print(f"vocal gate: max={v_schedule.max():.2f} mean={v_schedule.mean():.2f}")

    if cfg["layout"] == "7.1":
        upd, close = make_bar("render", (buf.frame_count + CHUNK - 1) // CHUNK)
        out = renderer.render(buf.left, buf.right,
                              beta_schedule=(schedules or {}).get("beta"),
                              gain_schedule=(schedules or {}).get("gain"),
                              decorr_schedule=(schedules or {}).get("decorr"),
                              progress=lambda total: upd)
        close()
        layout_name = INTERNAL_7_1.name
    else:
        upd, close = make_bar("render", (buf.frame_count + CHUNK - 1) // CHUNK)
        primary = renderer.render(  # 7.1 primary representation
            buf.left, buf.right,
            beta_schedule=(schedules or {}).get("beta"),
            gain_schedule=(schedules or {}).get("gain"),
            decorr_schedule=(schedules or {}).get("decorr"),
            progress=lambda total: upd)
        close()
        out = derive_5_1_from_7_1(primary)
        layout_name = INTERNAL_5_1.name

    if v_schedule is not None:
        from .ai.centre import apply_vocal_centre
        updc, uclose = make_bar("vocal centre", (buf.frame_count + CHUNK - 1) // CHUNK)
        out = apply_vocal_centre(out, stems.vocals[0], stems.vocals[1],
                                 v_schedule,
                                 gain_db=float(cfg.get("vocal_gain_db", -6.0)),
                                 progress=lambda total: updc)
        uclose()

    dither = getattr(args, "dither", False) or bool(cfg.get("dither", False))
    encode_output(out.channels, buf.sample_rate, layout_name, args.out,
                  bit_depth=cfg["bit_depth"], dither=dither)
    peak = out.peak()
    print(f"Wrote {args.out}: {layout_name}, {buf.sample_rate} Hz, "
          f"{out.frame_count} frames, peak={peak:.4f}")
    if peak > 1.0:
        print("WARNING: output peak exceeds 1.0 — reduce surround gain", file=sys.stderr)
        return 2
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_file(args.input, reference=args.reference)
    print(result.to_json())
    return 0 if result.passed else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    buf, meta = decode_stereo(args.input)
    results = {}
    repo_root = Path(__file__).resolve().parent.parent
    for profile in profiles:
        prof_clean = profile.lower().strip()
        cfg_filename = "default.toml" if prof_clean in ("baseline", "default") else f"{prof_clean}.toml"
        candidates = [
            Path("configs") / cfg_filename,
            repo_root / "configs" / cfg_filename,
            Path(cfg_filename),
        ]
        cfg_path = candidates[0]
        for c in candidates:
            if c.is_file():
                cfg_path = c
                break
        cfg = load_config(cfg_path if cfg_path.is_file() else DEFAULT_CFG)
        cfg["layout"] = args.layout
        is_adaptive = prof_clean in ("adaptive", "adaptive-ai")
        cfg["adaptive"] = is_adaptive
        cfg["adaptive_beta"] = is_adaptive
        if prof_clean == "baseline":
            cfg["centre_mode"] = "disabled"
        elif prof_clean == "adaptive-ai":
            cfg["centre_mode"] = "vocal_mask"

        schedules = None
        if is_adaptive and cfg.get("adaptive_beta"):
            from .adaptive import (AdaptiveBetaConfig, PolicyConfig,
                                   compute_adaptive_schedules)
            bcfg = AdaptiveBetaConfig(
                enabled=True,
                beta_min=float(cfg.get("beta_min", 0.15)),
                beta_max=float(cfg.get("beta_max", 0.90)),
                tau_s=float(cfg.get("beta_tau_s", 1.5)),
            )
            pcfg = PolicyConfig(
                tau_s=float(cfg.get("transition_tau_s", 1.5)),
                transient_tau_s=float(cfg.get("transient_tau_s", 0.05)),
            )
            schedules = compute_adaptive_schedules(
                buf.left, buf.right, buf.sample_rate, bcfg, pcfg
            )

        renderer = make_renderer(buf.sample_rate, cfg)
        primary = renderer.render(
            buf.left, buf.right,
            beta_schedule=(schedules or {}).get("beta"),
            gain_schedule=(schedules or {}).get("gain"),
            decorr_schedule=(schedules or {}).get("decorr"),
        )
        out = primary if cfg["layout"] == "7.1" else derive_5_1_from_7_1(primary)
        out_path = Path(args.out) / f"{Path(args.input).stem}_{profile}.flac"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encode_flac(out.channels, buf.sample_rate, out.layout_name, str(out_path),
                    bit_depth=cfg["bit_depth"])

        idx = {lbl: i for i, lbl in enumerate(
            list(INTERNAL_7_1.order) if out.n_channels == 8 else list(INTERNAL_5_1.order))}
        sl_ch = out.channels[idx["SL"]]
        sr_ch = out.channels[idx["SR"]]
        fl_ch = out.channels[idx["FL"]]
        _, sr_rms = measure(sr_ch)
        _, l_rms = measure(fl_ch)

        rep = correlation_report(sl_ch, sr_ch, buf.sample_rate, AnalysisSettings())
        bands = band_average(rep)
        results[profile] = {
            "output": str(out_path),
            "peak": round(out.peak(), 6),
            "surround_rms": round(sr_rms, 6),
            "front_rms": round(l_rms, 6),
            "stereo_bands": [{"band": b, "rho": round(r, 4), "coherence": round(c, 4)}
                             for b, r, c in bands],
        }
        print(f"{profile}: {out_path} peak={out.peak():.4f} "
              f"surround_RMS={sr_rms:.4f}")
    summary = Path(args.out) / f"{Path(args.input).stem}_benchmark.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"Benchmark summary: {summary}")
    return 0


def cmd_routing_test(args: argparse.Namespace) -> int:
    """Self-test: render a labelled 8-channel routing file and verify it."""
    from .validate.tests import routing_test_signal

    sr = 48000
    chans, labels = routing_test_signal(sr, seconds=args.seconds)
    out_path = args.out
    encode_flac(chans, sr, INTERNAL_7_1.name, out_path, bit_depth=24)
    detected = detect_channel_frequencies(out_path)
    expected = {"FL": 1000.0, "FR": 2000.0, "FC": 3000.0, "LFE": 60.0,
                "SL": 4000.0, "SR": 5000.0, "BL": 6000.0, "BR": 7000.0}
    ok = True
    for label in labels:
        exp, got = expected[label], detected[label]
        match = abs(got - exp) < exp * 0.05
        ok &= match
        print(f"  {label}: expected {exp:.0f} Hz, detected {got:.0f} Hz "
              f"{'OK' if match else 'MISMATCH'}")
    print("Channel routing: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_ai_eval(args: argparse.Namespace) -> int:
    """Run AI separation + acceptance harness only (no render)."""
    buf, meta = decode_stereo(args.input)
    try:
        from .ai.harness import evaluate_stems
        from .ai.separation import separate_stems
    except Exception as e:
        print(f"AI unavailable: {e}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    stems = separate_stems(buf.left, buf.right, buf.sample_rate, progress=True)
    dt = time.perf_counter() - t0
    print(f"AI separation: {dt:.1f} s "
          f"({buf.frame_count / buf.sample_rate / dt:.2f}x realtime)")
    rep = evaluate_stems(buf.left, buf.right, stems, buf.sample_rate)
    print(rep.to_json())
    out = Path(args.out) / f"{Path(args.input).stem}_ai_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.to_json())
    print(f"Report: {out}")
    return 0 if rep.passed else 1

def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the interactive Concerto desktop GUI workstation."""
    from .gui.app import run_gui
    initial = getattr(args, "file", None) or getattr(args, "file_flag", None)
    return run_gui(initial_file=initial)

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="concerto", description=__doc__)
    p.add_argument("--version", action="version", version=f"concerto {__version__}")
    if argv is None and len(sys.argv) == 1:
        try:
            from .gui.app import run_gui
            return run_gui()
        except Exception:
            p.print_help()
            return 0

    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gui", help="launch interactive Concerto desktop workstation")
    g.add_argument("file", nargs="?", default=None, help="optional initial audio file to load")
    g.add_argument("--file", "-f", dest="file_flag", default=None, help="optional initial audio file to load (flag form)")
    g.set_defaults(func=cmd_gui)

    a = sub.add_parser("analyse", help="analyse a stereo recording")
    a.add_argument("input")
    a.add_argument("--out", default="reports")
    a.set_defaults(func=cmd_analyse)

    r = sub.add_parser("render", help="render a surround mix")
    r.add_argument("input")
    r.add_argument("--layout", choices=["5.1", "7.1"], default=None)
    r.add_argument("--mode", choices=["baseline", "adaptive"], default="baseline")
    r.add_argument("--ai", choices=["off", "on"], default="off")
    r.add_argument("--onnx-model", default=None, help="path to ONNX stem model (Intel Arc / DirectML accelerated)")
    r.add_argument("--no-cache", action="store_true", default=False, help="disable stem caching")
    r.add_argument("--dither", action="store_true", default=False, help="apply TPDF dither before integer quantization")
    r.add_argument("--stream", action="store_true", default=False, help="stream audio via ring buffer to bound RAM usage to O(chunk)")
    r.add_argument("--config", default=None)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_render)

    v = sub.add_parser("validate", help="validate an output file")
    v.add_argument("input")
    v.add_argument("--reference", default=None)
    v.set_defaults(func=cmd_validate)

    b = sub.add_parser("benchmark", help="compare profiles")
    b.add_argument("input")
    b.add_argument("--profiles", default="baseline,adaptive")
    b.add_argument("--layout", choices=["5.1", "7.1"], default="7.1")
    b.add_argument("--out", default="reports")
    b.set_defaults(func=cmd_benchmark)

    e = sub.add_parser("ai-eval", help="AI separation + acceptance harness")
    e.add_argument("input")
    e.add_argument("--out", default="reports")
    e.set_defaults(func=cmd_ai_eval)

    t = sub.add_parser("routing-test", help="channel-order routing self-test")
    t.add_argument("--out", default="routing_test.flac")
    t.add_argument("--seconds", type=float, default=1.0)
    t.set_defaults(func=cmd_routing_test)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
