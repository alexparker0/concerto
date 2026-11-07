"""Qt QThread background workers for Concerto GUI operations.

Ensures the UI thread remains responsive during DSP rendering, analysis,
validation, benchmarking, AI separation, and native core builds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..analysis.report import write_reports
from ..audio.buffer import measure
from ..audio.decode import decode_stereo, open_stereo_stream
from ..audio.encode import encode_flac, encode_output, open_multichannel_stream
from ..audio.layouts import INTERNAL_5_1, INTERNAL_7_1
from ..dsp.correlation import AnalysisSettings, band_average, correlation_report
from ..dsp.upmix import CHUNK, derive_5_1_from_7_1, load_config
from ..native import make_renderer
from ..validate.tests import (detect_channel_frequencies, routing_test_signal,
                              validate_file)


class RenderWorker(QThread):
    """Asynchronous DSP and AI rendering worker."""

    progress_signal = Signal(int, int, str)       # current, total, stage_name
    metrics_signal = Signal(float, float, float, float)  # speed_rt, elapsed_s, eta_s, peak
    status_signal = Signal(str)                  # human readable status message
    finished_signal = Signal(dict)               # dict summary: output, peak, frames, duration_s
    error_signal = Signal(str)                   # error message if failed

    def __init__(
        self,
        input_path: str,
        output_path: str,
        cfg: dict | None = None,
        mode: str = "baseline",
        layout: str = "7.1",
        ai: str = "off",
        onnx_model: str | None = None,
        stream: bool = False,
        dither: bool = False,
        no_cache: bool = False,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self.cfg = cfg or {}
        self.mode = mode
        self.layout = layout
        self.ai = ai
        self.onnx_model = onnx_model
        self.stream = stream
        self.dither = dither
        self.no_cache = no_cache
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        t_start = time.perf_counter()
        try:
            defaults = {
                "layout": "7.1", "bit_depth": 24, "preserve_front": True,
                "surround_gain": 0.25, "extraction": "stereo_difference", "stereo_beta": 0.25,
                "highpass_hz": 150.0, "lowpass_hz": 8000.0, "filter_order": 4,
                "target_rms_ratio": 0.35, "rms_window_s": 0.4, "gain_tau_s": 2.0,
                "gain_min": 0.0, "gain_max": 2.0, "decorrelate": True,
                "coeffs": {
                    "SL": [0.68, -0.41, 0.23], "SR": [-0.55, 0.37, -0.19],
                    "BL": [0.31, -0.62, 0.44], "BR": [-0.29, 0.48, -0.57],
                },
                "centre_mode": "disabled", "mid_fill_gain": 0.3548, "vocal_gain_db": -6.0,
                "adaptive": False, "adaptive_beta": False, "beta_min": 0.15,
                "beta_max": 0.90, "beta_tau_s": 1.5,
            }
            try:
                base_p = Path("configs/default.toml")
                if not base_p.exists():
                    base_p = Path(__file__).resolve().parent.parent.parent / "configs" / "default.toml"
                defaults.update(load_config(base_p))
            except Exception:
                pass

            cfg = dict(defaults)
            if self.cfg:
                cfg.update(self.cfg)

            cfg["layout"] = self.layout
            if self.mode == "baseline":
                cfg["adaptive"] = False
                cfg["adaptive_beta"] = False

            dither = self.dither or bool(cfg.get("dither", False))
            self.status_signal.emit(f"Starting render: {Path(self.input_path).name}")

            # -------------------------------------------------------------
            # STREAMING PATH (Ring-buffer, AI off)
            # -------------------------------------------------------------
            if self.stream and self.ai == "off":
                self.status_signal.emit("Opening stereo stream...")
                reader = open_stereo_stream(self.input_path, chunk_size=CHUNK)
                renderer = make_renderer(reader.sample_rate, cfg)
                layout_name = INTERNAL_7_1.name if cfg["layout"] == "7.1" else INTERNAL_5_1.name
                is_51 = cfg["layout"] != "7.1"
                n_chunks = (reader.frame_count + CHUNK - 1) // CHUNK if reader.frame_count > 0 else 1

                schedules = None
                if self.mode == "adaptive" and cfg.get("adaptive_beta"):
                    self.status_signal.emit("Computing adaptive schedules (streaming)...")
                    from ..adaptive import (AdaptiveBetaConfig, PolicyConfig,
                                           compute_adaptive_schedules_stream)
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

                    def sched_prog(*args):
                        sched_step = [0]
                        def _cb(*_a):
                            sched_step[0] += 1
                            if sched_step[0] % 5 == 0 or sched_step[0] == n_chunks:
                                self.progress_signal.emit(sched_step[0], n_chunks, "Schedules")
                            return _cb
                        return _cb

                    schedules = compute_adaptive_schedules_stream(
                        reader, reader.sample_rate, bcfg, pcfg, progress=sched_prog
                    )

                self.status_signal.emit("Rendering and streaming blocks to disk...")
                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
                with open_multichannel_stream(
                    self.output_path,
                    reader.sample_rate,
                    layout_name,
                    bit_depth=cfg.get("bit_depth", 24),
                    dither=dither,
                ) as writer:
                    reader.seek(0)
                    ci = 0
                    total_dur = reader.frame_count / reader.sample_rate if reader.sample_rate > 0 else 1.0

                    for l_chunk, r_chunk in reader.blocks():
                        if self._is_cancelled:
                            self.status_signal.emit("Render cancelled.")
                            return

                        b = schedules["beta"][ci] if schedules is not None and ci < len(schedules["beta"]) else None
                        g = schedules["gain"][ci] if schedules is not None and ci < len(schedules["gain"]) else None
                        d = schedules["decorr"][ci] if schedules is not None and ci < len(schedules["decorr"]) else None
                        blk = renderer.render_block(l_chunk, r_chunk, beta=b, gain=g, decorr=d)
                        if is_51:
                            from ..dsp.upmix import derive_5_1_from_7_1_block
                            blk = derive_5_1_from_7_1_block(blk)
                        writer.write_block(blk)
                        ci += 1

                        if ci % 5 == 0 or ci == n_chunks:
                            now = time.perf_counter()
                            elapsed = now - t_start
                            processed_dur = (ci * CHUNK) / reader.sample_rate
                            speed_rt = processed_dur / elapsed if elapsed > 0 else 0.0
                            eta = (total_dur - processed_dur) / speed_rt if speed_rt > 0 else 0.0
                            self.progress_signal.emit(ci, n_chunks, "Render (stream)")
                            self.metrics_signal.emit(speed_rt, elapsed, max(0.0, eta), writer.peak)

                    peak = writer.peak
                    frames = writer.frames_written

                reader.close()
                dur = frames / reader.sample_rate
                summary = {
                    "output": self.output_path,
                    "peak": float(peak),
                    "frames": frames,
                    "duration_s": dur,
                    "sample_rate": reader.sample_rate,
                    "layout": layout_name,
                }
                self.finished_signal.emit(summary)
                return

            # -------------------------------------------------------------
            # IN-MEMORY PATH
            # -------------------------------------------------------------
            self.status_signal.emit("Decoding input stereo file...")
            buf, meta = decode_stereo(self.input_path)
            renderer = make_renderer(buf.sample_rate, cfg)
            total_chunks = (buf.frame_count + CHUNK - 1) // CHUNK
            dur = buf.frame_count / buf.sample_rate

            stems = None
            if self.ai == "on":
                self.status_signal.emit("Separating stems with AI...")
                self.progress_signal.emit(0, 100, "AI Stem Separation")
                try:
                    from ..ai.separation import separate_stems
                    use_cache = not self.no_cache
                    stems = separate_stems(
                        buf.left,
                        buf.right,
                        buf.sample_rate,
                        progress=False,
                        use_cache=use_cache,
                        onnx_model_path=self.onnx_model,
                    )
                    self.progress_signal.emit(100, 100, "AI Stem Separation")
                except Exception as e:
                    self.status_signal.emit(f"AI separation warning: {e}")

                if stems is not None:
                    self.status_signal.emit("Evaluating AI stems with acceptance harness...")
                    try:
                        from ..ai.harness import evaluate_stems
                        rep = evaluate_stems(buf.left, buf.right, stems, buf.sample_rate)
                        if not rep.passed:
                            self.status_signal.emit("Warning: Stems did not meet strict harness thresholds (kept)")
                    except Exception as e:
                        self.status_signal.emit(f"AI harness evaluation note: {e}")

            schedules = None
            if self.mode == "adaptive" and cfg.get("adaptive_beta"):
                self.status_signal.emit("Computing adaptive schedules...")
                from ..adaptive import (AdaptiveBetaConfig, PolicyConfig,
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

                step_box = [0]
                def sched_bar(*args):
                    def _step(*_a):
                        step_box[0] += 1
                        if step_box[0] % 5 == 0 or step_box[0] == total_chunks:
                            self.progress_signal.emit(step_box[0], total_chunks, "Adaptive Schedules")
                        return _step
                    return _step

                schedules = compute_adaptive_schedules(
                    buf.left, buf.right, buf.sample_rate, bcfg, pcfg,
                    progress=sched_bar,
                )

            v_schedule = None
            if stems is not None and cfg.get("centre_mode") == "vocal_mask":
                self.status_signal.emit("Calculating vocal gate schedule...")
                from ..ai.harness import v_schedule_from_stems
                v_schedule = v_schedule_from_stems(
                    stems.vocals[0], stems.vocals[1], buf.left, buf.right,
                    buf.sample_rate, CHUNK
                )

            self.status_signal.emit("Rendering surround channels...")
            render_step = [0]
            def render_bar(*args):
                def _step(*_a):
                    render_step[0] += 1
                    if render_step[0] % 5 == 0 or render_step[0] == total_chunks:
                        now = time.perf_counter()
                        elapsed = now - t_start
                        processed_dur = (render_step[0] * CHUNK) / buf.sample_rate
                        speed_rt = processed_dur / elapsed if elapsed > 0 else 0.0
                        eta = (dur - processed_dur) / speed_rt if speed_rt > 0 else 0.0
                        self.progress_signal.emit(render_step[0], total_chunks, "Render DSP")
                        self.metrics_signal.emit(speed_rt, elapsed, max(0.0, eta), 0.0)
                    return _step
                return _step
            if cfg["layout"] == "7.1":
                out = renderer.render(
                    buf.left, buf.right,
                    beta_schedule=(schedules or {}).get("beta"),
                    gain_schedule=(schedules or {}).get("gain"),
                    decorr_schedule=(schedules or {}).get("decorr"),
                    progress=render_bar(total_chunks),
                )
                layout_name = INTERNAL_7_1.name
            else:
                primary = renderer.render(
                    buf.left, buf.right,
                    beta_schedule=(schedules or {}).get("beta"),
                    gain_schedule=(schedules or {}).get("gain"),
                    decorr_schedule=(schedules or {}).get("decorr"),
                    progress=render_bar(total_chunks),
                )
                out = derive_5_1_from_7_1(primary)
                layout_name = INTERNAL_5_1.name

            if v_schedule is not None:
                self.status_signal.emit("Injecting neural vocal centre...")
                from ..ai.centre import apply_vocal_centre
                vc_step = [0]
                def vc_bar(*args):
                    def _step(*_a):
                        vc_step[0] += 1
                        if vc_step[0] % 5 == 0 or vc_step[0] == total_chunks:
                            self.progress_signal.emit(vc_step[0], total_chunks, "Vocal Centre")
                        return _step
                    return _step
                out = apply_vocal_centre(
                    out,
                    stems.vocals[0],
                    stems.vocals[1],
                    v_schedule,
                    gain_db=float(cfg.get("vocal_gain_db", -6.0)),
                    progress=vc_bar(total_chunks),
                )

            self.status_signal.emit("Encoding output file...")
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            encode_output(
                out.channels,
                buf.sample_rate,
                layout_name,
                self.output_path,
                bit_depth=cfg.get("bit_depth", 24),
                dither=dither,
            )

            peak = float(out.peak())
            elapsed = time.perf_counter() - t_start
            speed_rt = dur / elapsed if elapsed > 0 else 0.0
            self.metrics_signal.emit(speed_rt, elapsed, 0.0, peak)

            summary = {
                "output": self.output_path,
                "peak": peak,
                "frames": out.frame_count,
                "duration_s": dur,
                "sample_rate": buf.sample_rate,
                "layout": layout_name,
            }
            self.status_signal.emit(f"Render complete: peak={peak:.4f}, {dur:.2f}s written")
            self.finished_signal.emit(summary)

        except Exception as e:
            self.error_signal.emit(str(e))


class AnalysisWorker(QThread):
    """Background analysis worker computing correlation, coherence, and band reports."""

    progress_signal = Signal(int, int, str)
    finished_signal = Signal(dict, str, str)  # summary_dict, csv_path, json_path
    error_signal = Signal(str)

    def __init__(
        self,
        input_path: str,
        out_dir: str,
        settings: AnalysisSettings | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.out_dir = out_dir
        self.settings = settings or AnalysisSettings()

    def run(self) -> None:
        try:
            self.progress_signal.emit(10, 100, "Decoding stereo audio")
            buf, meta = decode_stereo(self.input_path)

            self.progress_signal.emit(40, 100, "Calculating STFT and correlation metrics")
            rep = correlation_report(buf.left, buf.right, buf.sample_rate, self.settings)

            self.progress_signal.emit(70, 100, "Generating band summaries and reports")
            bands = band_average(rep)

            out_path = Path(self.out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            csv_path, json_path = write_reports(buf.left, buf.right, meta, str(out_path), self.settings)

            summary = {
                "sample_rate": buf.sample_rate,
                "frames": buf.frame_count,
                "duration_s": buf.frame_count / buf.sample_rate,
                "mid_rms": float(np.sqrt(np.mean(((buf.left + buf.right) * 0.5) ** 2))),
                "side_rms": float(np.sqrt(np.mean(((buf.left - buf.right) * 0.5) ** 2))),
                "bands": [{"band": b, "rho": float(r), "coherence": float(c)} for b, r, c in bands],
                "rho_freq": rep.rho.tolist(),
                "coherence_freq": rep.coherence.tolist(),
                "directness_freq": rep.directness.tolist(),
                "frequencies": rep.frequencies.tolist(),
            }

            self.progress_signal.emit(100, 100, "Analysis complete")
            self.finished_signal.emit(summary, csv_path, json_path)
        except Exception as e:
            self.error_signal.emit(str(e))


class ValidationWorker(QThread):
    """Background worker validating multichannel rendered audio."""

    finished_signal = Signal(dict)  # ValidationResult dict payload
    error_signal = Signal(str)

    def __init__(
        self,
        input_path: str,
        reference_path: str | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.reference_path = reference_path

    def run(self) -> None:
        try:
            result = validate_file(self.input_path, reference=self.reference_path)
            res_dict = {
                "passed": result.passed,
                "checks": result.checks,
                "json": result.to_json(),
            }
            self.finished_signal.emit(res_dict)
        except Exception as e:
            self.error_signal.emit(str(e))


class BenchmarkWorker(QThread):
    """Worker running comparative profile benchmarks."""

    progress_signal = Signal(int, int, str)
    finished_signal = Signal(dict, str)  # results_dict, summary_json_path
    error_signal = Signal(str)

    def __init__(
        self,
        input_path: str,
        profiles: list[str],
        layout: str = "7.1",
        out_dir: str = "reports",
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.profiles = profiles
        self.layout = layout
        self.out_dir = out_dir

    def run(self) -> None:
        try:
            buf, meta = decode_stereo(self.input_path)
            results = {}
            total_prof = len(self.profiles)
            default_cfg = Path(__file__).resolve().parent.parent.parent / "configs" / "default.toml"

            for idx, profile in enumerate(self.profiles):
                self.progress_signal.emit(idx, total_prof, f"Benchmarking profile: {profile}")
                cfg_path = Path("configs") / f"{profile}.toml"
                if not cfg_path.exists():
                    cfg_path = default_cfg
                cfg = load_config(cfg_path)
                cfg["layout"] = self.layout
                cfg["adaptive"] = profile == "adaptive"

                renderer = make_renderer(buf.sample_rate, cfg)
                primary = renderer.render(buf.left, buf.right)
                out = primary if self.layout == "7.1" else derive_5_1_from_7_1(primary)

                out_path = Path(self.out_dir) / f"{Path(self.input_path).stem}_{profile}.flac"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                encode_flac(out.channels, buf.sample_rate, out.layout_name, str(out_path),
                            bit_depth=cfg.get("bit_depth", 24))

                rep = correlation_report(buf.left, buf.right, buf.sample_rate, AnalysisSettings())
                bands = band_average(rep)

                channel_order = list(INTERNAL_7_1.order) if out.n_channels == 8 else list(INTERNAL_5_1.order)
                lbl_idx = {lbl: i for i, lbl in enumerate(channel_order)}
                _, sr_rms = measure(out.channels[lbl_idx["SR"]])
                _, l_rms = measure(out.channels[lbl_idx["FL"]])

                results[profile] = {
                    "output": str(out_path),
                    "peak": round(float(out.peak()), 6),
                    "surround_rms": round(float(sr_rms), 6),
                    "front_rms": round(float(l_rms), 6),
                    "stereo_bands": [{"band": b, "rho": round(float(r), 4), "coherence": round(float(c), 4)}
                                     for b, r, c in bands],
                }

            summary_path = Path(self.out_dir) / f"{Path(self.input_path).stem}_benchmark.json"
            summary_path.write_text(json.dumps(results, indent=2))

            self.progress_signal.emit(total_prof, total_prof, "Benchmark complete")
            self.finished_signal.emit(results, str(summary_path))
        except Exception as e:
            self.error_signal.emit(str(e))


class AiEvalWorker(QThread):
    """Worker running AI stem separation and acceptance harness independent of rendering."""

    progress_signal = Signal(int, int, str)
    finished_signal = Signal(dict, object)  # harness_report_dict, stems_obj
    error_signal = Signal(str)

    def __init__(
        self,
        input_path: str,
        onnx_model: str | None = None,
        no_cache: bool = False,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.onnx_model = onnx_model
        self.no_cache = no_cache

    def run(self) -> None:
        try:
            self.progress_signal.emit(10, 100, "Decoding audio...")
            buf, meta = decode_stereo(self.input_path)

            self.progress_signal.emit(25, 100, "Separating stems...")
            from ..ai.harness import evaluate_stems
            from ..ai.separation import separate_stems

            use_cache = not self.no_cache
            stems = separate_stems(
                buf.left,
                buf.right,
                buf.sample_rate,
                progress=False,
                use_cache=use_cache,
                onnx_model_path=self.onnx_model,
            )

            self.progress_signal.emit(75, 100, "Running 4-point acceptance harness...")
            rep = evaluate_stems(buf.left, buf.right, stems, buf.sample_rate)

            report_dict = {
                "passed": rep.passed,
                "checks": rep.checks,
                "metrics": rep.metrics,
                "json": rep.to_json(),
            }

            self.progress_signal.emit(100, 100, "AI evaluation complete")
            self.finished_signal.emit(report_dict, stems)
        except Exception as e:
            self.error_signal.emit(str(e))


class RoutingTestWorker(QThread):
    """Worker generating 8-channel routing calibration tone and verifying channel positions."""

    finished_signal = Signal(str, bool, list)  # output_path, passed, checks
    error_signal = Signal(str)

    def __init__(
        self,
        output_path: str,
        seconds: float = 1.0,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.output_path = output_path
        self.seconds = seconds

    def run(self) -> None:
        try:
            sr = 48000
            chans, labels = routing_test_signal(sr, seconds=self.seconds)
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            encode_flac(chans, sr, INTERNAL_7_1.name, self.output_path, bit_depth=24)

            detected = detect_channel_frequencies(self.output_path)
            expected = {
                "FL": 1000.0, "FR": 2000.0, "FC": 3000.0, "LFE": 60.0,
                "SL": 4000.0, "SR": 5000.0, "BL": 6000.0, "BR": 7000.0,
            }

            checks = []
            all_ok = True
            for label in labels:
                exp = expected[label]
                got = detected.get(label, 0.0)
                match = abs(got - exp) < exp * 0.05
                all_ok &= match
                checks.append({
                    "channel": label,
                    "expected_hz": exp,
                    "detected_hz": got,
                    "passed": match,
                })

            self.finished_signal.emit(self.output_path, all_ok, checks)
        except Exception as e:
            self.error_signal.emit(str(e))


class NativeBuildWorker(QThread):
    """Worker compiling C++20 DSP native core via CMake and streaming output logs."""

    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)

    def run(self) -> None:
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            native_dir = repo_root / "native"
            build_dir = native_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)

            self.log_signal.emit(f"Working directory: {build_dir}")
            self.log_signal.emit("Configuring native core with CMake...")

            # CMake configure
            cmake_cfg = ["cmake", "-S", str(native_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"]
            proc = subprocess.Popen(
                cmake_cfg,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(repo_root),
            )
            if proc.stdout:
                for line in proc.stdout:
                    self.log_signal.emit(line.rstrip())
            proc.wait()

            if proc.returncode != 0:
                self.finished_signal.emit(False, f"CMake configure failed with return code {proc.returncode}")
                return

            self.log_signal.emit("Building native core target (Release)...")
            cmake_build = ["cmake", "--build", str(build_dir), "--config", "Release"]
            proc = subprocess.Popen(
                cmake_build,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(repo_root),
            )
            if proc.stdout:
                for line in proc.stdout:
                    self.log_signal.emit(line.rstrip())
            proc.wait()

            if proc.returncode != 0:
                self.finished_signal.emit(False, f"CMake build failed with return code {proc.returncode}")
                return

            self.log_signal.emit("Native core built successfully.")
            self.finished_signal.emit(True, "Native build completed successfully.")
        except Exception as e:
            self.log_signal.emit(f"Build error: {e}")
            self.finished_signal.emit(False, str(e))
