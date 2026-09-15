"""Compare batch STT with LiveSTTSession on identical decoded PCM."""

import argparse
import hashlib
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.jobs import JobRegistry
from server import Config, models
from stt.buffer import decode_media_bytes
from stt.live import LiveSTTSession
from stt.pipeline import run_stt_job


class TracedModel:
    def __init__(self, model, mode: str):
        self.model = model
        self.mode = mode
        self.calls = []

    def transcribe(self, audio: np.ndarray, **kwargs):
        started = time.perf_counter()
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        result = self.model.transcribe(pcm, **kwargs)
        self.calls.append({
            "mode": self.mode,
            "call_index": len(self.calls),
            "samples": len(pcm),
            "duration_sec": len(pcm) / Config.SR,
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "inference_ms": (time.perf_counter() - started) * 1000,
            "text": str(result),
        })
        return result


def run_live(pcm: np.ndarray, model, jobs: JobRegistry, min_silence_sec: float):
    traced = TracedModel(model, "live")
    job = jobs.create("stt", "diagnostic", {"filename": "live"})
    live = LiveSTTSession(job.job_id, "diagnostic", traced, jobs, Config.SR)
    live._min_silence_windows = max(1, int(min_silence_sec * Config.SR / 512))
    spans = []
    engine = live._vad_engine

    class VADTrace:
        """Observe local VAD boundaries without changing production aggregation."""

        def __init__(self):
            self.windows = 0
            self.start = None
            self.silence = 0

        def process_frame(self, state, window):
            probability = engine.process_frame(state, window)
            self.windows += 1
            if probability >= 0.5:
                if self.start is None:
                    self.start = self.windows - 1
                self.silence = 0
            elif self.start is not None:
                self.silence += 1
                if self.silence >= live._min_silence_windows:
                    spans.append({"start": self.start * 512, "end": self.windows * 512, "reason": "VAD_ENDPOINT"})
                    self.start, self.silence = None, 0
            return probability

        def finish(self):
            if self.start is not None:
                spans.append({"start": self.start * 512, "end": self.windows * 512, "reason": "FLUSH"})

    traced_vad = VADTrace()
    live._vad_engine = traced_vad
    chunk_size = Config.SR
    for start in range(0, len(pcm), chunk_size):
        live.process_pcm_chunk(pcm[start : start + chunk_size])
    live.flush()
    traced_vad.finish()
    return job, traced, spans


def transcribe_raw_intervals(model, pcm: np.ndarray, intervals, mode: str):
    traced = TracedModel(model, mode)
    for start, end in intervals:
        if end > start:
            traced.transcribe(pcm[start:end])
    return traced


def split_intervals_at_boundaries(length: int, boundaries, window_samples: int):
    intervals = []
    for window_start in range(0, length, window_samples):
        window_end = min(length, window_start + window_samples)
        cuts = sorted({end for end in boundaries if window_start < end < window_end})
        points = [window_start, *cuts, window_end]
        intervals.extend(pairwise(points))
    return intervals


def aggregate_spans(spans, commit_pause_samples: int, max_context_samples: int):
    groups = []
    if not spans:
        return groups
    start, end = spans[0]["start"], spans[0]["end"]
    for span in spans[1:]:
        gap = span["start"] - end
        if gap >= commit_pause_samples or span["end"] - start > max_context_samples:
            groups.append((start, end))
            start = span["start"]
        end = span["end"]
    groups.append((start, end))
    return groups


def word_edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right.split()) + 1))
    for index, word in enumerate(left.split(), start=1):
        current = [index]
        for other_index, other_word in enumerate(right.split(), start=1):
            current.append(min(
                previous[other_index] + 1,
                current[other_index - 1] + 1,
                previous[other_index - 1] + (word != other_word),
            ))
        previous = current
    return previous[-1]


def preview_intervals(groups, spans, cadence: str):
    events = []
    for group_start, group_end in groups:
        if cadence.startswith("every-"):
            step = int(float(cadence.removeprefix("every-")) * Config.SR)
            times = range(group_start + step, group_end, step)
        else:
            throttle = float(cadence.removeprefix("endpoint-")) * Config.SR
            last = group_start - throttle
            times = []
            for span in spans:
                if group_start < span["end"] < group_end and span["end"] - last >= throttle:
                    times.append(span["end"])
                    last = span["end"]
        events.extend((group_start, event) for event in times)
    return events


def report_preview_policy(model, pcm, groups, spans, cadence: str):
    intervals = preview_intervals(groups, spans, cadence)
    if not intervals:
        print(f"PREVIEW {cadence}: calls=0 (no eligible snapshots)")
        return
    preview = transcribe_raw_intervals(model, pcm, intervals, f"preview-{cadence}")
    times = [end / Config.SR for _, end in intervals]
    updates = np.diff(times) if len(times) > 1 else np.array([])
    churn = [word_edit_distance(a["text"], b["text"]) for a, b in pairwise(preview.calls)]
    total_ms = sum(call["inference_ms"] for call in preview.calls)
    update_median = np.median(updates) if len(updates) else 0.0
    update_p95 = np.percentile(updates, 95) if len(updates) else 0.0
    print(
        f"PREVIEW {cadence}: calls={len(preview.calls)} "
        f"ttft={times[0]:.2f}s "
        f"updates_median={update_median:.2f}s p95={update_p95:.2f}s "
        f"inference={total_ms:.1f}ms ({total_ms / (len(pcm) / Config.SR):.1f}ms/audio-sec) "
        f"churn_median={np.median(churn) if churn else 0:.1f} words"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--chunk-sec", type=float, default=1.0)
    args = parser.parse_args()

    audio_buffer = decode_media_bytes(args.input.read_bytes(), target_sample_rate=Config.SR)
    pcm = audio_buffer.as_ndarray()
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    print(f"input samples={len(pcm)} duration_sec={len(pcm) / Config.SR:.3f}")

    base_model = models.get_stt_model("gigaam")

    batch_jobs = JobRegistry()
    batch_job = batch_jobs.create("stt", "diagnostic", {"filename": args.input.name})
    batch_model = TracedModel(base_model, "batch")
    run_stt_job(
        job_id=batch_job.job_id,
        input_paths=audio_buffer,
        jobs=batch_jobs,
        model=batch_model,
        log=__import__("logging").getLogger("compare_live_batch"),
        sample_rate=Config.SR,
        chunk_sec=20,
    )

    for label, registry, job, traced, reasons in [("BATCH", batch_jobs, batch_job, batch_model, [])]:
        status = registry.get_status(job.job_id)
        print(f"\n{label}: state={status['state'] if status else 'unknown'}")
        for call in traced.calls:
            print(
                f"  call={call['call_index']} samples={call['samples']} "
                f"duration={call['duration_sec']:.3f}s "
                f"sha={call['pcm_sha256'][:12]} inference={call['inference_ms']:.1f}ms "
                f"text={call['text']!r}"
            )

    live_runs = {}
    for silence in (0.35, 0.50, 0.75, 1.00, 1.25, 1.50):
        live_jobs = JobRegistry()
        _live_job, live_model, spans = run_live(pcm, base_model, live_jobs, silence)
        live_runs[silence] = (live_model, spans)
        calls = live_model.calls
        short = sum(call["duration_sec"] < 2 for call in calls)
        print(
            f"\nLIVE silence={silence:.2f}s calls={len(calls)} "
            f"short(<2s)={short} reasons={','.join(span['reason'] for span in spans)}"
        )
        print("  " + " ".join(call["text"] for call in calls))

    _, live_spans = live_runs[0.35]
    boundaries = [span["end"] for span in live_spans]
    raw_windows = split_intervals_at_boundaries(len(pcm), boundaries, 20 * Config.SR)
    raw_batch_windows = [(start, end) for start in range(0, len(pcm), 20 * Config.SR) for end in [min(len(pcm), start + 20 * Config.SR)]]
    direct = transcribe_raw_intervals(base_model, pcm, raw_batch_windows, "D-direct")
    split = transcribe_raw_intervals(base_model, pcm, raw_windows, "D-split")
    assert sum(end - start for start, end in raw_windows) == len(pcm)
    print(f"\nD canonical source windows: direct_calls={len(direct.calls)} split_calls={len(split.calls)}")
    print("  DIRECT " + " ".join(call["text"] for call in direct.calls))
    print("  SPLIT  " + " ".join(call["text"] for call in split.calls))

    for pause_sec, context_sec in ((1.0, 12.0), (1.25, 16.0)):
        groups = aggregate_spans(
            live_spans,
            int(pause_sec * Config.SR),
            int(context_sec * Config.SR),
        )
        aggregated = transcribe_raw_intervals(base_model, pcm, groups, "C-aggregate")
        print(
            f"\nC pause={pause_sec:.2f}s context={context_sec:.0f}s calls={len(aggregated.calls)} "
            f"median_input={np.median([call['duration_sec'] for call in aggregated.calls]):.2f}s"
        )
        print("  " + " ".join(call["text"] for call in aggregated.calls))

    final_groups = aggregate_spans(live_spans, int(1.0 * Config.SR), int(12.0 * Config.SR))
    final = transcribe_raw_intervals(base_model, pcm, final_groups, "final")
    print(
        f"\nPREVIEW CHECKPOINT final_calls={len(final.calls)} "
        f"final_pcm={[hashlib.sha256(pcm[start:end].tobytes()).hexdigest()[:12] for start, end in final_groups]}"
    )
    for cadence in ("every-1", "every-2", "endpoint-0", "endpoint-1", "endpoint-2"):
        report_preview_policy(base_model, pcm, final_groups, live_spans, cadence)


if __name__ == "__main__":
    main()
