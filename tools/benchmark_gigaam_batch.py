"""Diagnose GigaAM native batching from encoded media through publication.

Usage: uv run python -m tools.benchmark_gigaam_batch MEDIA.mp4 --device cuda
This bypasses HTTP upload. No media or transcript files are written.
Results are printed as JSON.
GPU utilization is device-wide and may include other processes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stt.media import EncodedMedia
from stt.models.base import STTModelAdapter
from stt.models.gigaam import GigaAMAdapter, load_gigaam
from stt.pipeline import run_stt_job


class BenchmarkJobs:
    def __init__(self) -> None:
        self.segments: list[dict[str, Any]] = []
        self.error: str | None = None
        self.completed = False

    def is_cancelled(self, job_id: str) -> bool:
        return False

    def update_event(self, job_id: str, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "progress":
            self.segments.append(data["segment"])
        elif event_type == "error":
            self.error = data["message"]
        elif event_type == "complete":
            self.completed = True


class TimedAdapter(STTModelAdapter):
    def __init__(self, inner: GigaAMAdapter, *, cuda: bool) -> None:
        self.inner = inner
        self.supports_native_batching = inner.supports_native_batching
        self.cuda = cuda
        self.calls: list[float] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> str:
        return self.inner.transcribe(audio, **kwargs)

    def transcribe_batch(self, chunks: tuple[np.ndarray, ...], **kwargs: Any) -> list[str]:
        if self.cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.inner.transcribe_batch(chunks, **kwargs)
        if self.cuda:
            torch.cuda.synchronize()
        self.calls.append(time.perf_counter() - start)
        return result


class ResourceSampler:
    def __init__(self, *, cuda: bool) -> None:
        self.cuda = cuda
        self.rss_peak_bytes = 0
        self.gpu_utilization: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _sample(self) -> None:
        while not self._stop.is_set():
            if os.path.exists("/proc/self/statm"):
                with open("/proc/self/statm") as statm:
                    resident_pages = int(statm.read().split()[1])
                self.rss_peak_bytes = max(self.rss_peak_bytes, resident_pages * os.sysconf("SC_PAGE_SIZE"))
            if self.cuda:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2, check=True,
                    )
                    self.gpu_utilization.append(int(result.stdout.splitlines()[0]))
                except (OSError, ValueError, subprocess.SubprocessError, IndexError):
                    pass
            self._stop.wait(1)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--chunk-sec", type=int, default=20)
    args = parser.parse_args()
    if 1 not in args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        parser.error("batch sizes must include 1 and be positive")

    media = EncodedMedia(args.media.read_bytes(), args.media.name)
    cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    model = load_gigaam(args.device)
    baseline: list[dict[str, Any]] | None = None
    for size in [1, *(candidate for candidate in args.batch_sizes if candidate != 1)]:
        if cuda:
            torch.cuda.synchronize()
        if cuda:
            torch.cuda.empty_cache()
            baseline_reserved = torch.cuda.memory_reserved()
            free_before, _ = torch.accelerator.memory.get_memory_info()
            torch.cuda.reset_peak_memory_stats()
        timed = TimedAdapter(model, cuda=cuda)
        jobs = BenchmarkJobs()
        with ResourceSampler(cuda=cuda) as resources:
            cpu_before = time.process_time()
            start = time.perf_counter()
            run_stt_job(
                job_id="benchmark", input_paths=media, jobs=jobs, model=timed,
                log=logging.getLogger("benchmark"), sample_rate=16000,
                chunk_sec=args.chunk_sec, batch_size=size,
            )
            elapsed = time.perf_counter() - start
            cpu_elapsed = time.process_time() - cpu_before
        if jobs.error or not jobs.completed:
            raise RuntimeError(jobs.error or "batch job did not complete")
        if baseline is None:
            baseline = jobs.segments
        differences = [
            index for index, (first, current) in enumerate(zip(baseline, jobs.segments, strict=True))
            if first["text"] != current["text"]
        ]
        boundaries_equal = [
            (item["start"], item["end"], item["source"]) for item in baseline
        ] == [(item["start"], item["end"], item["source"]) for item in jobs.segments]
        result = {
            "torch_version": torch.__version__,
            "batch_size": size,
            "segments": len(jobs.segments),
            "wall_sec": round(elapsed, 3),
            "cpu_cores_average": round(cpu_elapsed / elapsed, 2),
            "rss_peak_mib": round(resources.rss_peak_bytes / 2**20, 1) if resources.rss_peak_bytes else None,
            "device_gpu_utilization_percent_mean": round(statistics.mean(resources.gpu_utilization), 1)
            if resources.gpu_utilization else None,
            "inference_calls": len(timed.calls),
            "inference_total_sec": round(sum(timed.calls), 3),
            "inference_p95_sec": round(percentile(timed.calls, 0.95), 3),
            "inference_max_sec": round(max(timed.calls), 3),
            "boundaries_equal_to_b1": boundaries_equal,
            "text_difference_count": len(differences),
            "text_difference_indices": differences,
        }
        if cuda:
            result.update({
                "cuda_baseline_reserved_mib": round(baseline_reserved / 2**20, 1),
                "cuda_additional_peak_reserved_mib": round(
                    (torch.cuda.max_memory_reserved() - baseline_reserved) / 2**20, 1
                ),
                "cuda_free_before_mib": round(free_before / 2**20, 1),
            })
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
