#!/usr/bin/env python3
"""Record a short pitch clip from a capture device.

This is a convenience tool. OBS/capture-card software is also fine as long as
clips are saved into data/raw/ and can be opened by OpenCV.

Each recording also writes timing sidecars next to the video:
- ``pitch_001.frames.jsonl`` per-frame timestamps (seconds from recording start)
- ``pitch_001.meta.json`` capture metadata (requested/actual FPS, frame count, size)

The writer FPS is set from a short warmup measurement so encoded MP4 frame counts
match the sidecar entries. Use frame numbers from the labeled video for now;
treat sidecar timestamps as diagnostic only until capture_fps is stable.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2


def open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int) and sys.platform == "win32":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, requested_fps: float) -> None:
    """Apply capture settings that improve live FPS on Windows USB devices."""
    if sys.platform == "win32":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, requested_fps)


def draw_text(img, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def sidecar_paths(video_path: Path) -> tuple[Path, Path]:
    stem = video_path.with_suffix("")
    return stem.with_suffix(".frames.jsonl"), stem.with_suffix(".meta.json")


def count_encoded_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def measure_capture_fps(cap: cv2.VideoCapture, warmup_frames: int = 45) -> float:
    """Read frames without writing to estimate the device's sustained capture rate."""
    start = time.perf_counter()
    count = 0
    for _ in range(warmup_frames):
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    elapsed = time.perf_counter() - start
    if count == 0 or elapsed <= 0:
        return 30.0
    return count / elapsed


def drain_capture_buffer(cap: cv2.VideoCapture, max_frames: int = 45) -> int:
    """Discard buffered frames so the next reads reflect live capture rate."""
    drained = 0
    for _ in range(max_frames):
        ok, _ = cap.read()
        if not ok:
            break
        drained += 1
    return drained


def rolling_fps(frame_times: deque[float]) -> float:
    if len(frame_times) < 2:
        return 0.0
    span = frame_times[-1] - frame_times[0]
    if span <= 0:
        return 0.0
    return (len(frame_times) - 1) / span


def preview_frame(frame, width: int, height: int, preview_scale: float):
    """Downscale for display so imshow does less work than a full-frame copy."""
    if preview_scale >= 0.999:
        return frame.copy()
    preview_w = max(1, int(round(width * preview_scale)))
    preview_h = max(1, int(round(height * preview_scale)))
    return cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)


class AsyncClipWriter:
    """Encode frames on a worker thread so capture reads are not blocked by disk IO."""

    def __init__(self, writer: cv2.VideoWriter, rec_start: float) -> None:
        self._writer = writer
        self._rec_start = rec_start
        self._queue: queue.Queue = queue.Queue(maxsize=180)
        self._timestamps: list[dict[str, float | int]] = []
        self._submitted = 0
        self._lock = threading.Lock()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="clip-writer", daemon=True)
        self._thread.start()

    def submit(self, frame) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put(frame.copy())
        self._submitted += 1

    def submitted_count(self) -> int:
        return self._submitted

    def finish(self, timeout_s: float = 30.0) -> list[dict[str, float | int]]:
        self._queue.put(None)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("Timed out waiting for clip encoder thread")
        if self._error is not None:
            raise self._error
        with self._lock:
            return list(self._timestamps)

    def _run(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                self._writer.write(frame)
                elapsed = time.perf_counter() - self._rec_start
                with self._lock:
                    self._timestamps.append(
                        {"frame": len(self._timestamps), "timestamp": round(elapsed, 6)}
                    )
        except Exception as exc:  # noqa: BLE001 - propagate to main thread on finish()
            self._error = exc
        finally:
            self._writer.release()


class ClipEncodeJob:
    """Finish MP4 encoding off the capture loop so USB reads never stall."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(
        self,
        async_writer: AsyncClipWriter,
        out_path: Path,
        finalizer: AsyncFinalizer,
        *,
        requested_fps: float,
        writer_fps: float,
        capture_fps_measured: float,
        width: int,
        height: int,
        rec_start: float,
        verify_encoded_count: bool,
    ) -> None:
        if self.busy():
            self._thread.join()  # type: ignore[union-attr]

        def _run() -> None:
            frame_timestamps = async_writer.finish()
            finalizer.submit(
                out_path,
                frame_timestamps,
                requested_fps=requested_fps,
                writer_fps=writer_fps,
                capture_fps_measured=capture_fps_measured,
                width=width,
                height=height,
                rec_start=rec_start,
                verify_encoded_count=verify_encoded_count,
            )

        self._thread = threading.Thread(target=_run, name="clip-encode-job", daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()


class AsyncFinalizer:
    """Write sidecars off the preview loop so capture can keep draining."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(
        self,
        out_path: Path,
        frame_timestamps: list[dict[str, float | int]],
        **kwargs,
    ) -> None:
        if self.busy():
            self._thread.join()  # type: ignore[union-attr]
        self._thread = threading.Thread(
            target=finalize_recording,
            args=(out_path, frame_timestamps),
            kwargs=kwargs,
            name="clip-finalizer",
            daemon=True,
        )
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()


def sync_sidecars_to_encoded(
    frame_timestamps: list[dict[str, float | int]],
    encoded_frame_count: int,
) -> tuple[list[dict[str, float | int]], bool]:
    sidecar_count = len(frame_timestamps)
    if encoded_frame_count == sidecar_count:
        return frame_timestamps, True
    if encoded_frame_count < sidecar_count:
        print(
            f"WARNING: encoded MP4 has {encoded_frame_count} frames but sidecar has {sidecar_count}; "
            "trimming sidecar to match MP4."
        )
        return frame_timestamps[:encoded_frame_count], False
    print(
        f"WARNING: encoded MP4 has {encoded_frame_count} frames but sidecar has {sidecar_count}; "
        "keeping sidecar length (MP4 read may be inconsistent)."
    )
    return frame_timestamps, False


def write_sidecars(
    out_path: Path,
    *,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    frame_count: int,
    encoded_frame_count: int,
    width: int,
    height: int,
    actual_duration: float,
    frame_timestamps: list[dict[str, float | int]],
    frames_in_sync: bool,
) -> None:
    frames_path, meta_path = sidecar_paths(out_path)
    with frames_path.open("w", encoding="utf-8") as frames_file:
        for record in frame_timestamps:
            frames_file.write(json.dumps(record) + "\n")

    actual_fps = frame_count / actual_duration if actual_duration > 0 else 0.0
    meta = {
        "video": str(out_path),
        "requested_fps": float(requested_fps),
        "writer_fps": round(writer_fps, 2),
        "capture_fps_measured": round(capture_fps_measured, 2),
        "actual_fps": round(actual_fps, 2),
        "frame_count": frame_count,
        "encoded_frame_count": encoded_frame_count,
        "sidecar_frame_count": len(frame_timestamps),
        "frames_in_sync": frames_in_sync,
        "width": width,
        "height": height,
        "duration_s": round(actual_duration, 3),
        "timing_source": "capture_sidecar",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Saved {out_path}")
    print(f"Saved {frames_path}")
    print(f"Saved {meta_path}")
    print(
        f"frames={frame_count} encoded={encoded_frame_count} "
        f"duration={actual_duration:.3f}s capture_fps={capture_fps_measured:.2f} "
        f"writer_fps={writer_fps:.2f} in_sync={frames_in_sync}"
    )
    if capture_fps_measured < requested_fps * 0.75:
        print(
            f"WARNING: capture measured {capture_fps_measured:.1f} fps vs requested {requested_fps:.1f}. "
            "Expect fewer ball positions per pitch; labels still use video frame numbers."
        )


def finalize_recording(
    out_path: Path,
    frame_timestamps: list[dict[str, float | int]],
    *,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    width: int,
    height: int,
    rec_start: float,
    verify_encoded_count: bool = False,
) -> None:
    if verify_encoded_count:
        encoded_frame_count = count_encoded_frames(out_path)
    else:
        encoded_frame_count = len(frame_timestamps)
    frame_timestamps, frames_in_sync = sync_sidecars_to_encoded(frame_timestamps, encoded_frame_count)
    frame_count = len(frame_timestamps)
    actual_duration = time.perf_counter() - rec_start
    write_sidecars(
        out_path,
        requested_fps=requested_fps,
        writer_fps=writer_fps,
        capture_fps_measured=capture_fps_measured,
        frame_count=frame_count,
        encoded_frame_count=encoded_frame_count,
        width=width,
        height=height,
        actual_duration=actual_duration,
        frame_timestamps=frame_timestamps,
        frames_in_sync=frames_in_sync,
    )


def create_writer(out_path: Path, writer_fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {out_path}")
    return writer


def append_recorded_frame(
    writer: cv2.VideoWriter,
    frame,
    frame_timestamps: list[dict[str, float | int]],
    rec_start: float,
) -> None:
    writer.write(frame)
    elapsed = time.perf_counter() - rec_start
    frame_timestamps.append({"frame": len(frame_timestamps), "timestamp": round(elapsed, 6)})


def next_clip_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_num = 0
    for path in out_dir.glob("pitch_*.mp4"):
        suffix = path.stem.removeprefix("pitch_")
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    return out_dir / f"pitch_{max_num + 1:03d}.mp4"


def run_interactive_preview(
    cap: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    clip_seconds: float,
    clip_dir: Path,
    flush_frames: int,
    min_recover_fps: float,
    recover_seconds: float,
    recover_gate: bool,
    verify_encoded_count: bool,
    async_encode: bool,
    preview_scale: float,
) -> None:
    frame_count = 0
    preview_start = time.perf_counter()
    frame_times: deque[float] = deque(maxlen=60)
    finalizer = AsyncFinalizer()
    encode_job = ClipEncodeJob()

    recording = False
    async_writer: AsyncClipWriter | None = None
    writer: cv2.VideoWriter | None = None
    out_path: Path | None = None
    rec_start = 0.0
    frame_timestamps: list[dict[str, float | int]] = []
    rec_frames_submitted = 0

    recovering = False
    recover_start: float | None = None
    ready_since: float | None = None

    print(f"Interactive preview ready. Press r to record a {clip_seconds:.1f}s clip, q to quit.")
    if recover_gate:
        print(
            f"Recovery gate: wait for preview fps >= {min_recover_fps:.0f} "
            f"for {recover_seconds:.1f}s before the next clip."
        )

    def begin_recovery() -> None:
        nonlocal recovering, recover_start, ready_since
        drained = drain_capture_buffer(cap, flush_frames)
        recovering = True
        recover_start = time.perf_counter()
        ready_since = None
        frame_times.clear()
        if drained:
            print(f"Flushed {drained} buffered frame(s) after recording.")

    def recovery_ready(now_fps: float) -> bool:
        nonlocal ready_since
        if not recover_gate:
            return True
        if now_fps < min_recover_fps:
            ready_since = None
            return False
        if ready_since is None:
            ready_since = time.perf_counter()
        return (time.perf_counter() - ready_since) >= recover_seconds

    def finish_clip() -> None:
        nonlocal recording, async_writer, writer, out_path, frame_timestamps, rec_frames_submitted
        if out_path is None:
            return
        finished_path = out_path
        finished_rec_start = rec_start
        if async_writer is not None:
            encode_job.submit(
                async_writer,
                finished_path,
                finalizer,
                requested_fps=requested_fps,
                writer_fps=writer_fps,
                capture_fps_measured=capture_fps_measured,
                width=width,
                height=height,
                rec_start=finished_rec_start,
                verify_encoded_count=verify_encoded_count,
            )
            async_writer = None
            writer = None
        elif writer is not None:
            writer.release()
            writer = None
            finalizer.submit(
                finished_path,
                list(frame_timestamps),
                requested_fps=requested_fps,
                writer_fps=writer_fps,
                capture_fps_measured=capture_fps_measured,
                width=width,
                height=height,
                rec_start=finished_rec_start,
                verify_encoded_count=verify_encoded_count,
            )
        recording = False
        out_path = None
        frame_timestamps = []
        rec_frames_submitted = 0
        begin_recovery()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        frame_count += 1
        frame_times.append(now)
        live_fps = rolling_fps(frame_times)
        elapsed = now - preview_start
        fps_estimate = frame_count / elapsed if elapsed > 0 else 0.0

        if recovering and recovery_ready(live_fps) and not encode_job.busy():
            recovering = False
            recover_start = None
            ready_since = None
            print(f"Capture ready again (live fps={live_fps:.1f}).")

        preview = preview_frame(frame, width, height, preview_scale)

        if recording and out_path is not None:
            rec_elapsed = now - rec_start
            if async_writer is not None:
                async_writer.submit(frame)
                rec_frames_submitted = async_writer.submitted_count()
            else:
                append_recorded_frame(writer, frame, frame_timestamps, rec_start)  # type: ignore[arg-type]
                rec_frames_submitted = len(frame_timestamps)
            rec_fps = rec_frames_submitted / rec_elapsed if rec_elapsed > 0 else 0.0
            draw_text(
                preview,
                f"REC {rec_elapsed:.1f}s/{clip_seconds:.1f}s {out_path.name} "
                f"rec_fps={rec_fps:.1f} ({rec_frames_submitted}f)",
                20,
                40,
            )
            if rec_elapsed >= clip_seconds:
                finish_clip()
        else:
            if encode_job.busy():
                status = f"encoding previous clip  |  live_fps={live_fps:.1f}"
            elif recovering:
                need = max(0.0, recover_seconds - ((now - ready_since) if ready_since else 0.0))
                status = (
                    f"recovering live_fps={live_fps:.1f} need>={min_recover_fps:.0f} "
                    f"hold={need:.1f}s"
                    if recover_gate
                    else f"recovering live_fps={live_fps:.1f}"
                )
            elif finalizer.busy():
                status = f"finalizing previous clip  |  live_fps={live_fps:.1f}"
            else:
                status = f"READY live_fps={live_fps:.1f}  |  r=record  |  q=quit"
            draw_text(preview, status, 20, 40)

        cv2.imshow("record_clip", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if recording:
                finish_clip()
            break
        if key == ord("r") and not recording:
            if encode_job.busy():
                print("Still encoding previous clip; wait for live preview to recover.")
                continue
            if finalizer.busy():
                print("Still finalizing previous clip; wait a moment.")
                continue
            if recovering and recover_gate and not recovery_ready(live_fps):
                print(
                    f"Capture not ready yet (live_fps={live_fps:.1f}, need >={min_recover_fps:.0f} "
                    f"for {recover_seconds:.1f}s). Wait for READY."
                )
                continue
            out_path = next_clip_path(clip_dir)
            writer = create_writer(out_path, writer_fps, width, height)
            recording = True
            rec_start = time.perf_counter()
            frame_timestamps = []
            if async_encode:
                async_writer = AsyncClipWriter(writer, rec_start)
                writer = None
                async_writer.submit(frame)
            else:
                append_recorded_frame(writer, frame, frame_timestamps, rec_start)
            recovering = False
            print(f"Recording {clip_seconds:.1f}s to {out_path}")

    if encode_job.busy():
        encode_job.join()
    if finalizer.busy():
        finalizer.join()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Preview ended. frames={frame_count} session_avg_fps={fps_estimate:.2f}")


def record_clip(
    cap: cv2.VideoCapture,
    out_path: Path,
    *,
    seconds: float,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    width: int,
    height: int,
    first_frame,
    preview: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_writer(out_path, writer_fps, width, height)

    rec_start = time.perf_counter()
    frame_timestamps: list[dict[str, float | int]] = []
    append_recorded_frame(writer, first_frame, frame_timestamps, rec_start)

    print(f"Recording {seconds:.1f}s to {out_path} (writer_fps={writer_fps:.2f})")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        elapsed = time.perf_counter() - rec_start
        if elapsed >= seconds:
            break

        append_recorded_frame(writer, frame, frame_timestamps, rec_start)

        if preview:
            preview_img = frame.copy()
            fps_estimate = len(frame_timestamps) / elapsed if elapsed > 0 else 0.0
            draw_text(
                preview_img,
                f"REC frame={len(frame_timestamps) - 1} elapsed={elapsed:.2f}s fps={fps_estimate:.1f}",
                20,
                40,
            )
            cv2.imshow("record_clip", preview_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    writer.release()
    if preview:
        cv2.destroyAllWindows()

    finalize_recording(
        out_path,
        frame_timestamps,
        requested_fps=requested_fps,
        writer_fps=writer_fps,
        capture_fps_measured=capture_fps_measured,
        width=width,
        height=height,
        rec_start=rec_start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="Capture device index or video source path. Default: 0")
    parser.add_argument("--out", help="Output video path, e.g. data/raw/pitch_001.mp4")
    parser.add_argument("--seconds", type=float, default=8.0, help="Recording duration")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--warmup-frames", type=int, default=45, help="Frames to measure capture FPS before recording")
    parser.add_argument("--preview", action="store_true", help="Show a preview window while recording")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show live preview. Press r to record a clip, q to quit.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=4.5,
        help="Clip length when pressing r in preview mode. Default: 4.5",
    )
    parser.add_argument(
        "--clip-dir",
        default="data/raw",
        help="Output folder for clips recorded with r in preview mode. Default: data/raw",
    )
    parser.add_argument(
        "--flush-frames",
        type=int,
        default=45,
        help="Buffered frames to discard after each clip so live FPS recovers. Default: 45",
    )
    parser.add_argument(
        "--min-recover-fps",
        type=float,
        default=54.0,
        help="Preview rolling FPS required before the next clip. Default: 54",
    )
    parser.add_argument(
        "--recover-seconds",
        type=float,
        default=1.0,
        help="Seconds above min-recover-fps before recording is allowed again. Default: 1.0",
    )
    parser.add_argument(
        "--no-recover-gate",
        action="store_true",
        help="Allow back-to-back recordings without waiting for FPS recovery.",
    )
    parser.add_argument(
        "--verify-encoded-count",
        action="store_true",
        help="Re-read the MP4 after each clip to verify frame count (slower; can hurt FPS).",
    )
    parser.add_argument(
        "--no-async-encode",
        action="store_true",
        help="Encode on the main thread (legacy behavior; may lower preview FPS while recording).",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Preview window scale (0.5 = half resolution, less CPU). Default: 0.5",
    )
    args = parser.parse_args()

    if not args.preview_only and not args.out:
        parser.error("--out is required unless --preview-only is set")

    source = int(args.device) if str(args.device).isdigit() else args.device
    cap = open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open capture source: {args.device}")

    configure_capture(cap, args.width, args.height, args.fps)

    capture_fps_measured = measure_capture_fps(cap, warmup_frames=args.warmup_frames)
    drained = drain_capture_buffer(cap, max_frames=args.warmup_frames)
    if drained:
        print(f"Drained {drained} buffered frame(s) after warmup.")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Capture source opened but returned no frame after warmup.")

    height, width = frame.shape[:2]
    writer_fps = max(1.0, round(capture_fps_measured, 2))
    print(
        f"Capture: {width}x{height} device={args.device} "
        f"requested_fps={args.fps:.1f} measured_fps={capture_fps_measured:.2f} writer_fps={writer_fps:.2f}"
    )

    if args.preview_only:
        run_interactive_preview(
            cap,
            width=width,
            height=height,
            requested_fps=args.fps,
            writer_fps=writer_fps,
            capture_fps_measured=capture_fps_measured,
            clip_seconds=args.clip_seconds,
            clip_dir=Path(args.clip_dir),
            flush_frames=args.flush_frames,
            min_recover_fps=args.min_recover_fps,
            recover_seconds=args.recover_seconds,
            recover_gate=not args.no_recover_gate,
            verify_encoded_count=args.verify_encoded_count,
            async_encode=not args.no_async_encode,
            preview_scale=args.preview_scale,
        )
        return

    record_clip(
        cap,
        Path(args.out),
        seconds=args.seconds,
        requested_fps=args.fps,
        writer_fps=writer_fps,
        capture_fps_measured=capture_fps_measured,
        width=width,
        height=height,
        first_frame=frame,
        preview=args.preview,
    )
    cap.release()


if __name__ == "__main__":
    main()
