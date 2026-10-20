"""Stage A: the heavy, run-once pass over one video segment.

Decodes on the GPU, tracks the gimbal's motion, detects people, and writes everything *raw* to disk in checkpointed
chunks. Nothing here depends on the pitch or on tracking, so it never has to be repeated when the user re-draws
pitch landmarks or tunes tracking: those are Stage B and take seconds.

Output directory layout (one per segment)::

    meta.json          probe info, analysis fps/width, status
    status.json        live progress for the dashboard (written atomically)
    chunk_00000.npz    frames [0, chunk_frames)    camera state + detections
    ...

Each chunk holds, per analyzed frame ``i``: time, ok flag, inlier ratio, the *raw* camera step (so the focal length
can be recalibrated without re-reading video), and the detection arrays with a ``det_frame`` index.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from soccer_analytics.analysis.kit import DESCRIPTOR_SIZE, frame_grass_window, kit_descriptor
from soccer_analytics.analysis.log_motion import LogMotion, install_log_gmc
from soccer_analytics.geometry.camera_motion import (
    DEFAULT_FOCAL,
    CameraMotionTracker,
    RotationChain,
    overlay_mask,
)
from soccer_analytics.ingest.source import as_source, open_reader, probe_source

# 60 fps and 30 fps sources both divide cleanly by 15, so every analysis sample is an actual source frame - and a
# 15 fps sample captures a fast pass well inside the ball tracker's association gate.
ANALYSIS_FPS = 15.0
MOTION_WIDTH = 960  # motion is estimated on a small copy; detection uses the full analysis frame
DETECT_WIDTH = 0  # 0 = decode and detect at the source's own width (full resolution); see `analyze_segment`
FLOOR_WIDTH = 1920  # the frame width MIN_PERSON_HEIGHT_PX was measured at; the size floor scales with the width
CHUNK_FRAMES = 300  # 20 s at 15 fps; small checkpoints, so an interruption loses little
MIN_PERSON_HEIGHT_PX = 14  # at FLOOR_WIDTH; smaller boxes are far-side noise
PERSON_CONF = 0.25
SCHEMA_VERSION = 2  # v2 adds det_track (BoT-SORT identity per detection)
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/soccer_analytics/analysis/stage_a.py -> repo root
MODELS_DIR = REPO_ROOT / "data" / "models"  # where a locally-supplied checkpoint is looked for

# BoT-SORT with appearance re-identification. The camera whips between the ends of the pitch and players cross
# constantly, so IoU/motion association alone mixes identities; the ReID encoder's appearance embedding is what
# keeps a label on the same shirt through a crossing or a short occlusion. The buffer is sized in *seconds* and
# scaled to the analysis rate, the same window Stage B's stitcher uses, so the two layers agree on how long a
# player may vanish and still be the same person at any frame rate.
TRACK_BUFFER_S = 6.0
TRACKER_CONFIG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    # track_buffer is added per rate by tracker_config() - it is a duration, not a frame count.
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.8,
    "with_reid": True,
    "model": "auto",
}


def tracker_config(fps: float) -> dict:
    """The BoT-SORT config for one analysis rate: the identity buffer is the 6 s window, in frames."""
    config = dict(TRACKER_CONFIG)
    config["track_buffer"] = max(1, int(round(TRACK_BUFFER_S * float(fps))))
    return config


def tracker_config_path(fps: float) -> str:
    """The BoT-SORT config as a file Ultralytics' tracker loader accepts, written once per process and rate.

    ``model.track(tracker=...)`` takes a YAML path (or a built-in name); the ReID switches are not predict
    arguments, so the config has to travel as a file. A temp file per process and rate keeps the repo
    read-only-friendly, and JSON is valid YAML for a flat mapping, so the same atomic writer the chunks use
    serves here.
    """
    path = Path(tempfile.gettempdir()) / f"soccer_botsort_{os.getpid()}_{float(fps):g}.yaml"
    if not path.exists():
        _write_json_atomic(path, tracker_config(fps))
    return str(path)


@dataclass(frozen=True)
class SegmentConfig:
    """One segment's analysis settings.

    ``detect_width = 0`` (the default) is full resolution: decode and detect at the source's own width. On the
    reference 4K footage that finds ~40 players per frame against ~33 at the old 1920 default - and ~73% more
    far-side players (the 14-40 px band at 1920) - for roughly 3.5x the detection time, which is why the width
    can be pinned lower per run (``--width``).
    """

    fps: float = ANALYSIS_FPS
    motion_width: int = MOTION_WIDTH
    detect_width: int = DETECT_WIDTH
    chunk_frames: int = CHUNK_FRAMES
    confidence: float = PERSON_CONF
    weights: str = "yolov8n.pt"  # stock COCO name: Ultralytics fetches it on first use, so a fresh clone runs
    device: str | int = 0
    # When a gimbal log is found, reuse the camera-motion step Stage A already measured for BoT-SORT's
    # compensation instead of making the tracker estimate it again at full resolution. The log fills gaps where
    # the chain step was lost. Off means the tracker keeps its own optical-flow GMC.
    use_log_motion: bool = True


def resolve_weights(weights: str | None = None) -> str:
    """The weights to actually load, preferring a locally-supplied model over the stock download.

    The default used to be ``data/models/yolov8n-coco-baseline.pt``, a file that is gitignored (weights are
    fetched again rather than stored) and that Ultralytics' downloader does not recognize by name - so a fresh
    clone died with a bare ``FileNotFoundError`` at ``YOLO(...)`` and no way to recover. The default is now the
    stock ``yolov8n.pt``, which Ultralytics fetches itself. With no explicit ``--weights``, a checkpoint under
    ``data/models/`` is still honored - the most recently modified ``*.pt`` there wins, because that is where a
    fine-tuned model for this sideline camera would be dropped. Person detection is the only class Stage A needs,
    so stock COCO works - but a real fine-tuned checkpoint should detect far-side players better.
    """
    name = weights or SegmentConfig.weights
    if name and Path(name).exists():
        return name
    # Only the *default* falls back to a local checkpoint: an explicit path is the caller's choice, and a stock
    # name is a deliberate "give me exactly this model" - silently swapping either for whatever sits in
    # data/models would make a run's weights depend on what happens to be on disk.
    if weights is None:
        local = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime) if MODELS_DIR.exists() else []
        if local:
            return str(local[-1])
    return name  # a stock name Ultralytics can download, or the caller's own (missing) path, reported as such


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def segment_dir_for(video_path: str | Path, root: str | Path, *, window_label: str | None = None) -> Path:
    """Stable per-source output directory; keyed by name + size so a replaced source is not mistaken for a done one.

    ``window_label`` separates runs over different parts of the same video - the two halves of a game, say. Stored
    meta refuses a different window in the same directory (correctly), so each window needs its own directory; the
    label must be filename-safe, and :meth:`soccer_analytics.analysis.game.GameRecord.window_label` produces one.

    A ``game.json`` manifest keys its segments by the game id instead of a file stat: the id already carries the
    clips' total size, so a replaced clip is still a different game.
    """
    source = as_source(video_path)
    name = f"{source.key_name}_{source.key_size}" if source.key_size else source.key_name
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    if window_label:
        safe = f"{safe}__{window_label}"
    return Path(root) / safe


def chunk_path(directory: Path, index: int) -> Path:
    return directory / f"chunk_{index:05d}.npz"


def completed_chunks(directory: Path) -> int:
    """Number of leading, fully-written, readable chunks (a torn last file is not counted)."""
    count = 0
    while chunk_path(directory, count).exists():
        try:
            with np.load(chunk_path(directory, count)) as data:
                _ = data["time"]
        except Exception:
            break
        count += 1
    return count


def read_status(directory: Path) -> dict | None:
    path = Path(directory) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def resolve_window(start_s: float, duration_s: float | None, source_duration_s: float) -> tuple[float, float]:
    """The analyzed ``[start, end)`` window in source seconds.

    ``duration_s`` of ``None`` or ``0`` means "to the end of the video from the offset" - the dashboard's default,
    so picking a video and pressing run analyzes everything unless a shorter length is typed in.
    """
    start = max(0.0, float(start_s))
    source_end = max(0.0, float(source_duration_s))
    if duration_s is None or float(duration_s) <= 0:
        return start, max(start, source_end)
    return start, max(start, min(source_end, start + float(duration_s)))


def _save_chunk(directory: Path, index: int, rows: dict) -> None:
    path = chunk_path(directory, index)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **rows)
    os.replace(tmp, path)


def _empty_rows() -> dict[str, list]:
    return {
        k: []
        for k in ("time", "ok", "inlier", "step", "focal", "det_frame", "det_box", "det_conf", "det_kit", "det_track")
    }


def _pack_rows(rows: dict[str, list]) -> dict[str, np.ndarray]:
    n_det = len(rows["det_frame"])
    return {
        "time": np.asarray(rows["time"], dtype=np.float64),
        "ok": np.asarray(rows["ok"], dtype=bool),
        "inlier": np.asarray(rows["inlier"], dtype=np.float32),
        "step": np.asarray(rows["step"], dtype=np.float64).reshape(-1, 3, 3),
        "focal": np.asarray(rows["focal"], dtype=np.float32),
        "det_frame": np.asarray(rows["det_frame"], dtype=np.int32),
        "det_box": np.asarray(rows["det_box"], dtype=np.float32).reshape(n_det, 4),
        "det_conf": np.asarray(rows["det_conf"], dtype=np.float32),
        "det_kit": np.asarray(rows["det_kit"], dtype=np.float32).reshape(n_det, DESCRIPTOR_SIZE),
        "det_track": np.asarray(rows["det_track"], dtype=np.int32),
    }


def detect_people(model, frame: np.ndarray, config: SegmentConfig, ignore: np.ndarray) -> list[tuple]:
    """COCO person boxes ``(x1, y1, x2, y2, conf)`` in ``frame`` pixels, minus burned-in overlays and tiny boxes."""
    result = model.predict(
        frame, imgsz=config.detect_width, conf=config.confidence, classes=[0], device=config.device, verbose=False
    )[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    out = []
    height, width = frame.shape[:2]
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        if y2 - y1 < MIN_PERSON_HEIGHT_PX * width / FLOOR_WIDTH:
            continue
        foot_x, foot_y = int(np.clip((x1 + x2) / 2, 0, width - 1)), int(np.clip(y2 - 1, 0, height - 1))
        if ignore[foot_y, foot_x] == 0:  # the box's foot sits on the logo/clock overlay
            continue
        out.append((x1 / width, y1 / width, x2 / width, y2 / width, float(conf)))  # normalized by frame width
    return out


def track_people(
    model,
    frame: np.ndarray,
    config: SegmentConfig,
    ignore: np.ndarray,
    *,
    log_motion: LogMotion | None = None,
    frame_index: int = -1,
) -> list[tuple]:
    """Tracked person boxes ``(x1, y1, x2, y2, conf, track_id)``; the identity is BoT-SORT's, not the caller's.

    Tracking runs on the same frame the detector sees, so the boxes are the detector's own - the tracker only
    attaches identities to them. ``track_id`` is ``-1`` for a detection the tracker did not carry over (the first
    frame of a stream, or a detection below the tracker's own threshold), and Stage B treats those as fresh
    observations rather than continuations.

    The tracker is stateful and must persist across frames of one segment (``persist=True``); it is created on the
    first call and reused, which is what makes an identity survive a chunk boundary. When ``log_motion`` is
    available, its warp replaces the tracker's own full-resolution optical-flow GMC after the model is fitted.
    """
    gmc = getattr(model, "_log_gmc", None)
    if gmc is not None:
        gmc.frame_index = int(frame_index)
    result = model.track(
        frame,
        persist=True,
        tracker=tracker_config_path(config.fps),
        imgsz=config.detect_width,
        conf=config.confidence,
        classes=[0],
        device=config.device,
        verbose=False,
    )[0]
    if log_motion is not None and getattr(model, "_log_motion", None) is not log_motion:
        # The first call above is what makes the predictor create its tracker. From the next frame on, the
        # tracker's GMC is the logged warp (or the old GMC until the pan model is ready).
        installed = install_log_gmc(model, log_motion)
        if installed is not None:
            model._log_gmc = installed
            model._log_motion = log_motion
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    ids = result.boxes.id
    ids = ids.cpu().numpy() if ids is not None else np.full(len(boxes), -1.0)
    out = []
    height, width = frame.shape[:2]
    for (x1, y1, x2, y2), conf, tid in zip(boxes, confs, ids):
        if y2 - y1 < MIN_PERSON_HEIGHT_PX * width / FLOOR_WIDTH:
            continue
        foot_x, foot_y = int(np.clip((x1 + x2) / 2, 0, width - 1)), int(np.clip(y2 - 1, 0, height - 1))
        if ignore[foot_y, foot_x] == 0:  # the box's foot sits on the logo/clock overlay
            continue
        out.append((x1 / width, y1 / width, x2 / width, y2 / width, float(conf), int(tid)))
    return out


def analyze_segment(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    config: SegmentConfig | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    model=None,
    should_stop=lambda: False,
    on_progress=lambda status: None,
) -> dict:
    """Runs (or resumes) Stage A for one segment. Returns the final status dict.

    ``video_path`` is a video file, or a ``game.json`` manifest: both are read through the clip-aware source
    resolver, so a game that was never merged analyzes exactly like a single file, across its clip joins.
    """
    config = config or SegmentConfig()
    source = as_source(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_source(source)
    if not config.detect_width:
        # Full resolution: decode and detect at the source's own width, which is what "4K detection" means for a
        # 4K recording. The resolved number goes into meta, so resuming, the kit refresh and the dashboard's
        # conflict check all see the width that was actually used - not the sentinel 0.
        config = replace(config, detect_width=probe.width)
    start_s, end_s = resolve_window(start_s, duration_s, probe.duration_s)
    total_frames = max(1, int((end_s - start_s) * config.fps))
    total_chunks = -(-total_frames // config.chunk_frames)

    meta = {
        "schema": SCHEMA_VERSION,
        "video": source.spec,
        "width": probe.width,
        "height": probe.height,
        "source_fps": probe.fps,
        "duration_s": probe.duration_s,
        "start_s": start_s,
        "end_s": end_s,
        "fps": config.fps,
        "motion_width": config.motion_width,
        "detect_width": config.detect_width,
        "chunk_frames": config.chunk_frames,
        "total_frames": total_frames,
        "total_chunks": total_chunks,
        "weights": config.weights,
        "default_focal": DEFAULT_FOCAL,
    }
    existing = out_dir / "meta.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        compatible = all(previous.get(k) == meta[k] for k in ("schema", "fps", "motion_width", "detect_width", "chunk_frames", "start_s", "end_s"))
        if not compatible:
            raise ValueError(f"{out_dir} holds results with different settings; choose a new output directory")
    _write_json_atomic(existing, meta)

    if model is None:
        from ultralytics import YOLO

        model = YOLO(resolve_weights(config.weights))
    for attribute in ("_log_gmc", "_log_motion"):
        if hasattr(model, attribute):
            delattr(model, attribute)

    first_chunk = completed_chunks(out_dir)
    started = time.time()
    status = {"state": "running", "chunk": first_chunk, "total_chunks": total_chunks, "frames_done": first_chunk * config.chunk_frames,
              "total_frames": total_frames, "fps": 0.0, "eta_s": None, "pid": os.getpid(), "updated": started, "lost": 0, "error": None}

    def publish(**changes) -> dict:
        status.update(changes, updated=time.time())
        _write_json_atomic(out_dir / "status.json", status)
        on_progress(dict(status))
        return status

    if first_chunk >= total_chunks:
        return publish(state="done", chunk=total_chunks, frames_done=total_frames, eta_s=0.0)
    publish()

    chain = RotationChain(DEFAULT_FOCAL, probe.height / probe.width)
    motion_height = int(round(config.motion_width * probe.height / probe.width)) // 2 * 2
    resume_overlap = 1 if first_chunk > 0 else 0
    if first_chunk > 0:
        chain = _restore_chain(out_dir, first_chunk, probe)
    tracker = CameraMotionTracker((motion_height, config.motion_width), chain=chain)
    detect_height = int(round(config.detect_width * probe.height / probe.width)) // 2 * 2
    log_motion = (
        LogMotion.for_source(
            meta,
            total_frames,
            frame_offset=first_chunk * config.chunk_frames,
            width=config.detect_width,
            height=detect_height,
        )
        if config.use_log_motion
        else None
    )

    # On resume, decode one extra frame *before* the resume point. It is the frame the restored chain belongs to
    # (the last frame of the previous chunk), so seeding the tracker with it keeps the motion across the chunk
    # boundary instead of measuring the first new frame against itself.
    frame_step = 1.0 / config.fps
    reader_start = start_s + first_chunk * config.chunk_frames / config.fps - resume_overlap * frame_step
    reader = open_reader(
        source, fps=config.fps, width=config.detect_width,
        start_s=max(0.0, reader_start), duration_s=end_s - max(0.0, reader_start),
    )
    ignore_mask = None
    rows = _empty_rows()
    chunk_index, frames_in_chunk = first_chunk, 0
    lost_total = 0
    to_skip = resume_overlap

    try:
        for source_time, frame in reader.frames():
            if should_stop():
                publish(state="stopped")
                return status
            if ignore_mask is None:
                ignore_mask = overlay_mask(frame.shape[:2])
            small = cv2.resize(frame, (config.motion_width, motion_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if to_skip:
                tracker.seed_good_frame(gray)
                to_skip -= 1
                continue
            state = tracker.update(gray)
            lost_total += 0 if state.ok else 1

            frame_idx = frames_in_chunk
            analysis_index = first_chunk * config.chunk_frames + frame_idx
            rows["time"].append(source_time)
            rows["ok"].append(state.ok)
            rows["inlier"].append(state.inlier_ratio)
            rows["step"].append(state.step if state.step is not None else np.eye(3))
            rows["focal"].append(state.focal)
            if log_motion is not None:
                log_motion.observe_chain(analysis_index, tracker.chain.q, state.focal, state.ok, state.step)
                log_motion.add_pose(tracker.chain.q, state.ok)

            grass = frame_grass_window(frame)
            tracked = track_people(
                model,
                frame,
                config,
                ignore_mask,
                log_motion=log_motion,
                frame_index=analysis_index,
            )
            for x1, y1, x2, y2, conf, tid in tracked:
                width = frame.shape[1]
                rows["det_frame"].append(frame_idx)
                rows["det_box"].append((x1, y1, x2, y2))
                rows["det_conf"].append(conf)
                rows["det_track"].append(tid)
                rows["det_kit"].append(kit_descriptor(frame, (x1 * width, y1 * width, x2 * width, y2 * width), grass))

            frames_in_chunk += 1
            if frames_in_chunk >= config.chunk_frames:
                _save_chunk(out_dir, chunk_index, _pack_rows(rows))
                chunk_index += 1
                done = min(chunk_index * config.chunk_frames, total_frames)
                elapsed = max(time.time() - started, 1e-6)
                rate = (done - first_chunk * config.chunk_frames) / elapsed
                publish(chunk=chunk_index, frames_done=done, fps=rate, eta_s=(total_frames - done) / max(rate, 1e-6), lost=lost_total)
                rows, frames_in_chunk = _empty_rows(), 0
        if frames_in_chunk:
            _save_chunk(out_dir, chunk_index, _pack_rows(rows))
            chunk_index += 1
        publish(state="done", chunk=chunk_index, frames_done=total_frames, eta_s=0.0, lost=lost_total)
    except Exception as exc:  # leave a readable status for the dashboard instead of a silently dead job
        publish(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    return status


def _restore_chain(directory: Path, chunks: int, probe) -> RotationChain:
    """Rebuild the camera chain by replaying every stored accepted step up to the last completed chunk.

    Lost frames and the very first frame store ``step = None`` semantics as an identity step *with ok=False / init*;
    those carry no motion, so only frames flagged ``ok`` that actually had a measured step are replayed. Frame 0 of the
    segment is the only ok frame without a measured step, and it is identified by being the first row.
    """
    chain = RotationChain(DEFAULT_FOCAL, probe.height / probe.width)
    for index in range(chunks):
        with np.load(chunk_path(directory, index)) as data:
            for row, (ok, step) in enumerate(zip(data["ok"], data["step"])):
                if not ok or (index == 0 and row == 0):
                    continue
                q, focal, _ = chain.preview(step)
                chain.commit(q, focal)
    return chain


# --------------------------------------------------------------------------------------------------------------
# Reading results back
# --------------------------------------------------------------------------------------------------------------


@dataclass
class SegmentData:
    meta: dict
    time: np.ndarray  # (F,) source seconds
    ok: np.ndarray  # (F,)
    inlier: np.ndarray  # (F,)
    step: np.ndarray  # (F, 3, 3) raw normalized step (last good frame -> this frame); identity if none
    focal: np.ndarray  # (F,) chain focal at analysis time
    det_frame: np.ndarray  # (D,) global frame index
    det_box: np.ndarray  # (D, 4) x1,y1,x2,y2 normalized by frame width
    det_conf: np.ndarray  # (D,)
    det_kit: np.ndarray  # (D, DESCRIPTOR_SIZE)
    det_track: np.ndarray  # (D,) BoT-SORT identity, -1 where the tracker had none (v1 chunks, or a fresh track)
    directory: Path | None = None  # where the segment's chunks live (set by load_segment; None for hand-built data)

    @property
    def aspect(self) -> float:
        return self.meta["height"] / self.meta["width"]


def _resolve_video_path(stored: str, directory: Path) -> str:
    """The stored source path in its usable form for the segment directory it was recorded beside.

    ``meta.json`` records the source as the machine that ran the analysis saw it - an absolute path - and an
    archive copied to another root, or a machine mounting the same footage elsewhere, keeps the layout without
    the prefix. The same file name sits either in the segment's analysis directory, one level up (the
    ``game.json`` manifest of the never-merged workflow), or at the footage root, three levels up (the combined
    video and the raw clips). Those are tried in that order, and the stored path survives when neither exists,
    so callers receive what was written rather than a silent miss.
    """
    path = Path(stored)
    if not stored or path.exists():
        return stored
    parents = directory.resolve().parents
    for index in (1, 3):  # the analysis directory holding game.json; the footage folder holding the clips
        if index >= len(parents):
            continue
        candidate = parents[index] / path.name
        if candidate.exists():
            return str(candidate)
    return stored


def load_segment(directory: str | Path) -> SegmentData:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    # Results are read on machines that may see the footage under a different root than the one that ran the
    # analysis, so the stored video path is re-pointed the way the match record's own paths are.
    meta["video"] = _resolve_video_path(str(meta.get("video") or ""), directory)
    chunks = completed_chunks(directory)
    if chunks == 0:
        raise FileNotFoundError(f"no completed chunks in {directory}")
    keys = ("time", "ok", "inlier", "step", "focal", "det_frame", "det_box", "det_conf", "det_kit", "det_track")
    parts: dict[str, list] = {k: [] for k in keys}
    offset = 0
    for index in range(chunks):
        with np.load(chunk_path(directory, index)) as data:
            for key in keys:
                if key not in data.files:
                    # A v1 chunk predates the tracker column: every detection was its own identity then, which is
                    # exactly what Stage B's online pass would have concluded anyway.
                    parts[key].append(np.full(len(data["det_frame"]), -1, dtype=np.int32))
                    continue
                value = data[key]
                parts[key].append(value + offset if key == "det_frame" else value)
            offset += len(data["time"])
    joined = {k: np.concatenate(v) if v else np.empty(0) for k, v in parts.items()}
    return SegmentData(meta=meta, **joined, directory=directory.resolve())
