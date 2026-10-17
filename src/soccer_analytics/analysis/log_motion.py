"""Use the gimbal log as BoT-SORT's camera-motion compensation.

BoT-SORT normally estimates camera motion from each decoded frame with sparse optical flow. That is a second
camera-motion estimate at full resolution, on top of the one Stage A already runs for the pose chain. When the
gimbal log exists, the tracker reuses the chain's measured step for that warp and keeps the logged orientation as
the fallback for frames whose chain step was lost. The full-resolution optical GMC is then unnecessary.

Using the logged yaw directly as the tracker warp was tried first and was not good enough: the gimbal occasionally
recentres by several degrees between two frames while the image barely moves, and the log has no translation for
the gimbal's rotation-centre offset. On a real 80 s excerpt that fragmented 534 tracks into 723. The chain step
matches the optical warp to under a pixel, so it is the safer source; the log still gates the fast path and fills
its gaps. If the log or the fit is missing, the standard GMC is left in place.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from soccer_analytics.geometry.camera_motion import DEFAULT_FOCAL, to_reference
from soccer_analytics.geometry.gimbal_log import load_gimbal_log
from soccer_analytics.geometry.gimbal_motion import (
    _clips_for_segment,
    align_log,
    find_logs_for_clips,
    fit_pan_model,
    log_orientation,
)

LOG_FIT_FRAMES = 300  # 20 s at 15 fps: enough large rotations for the pan model, short enough to switch early


class LogMotion:
    """Per-frame logged orientation, and the affine warp BoT-SORT needs between two frames."""

    def __init__(
        self,
        yaw: np.ndarray,
        pitch: np.ndarray,
        *,
        focal0: float,
        aspect: float,
        width: int,
        height: int,
        frame_offset: int = 0,
    ) -> None:
        self.yaw = np.asarray(yaw, dtype=np.float64)
        self.pitch = np.asarray(pitch, dtype=np.float64)
        self.focal0 = float(focal0)
        self.aspect = float(aspect)
        self.width = int(width)
        self.height = int(height)
        self.frame_offset = int(frame_offset)
        self.q: np.ndarray | None = None
        self.focals: dict[int, float] = {}
        self.chain_q: dict[int, np.ndarray] = {}
        self.chain_step: dict[int, np.ndarray | None] = {}
        self.chain_ok: dict[int, bool] = {}
        self._poses: list[np.ndarray] = []
        self._ok: list[bool] = []
        self._next_fit = LOG_FIT_FRAMES
        # Sample the frame for the similarity fit. A camera rotation+zoom is not exactly a similarity, but the
        # residual over one 15 fps step is small; corners plus a centre grid keep the fit away from any one region.
        self._samples = np.asarray(
            [
                [x * self.width, y * self.height]
                for y in (0.0, 0.25, 0.5, 0.75, 1.0)
                for x in (0.0, 0.25, 0.5, 0.75, 1.0)
            ],
            dtype=np.float32,
        )

    @classmethod
    def for_source(
        cls,
        meta: dict,
        total_frames: int,
        *,
        frame_offset: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> "LogMotion | None":
        """Build the logged orientation for a Stage A segment, or ``None`` when its footage has no usable log.

        ``width``/``height`` default to the source size in ``meta``; pass the detection frame's size when Stage A
        decodes at a capped width, because BoT-SORT's warp is in the coordinates of the frame it actually sees.
        """
        try:
            clip_paths, clip_starts = _clips_for_segment(SimpleNamespace(meta=meta))
            pairs = find_logs_for_clips(clip_paths)
            if pairs is None:
                return None
            logs = [(load_gimbal_log(path), float(start)) for (path, _), start in zip(pairs, clip_starts)]
            aligned = align_log(
                logs,
                start_s=float(meta["start_s"]),
                fps=float(meta["fps"]),
                frame_count=int(total_frames),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return cls(
            aligned["yaw"],
            aligned["pitch"],
            focal0=float(meta.get("default_focal", DEFAULT_FOCAL)),
            aspect=float(height or meta["height"]) / float(width or meta["width"]),
            width=int(width or meta["width"]),
            height=int(height or meta["height"]),
            frame_offset=frame_offset,
        )

    @property
    def ready(self) -> bool:
        return self.q is not None

    def observe(self, index: int, focal: float) -> None:
        """Remember the chain focal for a frame; the logged orientation itself carries no focal."""
        self.focals[int(index)] = float(focal)

    def observe_chain(
        self,
        index: int,
        orientation: np.ndarray,
        focal: float,
        ok: bool,
        step: np.ndarray | None = None,
    ) -> None:
        """Remember the chain pose too: it is a free, per-frame motion estimate Stage A already made."""
        index = int(index)
        self.focals[index] = float(focal)
        self.chain_q[index] = np.asarray(orientation, dtype=np.float64).copy()
        self.chain_step[index] = None if step is None else np.asarray(step, dtype=np.float64).copy()
        self.chain_ok[index] = bool(ok)

    def add_pose(self, orientation: np.ndarray, ok: bool) -> None:
        """Feed one measured chain pose; once enough are collected, fit the logged pan model.

        The chain is only used for the fit, not by the warp: it tells us the physical pan axis and the scale from
        logged degrees to rotation. That is the one calibration the log cannot supply on its own.
        """
        if self.ready:
            return
        self._poses.append(np.asarray(orientation, dtype=np.float64).copy())
        self._ok.append(bool(ok))
        if len(self._poses) < self._next_fit:
            return
        self._next_fit += LOG_FIT_FRAMES
        start = self.frame_offset
        try:
            model = fit_pan_model(
                np.asarray(self._poses),
                np.asarray(self._ok, dtype=bool),
                self.yaw[start : start + len(self._poses)],
                self.pitch[start : start + len(self._poses)],
                # Tracker GMC only needs relative pan, so a short segment may fit with fewer aligned windows
                # than the stricter pose fit. The scale/rms guard below keeps a noisy fit out.
                min_samples=20,
            )
        except (ValueError, np.linalg.LinAlgError):
            model = None
        if model is not None and (abs(model.scale) < 0.6 or abs(model.scale) > 1.4 or model.rms_deg > 3.0):
            model = None
        if model is not None:
            self.q = log_orientation(self.yaw, model)

    def _warp_from(self, orientations: dict[int, np.ndarray] | np.ndarray | None, index: int) -> np.ndarray | None:
        if orientations is None or index <= 0:
            return None
        previous_focal = self.focals.get(index - 1)
        current_focal = self.focals.get(index)
        if not previous_focal or not current_focal:
            return None
        try:
            previous_q = orientations[index - 1]
            current_q = orientations[index]
            previous = to_reference(previous_q, previous_focal, self.focal0, self.aspect)
            current = to_reference(current_q, current_focal, self.focal0, self.aspect)
            step = np.linalg.inv(current) @ previous
        except (IndexError, KeyError, np.linalg.LinAlgError):
            return None
        scale = np.diag([float(self.width), float(self.width), 1.0])
        step_pixels = scale @ step @ np.diag([1.0 / self.width, 1.0 / self.width, 1.0])
        mapped = cv2.perspectiveTransform(self._samples.reshape(-1, 1, 2), step_pixels).reshape(-1, 2)
        affine, _ = cv2.estimateAffinePartial2D(self._samples, mapped, method=cv2.LMEDS)
        if affine is None or not np.all(np.isfinite(affine)):
            return None
        return affine.astype(np.float32)

    def _warp_from_step(self, index: int) -> np.ndarray | None:
        step = self.chain_step.get(index)
        if step is None:
            return None
        scale = np.diag([float(self.width), float(self.width), 1.0])
        step_pixels = scale @ step @ np.diag([1.0 / self.width, 1.0 / self.width, 1.0])
        mapped = cv2.perspectiveTransform(self._samples.reshape(-1, 1, 2), step_pixels).reshape(-1, 2)
        affine, _ = cv2.estimateAffinePartial2D(self._samples, mapped, method=cv2.LMEDS)
        if affine is None or not np.all(np.isfinite(affine)):
            return None
        return affine.astype(np.float32)

    def warp(self, index: int) -> np.ndarray | None:
        """The 2x3 affine mapping frame ``index - 1`` to ``index``, or ``None`` when neither source has one.

        The raw chain step is the same motion the optical GMC would measure, already paid for by the pose pass;
        it is used when present. The logged orientation is the fallback for a frame whose chain step was lost,
        and its warp is also what the tracker uses once the log pan model is fitted if no chain step exists.
        """
        index = int(index)
        if index <= 0:
            return None
        if not self.chain_ok.get(index, True):
            # A lost chain frame carries no measured step; the log is the only estimate we have for it.
            return self._warp_from(self.q, index)
        chained = self._warp_from_step(index)
        if chained is None:
            chained = self._warp_from(self.chain_q, index)
        if chained is not None:
            return chained
        return self._warp_from(self.q, index)


class LogGMC:
    """A BoT-SORT GMC drop-in that uses the chain/log warp instead of full-resolution optical flow."""

    method = "log"

    def __init__(self, motion: LogMotion, fallback) -> None:
        self.motion = motion
        self.fallback = fallback
        self.frame_index = -1

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        warp = self.motion.warp(self.frame_index)
        if warp is not None:
            return warp
        # No chain step and no fitted log yet: the old optical GMC is still the best estimate for this frame.
        if self.fallback is not None and not self.motion.ready:
            return self.fallback.apply(raw_frame, detections)
        return np.eye(2, 3, dtype=np.float32)


def install_log_gmc(model, motion: LogMotion) -> LogGMC | None:
    """Wrap the tracker's GMC on a model that has already created its trackers; ``None`` if it has not."""
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if not trackers:
        return None
    tracker = trackers[0]
    existing = getattr(tracker, "gmc", None)
    if isinstance(existing, LogGMC) and existing.motion is motion:
        return existing
    wrapped = LogGMC(motion, existing)
    tracker.gmc = wrapped
    return wrapped
