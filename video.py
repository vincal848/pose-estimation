"""Video -> landmark extraction via MediaPipe Pose.

Layers, so everything but the last is testable without a model or footage:

    read_frames(path)                 -> (fps, RGB frame iterator)   [opencv]
    poses_from_frames(frames, d)      -> all poses per frame          [pure]
    track.select_runner(poses, fps)   -> (frames, 33, 3) runner array [pure]
    mediapipe_detector(model_path)    -> d: RGB frame -> (33, 3)|None [mediapipe]
    extract_landmarks(path, ...)      -> (landmarks, fps)             [glue]

A detector `d` is any callable taking one RGB frame (H, W, 3, uint8) and
returning a list of (33, 3) arrays of (x_px, y_px, visibility), one per
person found (empty if none). track.select_runner picks the runner. Tests inject stand-ins; real runs use mediapipe_detector.

Current mediapipe (1.x) ships only the Tasks API (no mp.solutions), which needs
a pose_landmarker .task model file. This repo does not bundle or download
one: pass its path (see README, "Running on a video"). mediapipe and
opencv-python are optional (requirements-video.txt) and imported lazily, so
the rest of the package never needs them.
"""

import os
from collections.abc import Callable, Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

from track import select_runner

Detector = Callable[[NDArray[np.uint8]], list[NDArray[np.float64]]]


def read_frames(path: str) -> tuple[float, Iterator[NDArray[np.uint8]]]:
    """(fps, generator of RGB uint8 frames). fps is read from the container."""
    import cv2

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"opencv could not open {path!r}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise ValueError(f"{path!r} reports no frame rate")

    def frames():
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    return fps, frames()


def poses_from_frames(frames: Iterable[NDArray[np.uint8]], detect: Detector) -> list[list[NDArray[np.float64]]]:
    """Run `detect` on each frame: poses[i] is every pose found in frame i."""
    return [[np.asarray(p, dtype=float) for p in detect(frame)] for frame in frames]


def mediapipe_detector(model_path: str, fps: float = 30.0, num_poses: int = 4, **options: object) -> Detector:
    """A detector backed by MediaPipe's PoseLandmarker (VIDEO mode, so it
    tracks between frames). Frames must be fed in order. fps only sets the
    timestamps. Extra options go to PoseLandmarkerOptions."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"pose landmarker model not found: {model_path}")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            **options,
        )
    )
    state = {"i": 0}

    def detect(frame):
        h, w = frame.shape[:2]
        ts_ms = int(round(state["i"] * 1000.0 / fps))
        state["i"] += 1
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = landmarker.detect_for_video(image, ts_ms)
        return [np.array([[p.x * w, p.y * h, p.visibility] for p in pts]) for pts in result.pose_landmarks]

    return detect


def extract_landmarks(
    path: str,
    model_path: str | None = None,
    detect: Detector | None = None,
    roi: tuple[float, float, float, float] | None = None,
    **options: object,
) -> tuple[NDArray[np.float64], float]:
    """Video file -> ((frames, 33, 3) landmark array, fps).

    Give either `detect` (see module docstring) or `model_path` for MediaPipe.
    roi=(x0, y0, x1, y1) in pixels picks the person whose hip starts inside it;
    without it the runner is the person with periodic ankle motion.
    """
    fps, frames = read_frames(path)
    if detect is None:
        if model_path is None:
            raise ValueError("pass model_path (a pose_landmarker .task file) or detect")
        detect = mediapipe_detector(model_path, fps=fps, **options)
    return select_runner(poses_from_frames(frames, detect), fps, roi=roi), fps


def save_landmarks(path: str, landmarks: NDArray[np.float64], fps: float) -> None:
    np.savez_compressed(path, landmarks=landmarks, fps=fps)


def load_landmarks(path: str) -> tuple[NDArray[np.float64], float]:
    with np.load(path) as f:
        return f["landmarks"], float(f["fps"])
