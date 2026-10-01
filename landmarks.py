"""MediaPipe Pose landmark layout, and small helpers for pulling one joint's
trajectory out of a (frames, 33, 3) landmark array.

Landmark array convention used throughout this project: shape
(frames, 33, 3), channels (x_px, y_px, visibility). This is MediaPipe's
33-point BlazePose topology in image-pixel space -- not MediaPipe's own
normalized [0, 1] or world-coordinate output -- because every metric here
works in pixels first and converts to metres explicitly, once, through a
single known-length scale (see pixel_scale below). Image y grows downward,
as in every image library; that convention shows up again in kinematics.py.
"""

import numpy as np

POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

NUM_LANDMARKS = 33


def joint_xy(landmarks, name):
    """(x, y) pixel series for one joint.

    landmarks: a (frames, 33, 3) or (33, 3) array.
    name: a key of POSE_LANDMARKS, e.g. "left_heel".
    Returns an (frames, 2) or (2,) array -- the last axis is (x, y).
    """
    if name not in POSE_LANDMARKS:
        raise ValueError(f"unknown landmark name: {name!r}")
    landmarks = np.asarray(landmarks, dtype=float)
    return landmarks[..., POSE_LANDMARKS[name], :2]


def joint_visibility(landmarks, name):
    """Visibility/confidence series for one joint (the third channel)."""
    if name not in POSE_LANDMARKS:
        raise ValueError(f"unknown landmark name: {name!r}")
    landmarks = np.asarray(landmarks, dtype=float)
    return landmarks[..., POSE_LANDMARKS[name], 2]


def pixel_scale(real_length_m, pixel_length):
    """Metres per pixel, from one known real-world length and its measured
    pixel length in the video -- the runner's height standing still, or a
    marker of known size placed in the scene. Everything in kinematics.py
    that needs real units takes this number, not the raw pixel length,
    so the conversion happens in exactly one place.
    """
    if real_length_m <= 0:
        raise ValueError(f"real_length_m must be positive, got {real_length_m}")
    if pixel_length <= 0:
        raise ValueError(f"pixel_length must be positive, got {pixel_length}")
    return real_length_m / pixel_length


def to_metres(series_px, metres_per_pixel):
    """Apply a pixel-to-metre scale to a position, length, or amplitude."""
    return np.asarray(series_px, dtype=float) * metres_per_pixel
