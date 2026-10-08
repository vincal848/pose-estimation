"""video.py: frame reading, the detector seam, and landmark export. No
MediaPipe model and no real footage -- a tiny video is rendered on the fly
and the pose detector is a stand-in.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import video
from landmarks import NUM_LANDMARKS
from synth import make_synthetic_gait


@pytest.fixture
def clip(tmp_path):
    """A 30-frame, 20 fps, 64x48 video whose frame i is filled with value 5*i."""
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), 5 * i, dtype=np.uint8))
    writer.release()
    return path


def test_read_frames_gives_fps_and_every_frame_as_rgb(clip):
    fps, frames = video.read_frames(clip)
    frames = list(frames)
    assert fps == pytest.approx(20.0)
    assert len(frames) == 30
    assert frames[0].shape == (48, 64, 3)


def test_read_frames_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        video.read_frames(str(tmp_path / "nope.mp4"))


def test_poses_from_frames_collects_every_person_per_frame():
    answers = iter([[np.ones((NUM_LANDMARKS, 3))] * 2, []])
    out = video.poses_from_frames([0, 1], lambda _f: next(answers))
    assert [len(f) for f in out] == [2, 0]


def test_extract_landmarks_runs_a_video_through_the_injected_detector(clip):
    lm, _ = make_synthetic_gait(cadence_spm=170, contact_time_s=0.25, fps=20, seconds=1.5)
    it, seen = iter(lm), []

    def detect(frame):
        seen.append(int(frame.mean()))
        return [next(it)]

    out, fps = video.extract_landmarks(clip, detect=detect)
    assert np.array_equal(out, lm)
    assert fps == pytest.approx(20.0)
    assert seen[0] < seen[-1]  # frames arrived in order


def test_extract_landmarks_without_a_model_or_detector_explains_itself(clip):
    with pytest.raises(ValueError, match="model_path"):
        video.extract_landmarks(clip)


def test_mediapipe_detector_raises_for_a_missing_model_file(tmp_path):
    pytest.importorskip("mediapipe")
    with pytest.raises(FileNotFoundError):
        video.mediapipe_detector(str(tmp_path / "pose_landmarker.task"))


def test_landmark_export_round_trips(tmp_path):
    lm, _ = make_synthetic_gait(cadence_spm=170, contact_time_s=0.25, fps=60, seconds=1)
    path = str(tmp_path / "lm.npz")
    video.save_landmarks(path, lm, 60.0)
    lm2, fps2 = video.load_landmarks(path)
    assert np.array_equal(lm, lm2) and fps2 == 60.0
