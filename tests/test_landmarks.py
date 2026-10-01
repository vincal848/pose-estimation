"""Landmark indexing and pixel-to-metre scale helpers."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

import landmarks as lm


def test_pose_landmarks_has_all_33_mediapipe_points():
    assert len(lm.POSE_LANDMARKS) == lm.NUM_LANDMARKS == 33
    assert set(lm.POSE_LANDMARKS.values()) == set(range(33))


def test_pose_landmarks_known_indices_match_mediapipe():
    # Spot-check against the published BlazePose topology, since a mixup
    # here would silently feed every metric the wrong joint.
    assert lm.POSE_LANDMARKS["nose"] == 0
    assert lm.POSE_LANDMARKS["left_shoulder"] == 11
    assert lm.POSE_LANDMARKS["right_shoulder"] == 12
    assert lm.POSE_LANDMARKS["left_hip"] == 23
    assert lm.POSE_LANDMARKS["right_hip"] == 24
    assert lm.POSE_LANDMARKS["left_heel"] == 29
    assert lm.POSE_LANDMARKS["right_heel"] == 30


def test_joint_xy_pulls_the_right_column_out_of_a_landmark_array():
    landmarks = np.zeros((5, 33, 3))
    landmarks[:, lm.POSE_LANDMARKS["left_heel"], 0] = np.arange(5)
    landmarks[:, lm.POSE_LANDMARKS["left_heel"], 1] = np.arange(5) * 10
    xy = lm.joint_xy(landmarks, "left_heel")
    assert xy.shape == (5, 2)
    assert xy[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert xy[:, 1].tolist() == [0, 10, 20, 30, 40]


def test_joint_xy_raises_for_an_unknown_joint_name():
    landmarks = np.zeros((5, 33, 3))
    with pytest.raises(ValueError):
        lm.joint_xy(landmarks, "left_pinky_toe")


def test_pixel_scale_converts_a_known_height_to_metres_per_pixel():
    scale = lm.pixel_scale(real_length_m=1.8, pixel_length=720)
    assert scale == pytest.approx(1.8 / 720)
    assert lm.to_metres(720, scale) == pytest.approx(1.8)


@pytest.mark.parametrize("real_length_m,pixel_length", [(0, 100), (-1, 100), (1, 0), (1, -5)])
def test_pixel_scale_raises_for_non_positive_inputs(real_length_m, pixel_length):
    with pytest.raises(ValueError):
        lm.pixel_scale(real_length_m, pixel_length)
