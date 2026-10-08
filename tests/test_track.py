"""Picking the runner out of a scene with other people in it."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from landmarks import joint_xy
from synth import make_synthetic_gait
from track import select_runner

FPS = 30


@pytest.fixture(scope="module")
def runner():
    return make_synthetic_gait(cadence_spm=170.0, contact_time_s=0.25, fps=FPS, seconds=6)[0]


def _scene(runner, bystander_first):
    still = runner[0].copy()
    still[:, 0] += 400.0  # stands 400 px away; ankles never move
    return [[still, r] if bystander_first else [r, still] for r in runner]


@pytest.mark.parametrize("bystander_first", [True, False])
def test_picks_the_runner_not_the_still_bystander(runner, bystander_first):
    out = select_runner(_scene(runner, bystander_first), FPS)
    assert np.array_equal(out, runner)


def test_roi_overrides_the_automatic_pick(runner):
    out = select_runner(_scene(runner, True), FPS, roi=(700, 0, 1000, 600))
    assert np.allclose(joint_xy(out, "left_hip")[:, 0], runner[0, 23, 0] + 400.0)


def test_runner_frames_the_detector_missed_come_back_nan(runner):
    scene = _scene(runner, True)
    scene[40] = scene[40][:1]  # only the bystander found in frame 40
    out = select_runner(scene, FPS)
    assert np.isnan(out[40, :, :2]).all() and (out[40, :, 2] == 0).all()
    assert np.array_equal(out[41], runner[41])


def test_says_so_when_nobody_runs():
    still = make_synthetic_gait(cadence_spm=170.0, contact_time_s=0.25, fps=FPS, seconds=4)[0][0]
    with pytest.raises(ValueError, match="periodic"):
        select_runner([[still]] * 120, FPS)
