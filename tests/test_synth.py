"""The synthetic gait generator produces the shape and invariants its
ground truth claims, independent of any particular kinematics function.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from landmarks import NUM_LANDMARKS
from synth import make_synthetic_gait


def test_make_synthetic_gait_returns_the_documented_array_shape():
    lm, truth = make_synthetic_gait(cadence_spm=160, contact_time_s=0.2, fps=60, seconds=5)
    assert lm.shape == (300, NUM_LANDMARKS, 3)


def test_make_synthetic_gait_visibility_channel_is_always_one():
    lm, truth = make_synthetic_gait(cadence_spm=160, contact_time_s=0.2, fps=60, seconds=5)
    assert np.all(lm[:, :, 2] == 1.0)


def test_make_synthetic_gait_is_reproducible_with_a_seed():
    lm1, _ = make_synthetic_gait(
        cadence_spm=160, contact_time_s=0.2, fps=60, seconds=5, noise=1.0, seed=42)
    lm2, _ = make_synthetic_gait(
        cadence_spm=160, contact_time_s=0.2, fps=60, seconds=5, noise=1.0, seed=42)
    assert np.array_equal(lm1, lm2)


def test_make_synthetic_gait_noise_actually_perturbs_the_landmarks():
    lm1, _ = make_synthetic_gait(cadence_spm=160, contact_time_s=0.2, fps=60, seconds=5, noise=0.0)
    lm2, _ = make_synthetic_gait(
        cadence_spm=160, contact_time_s=0.2, fps=60, seconds=5, noise=3.0, seed=1)
    assert not np.array_equal(lm1, lm2)


def test_make_synthetic_gait_truth_strikes_are_contact_time_apart_from_their_toe_offs():
    _, truth = make_synthetic_gait(cadence_spm=160, contact_time_s=0.2, fps=60, seconds=5)
    gap = truth["toe_off_times_left"] - truth["strike_times_left"]
    assert gap == pytest.approx(0.2)


@pytest.mark.parametrize("fps,seconds,cadence_spm", [(0, 5, 160), (60, 0, 160), (60, 5, 0)])
def test_make_synthetic_gait_raises_for_non_positive_inputs(fps, seconds, cadence_spm):
    with pytest.raises(ValueError):
        make_synthetic_gait(cadence_spm=cadence_spm, contact_time_s=0.2, fps=fps, seconds=seconds)


def test_make_synthetic_gait_raises_when_contact_time_does_not_fit_the_cadence():
    # At 300 spm the stride time is 0.4 s; a 0.5 s contact phase cannot fit.
    with pytest.raises(ValueError):
        make_synthetic_gait(cadence_spm=300, contact_time_s=0.5, fps=60, seconds=5)
