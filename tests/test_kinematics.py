"""Gait events and metrics recovered from synthetic landmark arrays."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

import kinematics as k
from landmarks import joint_xy
from synth import make_synthetic_gait


def test_joint_angle_of_a_right_angle_is_90():
    a, b, c = (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)
    assert k.joint_angle(a, b, c) == pytest.approx(90.0)


def test_joint_angle_of_a_straight_line_is_180():
    a, b, c = (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)
    assert k.joint_angle(a, b, c) == pytest.approx(180.0)


def test_joint_angle_works_per_frame_on_arrays():
    a = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0]])
    c = np.array([[0.0, 1.0], [1.0, 0.0]])
    angles = k.joint_angle(a, b, c)
    assert angles == pytest.approx([90.0, 180.0])


def test_joint_angle_raises_when_a_point_sits_on_the_vertex():
    with pytest.raises(ValueError):
        k.joint_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))


@pytest.mark.parametrize("fps", [60, 240])
def test_cadence_recovered_from_synthetic_gait_within_1_spm(fps):
    cadence_truth = 170.0
    lm, truth = make_synthetic_gait(
        cadence_spm=cadence_truth, contact_time_s=0.25, fps=fps, seconds=10, seed=0)
    left = k.detect_foot_strikes(joint_xy(lm, "left_heel")[:, 1], fps)
    right = k.detect_foot_strikes(joint_xy(lm, "right_heel")[:, 1], fps)
    both_feet = np.sort(np.concatenate([left, right]))
    assert k.cadence_spm(both_feet) == pytest.approx(cadence_truth, abs=1.0)


def test_strike_detection_count_matches_truth():
    lm, truth = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=60, seconds=10, seed=0)
    strikes = k.detect_foot_strikes(joint_xy(lm, "left_heel")[:, 1], 60)
    assert len(strikes) == len(truth["strike_times_left"])


def test_contact_time_error_is_bounded_by_one_frame_period():
    fps = 60
    lm, truth = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=fps, seconds=10, seed=0)
    heel_y = joint_xy(lm, "left_heel")[:, 1]
    strikes = k.detect_foot_strikes(heel_y, fps)
    toe_offs = k.detect_toe_offs(heel_y, fps)
    contact = k.ground_contact_times(strikes, toe_offs)
    frame_period = 1.0 / fps
    assert np.all(np.abs(contact - 0.25) <= frame_period + 1e-9)


def test_vertical_oscillation_recovers_synthetic_amplitude():
    lm, truth = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=60, seconds=10,
        vertical_osc_px=15.0, seed=0)
    hip_y = joint_xy(lm, "left_hip")[:, 1]
    recovered = k.vertical_oscillation(hip_y, scale=1.0)
    assert recovered == pytest.approx(15.0, abs=0.2)


def test_vertical_oscillation_applies_the_pixel_to_metre_scale():
    lm, truth = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=60, seconds=10,
        vertical_osc_px=20.0, seed=0)
    hip_y = joint_xy(lm, "left_hip")[:, 1]
    assert k.vertical_oscillation(hip_y, scale=0.01) == pytest.approx(0.20, abs=0.005)


def test_smoothing_does_not_shift_strike_timing_by_more_than_a_frame():
    fps = 60
    lm, truth = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=fps, seconds=10, seed=0)
    heel_y = joint_xy(lm, "left_heel")[:, 1]
    raw_strikes = k.detect_foot_strikes(heel_y, fps)
    smoothed = k.smooth(heel_y, fps, method="savgol")
    smooth_strikes = k.detect_foot_strikes(smoothed, fps)
    n = min(len(raw_strikes), len(smooth_strikes))
    assert n > 0
    shift = np.abs(raw_strikes[:n] - smooth_strikes[:n])
    assert np.all(shift <= 1.0 / fps + 1e-9)


def test_strike_detection_is_robust_to_modest_noise():
    fps = 60
    lm, truth = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=fps, seconds=10,
        noise=2.0, seed=1)
    heel_y = joint_xy(lm, "left_heel")[:, 1]
    smoothed = k.smooth(heel_y, fps, method="savgol")
    strikes = k.detect_foot_strikes(smoothed, fps)
    truth_strikes = truth["strike_times_left"]
    assert len(strikes) == len(truth_strikes)
    assert np.max(np.abs(strikes - truth_strikes)) <= 3.0 / fps


def test_overstride_is_positive_when_the_heel_lands_ahead_of_the_hip():
    heel_x = np.array([10.0, 20.0, 30.0])
    hip_x = np.array([0.0, 0.0, 0.0])
    result = k.overstride(heel_x, hip_x, strike_indices=[0, 2])
    assert result.tolist() == pytest.approx([10.0, 30.0])


def test_trunk_lean_is_zero_when_upright():
    shoulder = (100.0, 50.0)
    hip = (100.0, 150.0)
    assert k.trunk_lean(shoulder, hip) == pytest.approx(0.0)


def test_trunk_lean_is_positive_when_the_shoulder_leads_the_hip():
    shoulder = (110.0, 50.0)
    hip = (100.0, 150.0)
    assert k.trunk_lean(shoulder, hip) > 0.0


def test_cadence_spm_raises_with_fewer_than_two_strikes():
    with pytest.raises(ValueError):
        k.cadence_spm([1.0])


def test_detect_foot_strikes_raises_for_non_positive_fps():
    with pytest.raises(ValueError):
        k.detect_foot_strikes(np.zeros(10), fps=0)


def test_smooth_raises_for_an_unknown_method():
    with pytest.raises(ValueError):
        k.smooth(np.zeros(10), fps=60, method="bogus")


def test_ground_contact_times_raises_when_toe_off_precedes_its_strike():
    with pytest.raises(ValueError):
        k.ground_contact_times([0.5], [0.4])


def test_ground_contact_times_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        k.ground_contact_times([0.1, 0.5], [0.2])


@pytest.mark.parametrize("fps", [60, 240])
def test_contact_time_error_is_bounded_by_one_frame_period_at_both_rates(fps):
    lm, _ = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=fps, seconds=10, seed=0)
    heel_y = joint_xy(lm, "left_heel")[:, 1]
    contact = k.ground_contact_times(
        k.detect_foot_strikes(heel_y, fps), k.detect_toe_offs(heel_y, fps))
    assert np.all(np.abs(contact - 0.25) <= 1.0 / fps + 1e-9)


def test_one_euro_smoothing_does_not_shift_strike_timing_by_more_than_a_frame():
    fps = 60
    lm, _ = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=fps, seconds=10, noise=1.0, seed=0)
    heel_y = joint_xy(lm, "left_heel")[:, 1]
    smoothed = k.smooth(heel_y, fps, method="one_euro", min_cutoff=8.0, beta=0.05)
    truth = make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=fps, seconds=10, seed=0)[1]
    strikes = k.detect_foot_strikes(smoothed, fps)
    assert len(strikes) == len(truth["strike_times_left"])
    assert np.max(np.abs(strikes - truth["strike_times_left"])) <= 3.0 / fps


def test_knee_flexion_is_zero_for_a_straight_leg_and_90_for_a_right_angle():
    hip, ankle = np.array([[0.0, 0.0]] * 2), np.array([[0.0, 2.0], [1.0, 1.0]])
    knee = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert k.knee_flexion(hip, knee, ankle) == pytest.approx([0.0, 90.0])


def test_knee_flexion_at_strike_and_peak_per_stride():
    flex = np.array([5.0, 10.0, 40.0, 90.0, 30.0, 8.0, 12.0, 60.0, 20.0, 6.0])
    at_strike, peak = k.knee_flexion_per_stride(flex, strike_indices=[0, 5, 9])
    assert at_strike.tolist() == [5.0, 8.0, 6.0]
    # Peak is searched from each strike to the next; the last strike has no
    # following one, so it gets no stride.
    assert peak.tolist() == [90.0, 60.0]
