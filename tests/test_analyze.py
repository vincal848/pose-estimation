"""landmarks -> metrics report, checked against the synthetic generator's
known truth, including dropped and low-confidence frames."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from analyze import analyze_landmarks, format_report
from synth import make_synthetic_gait

FPS = 60


@pytest.fixture(scope="module")
def clean():
    return make_synthetic_gait(
        cadence_spm=170.0, contact_time_s=0.25, fps=FPS, seconds=12, noise=1.0, seed=0)


def test_report_recovers_cadence_contact_time_and_oscillation(clean):
    lm, truth = clean
    r = analyze_landmarks(lm, FPS)
    assert r["cadence_spm"] == pytest.approx(170.0, abs=2.0)
    assert r["ground_contact_time_s"] == pytest.approx(0.25, abs=2.0 / FPS)
    assert r["vertical_oscillation_px"] == pytest.approx(15.0, abs=1.5)
    assert r["n_strikes"] >= 2 * len(truth["strike_times_left"]) - 2


def test_report_converts_to_metres_only_when_given_a_scale(clean):
    lm, _ = clean
    assert "vertical_oscillation_m" not in analyze_landmarks(lm, FPS)
    r = analyze_landmarks(lm, FPS, m_per_px=0.002)
    assert r["vertical_oscillation_m"] == pytest.approx(r["vertical_oscillation_px"] * 0.002)


def test_report_survives_short_dropouts_and_low_confidence_frames(clean):
    lm, _ = clean
    bad = lm.copy()
    bad[100:104, :, :2] = np.nan           # detector lost the runner
    bad[100:104, :, 2] = 0.0
    bad[300:306, :, 2] = 0.1               # present but low confidence
    r = analyze_landmarks(bad, FPS)
    assert r["cadence_spm"] == pytest.approx(170.0, abs=3.0)
    assert r["ground_contact_time_s"] == pytest.approx(0.25, abs=3.0 / FPS)
    assert r["frames_interpolated"] == 10


def test_report_survives_scattered_random_frame_loss(clean):
    lm, _ = clean
    rng = np.random.default_rng(3)
    bad = lm.copy()
    lost = rng.random(len(lm)) < 0.05
    bad[lost, :, 2] = 0.0
    r = analyze_landmarks(bad, FPS)
    assert r["cadence_spm"] == pytest.approx(170.0, abs=3.0)


def test_report_does_not_invent_steps_across_a_long_gap(clean):
    lm, _ = clean
    bad = lm.copy()
    bad[200:400, :, 2] = 0.0               # ~3.3 s with nobody visible
    r = analyze_landmarks(bad, FPS)
    assert r["cadence_spm"] == pytest.approx(170.0, abs=3.0)
    assert r["frames_unusable"] == 200


def test_report_says_nothing_found_on_a_signal_free_clip():
    lm, _ = make_synthetic_gait(cadence_spm=170.0, contact_time_s=0.25, fps=FPS, seconds=8)
    lm[:, :, 1] = 300.0
    with pytest.raises(ValueError, match="no gait"):
        analyze_landmarks(lm, FPS)


def test_report_raises_when_too_few_frames_are_visible(clean):
    lm, _ = clean
    bad = lm.copy()
    bad[:, :, 2] = 0.0
    with pytest.raises(ValueError, match="visible"):
        analyze_landmarks(bad, FPS)


def test_facing_left_flips_overstride_sign():
    lm, _ = make_synthetic_gait(cadence_spm=170.0, contact_time_s=0.25, fps=FPS, seconds=8)
    right = analyze_landmarks(lm, FPS, facing="right")["overstride_px"]
    left = analyze_landmarks(lm, FPS, facing="left")["overstride_px"]
    assert left == pytest.approx(-right)


def test_format_report_lists_each_metric(clean):
    text = format_report(analyze_landmarks(clean[0], FPS))
    for word in ("cadence", "contact", "oscillation"):
        assert word in text
