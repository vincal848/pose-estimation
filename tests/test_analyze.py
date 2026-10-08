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


def test_events_touching_a_long_gap_are_discarded_not_interpolated_into_a_contact():
    # The gap swallows one foot's whole swing, so interpolating its heel
    # joins the two neighbouring contacts into one long fake contact.
    lm, truth = make_synthetic_gait(cadence_spm=170.0, contact_time_s=0.25, fps=FPS, seconds=12)
    s = truth["strike_times_left"]
    a, b = int((s[3] + 0.26) * FPS), int((s[4] + 0.02) * FPS)
    bad = lm.copy()
    bad[a:b, :, 2] = 0.0
    r = analyze_landmarks(bad, FPS)
    # Every strike whose contact or neighbourhood overlaps the gap is gone.
    assert r["n_strikes"] <= analyze_landmarks(lm, FPS)["n_strikes"] - 3


def test_cadence_is_not_quantised_to_whole_frames_at_30_fps():
    # Step interval 10.84 frames: a median of integer-frame gaps lands on 11
    # (163 spm), 2.7 spm off. Found on the real 30 fps clip.
    lm, _ = make_synthetic_gait(cadence_spm=166.0, contact_time_s=0.25, fps=30, seconds=20, noise=0.5, seed=1)
    assert analyze_landmarks(lm, 30)["cadence_spm"] == pytest.approx(166.0, abs=1.5)


def test_one_garbage_foot_does_not_corrupt_cadence_or_contact_time():
    # On the real clip a foreground object hid the far leg; its heel track
    # was noise while visibility stayed high.
    lm, _ = make_synthetic_gait(cadence_spm=170.0, contact_time_s=0.25, fps=FPS, seconds=12, seed=0)
    rng = np.random.default_rng(0)
    lm[:, 30, 1] = 600.0 - rng.uniform(0, 120, len(lm))  # right_heel
    r = analyze_landmarks(lm, FPS)
    assert r["cadence_spm"] == pytest.approx(170.0, abs=2.0)
    assert r["ground_contact_time_s"] == pytest.approx(0.25, abs=2.0 / FPS)


def test_contact_time_survives_the_rounded_heel_traces_of_real_footage():
    # Real heel tracks ramp into and out of stance instead of being flat;
    # a tight velocity gate then keeps only the middle of the contact.
    lm, _ = make_synthetic_gait(cadence_spm=170.0, contact_time_s=0.27, fps=30, seconds=12, noise=1.0, seed=0)
    for i in (27, 28, 29, 30):  # left/right ankle and heel
        lm[:, i, 1] = np.convolve(np.pad(lm[:, i, 1], 2, mode="edge"), np.ones(5) / 5, "valid")
    assert analyze_landmarks(lm, 30)["ground_contact_time_s"] == pytest.approx(0.27, abs=1.5 / 30)
