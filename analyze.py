"""Landmark array -> gait metrics report. Pure numpy: no video, no files.

Frames where a key joint is missing (NaN) or under min_vis are "bad". Short
runs of bad frames (<= max_gap_s) are linearly interpolated; longer runs are
interpolated only so the filters have a signal, and any gait event touching
one is discarded rather than trusted.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

import kinematics as k
from landmarks import joint_visibility, joint_xy

# Looser than kinematics' default: real heel traces ramp into stance, and the
# tight gate kept only the middle of each contact (found on the real clip).
VEL_FRAC = 1.5

_KEY_JOINTS = [f"{s}_{j}" for s in ("left", "right") for j in ("hip", "knee", "ankle", "heel", "shoulder")]


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def _stride_cadence(strikes: NDArray[np.float64]) -> tuple[float, float]:
    """(steps/min, fraction of intervals that look like a real stride) from
    one foot's strike times. Each interval counts as k strides, k being the
    whole number nearest to interval/median; intervals that are not near a
    whole number (a spurious or missing strike) are ignored. Summing k over
    summed time keeps full-clip precision instead of whole-frame quantised
    medians. Returns (nan, 0) with fewer than three strikes."""
    if len(strikes) < 3:
        return float("nan"), 0.0
    d = np.diff(strikes)
    ratio = d / np.median(d)
    k_strides = np.round(ratio)
    ok = (k_strides >= 1) & (np.abs(ratio - k_strides) < 0.25)
    return 120.0 * float(k_strides[ok].sum() / d[ok].sum()), float(ok.mean())


def analyze_landmarks(
    landmarks: NDArray[np.float64],
    fps: float,
    m_per_px: float | None = None,
    facing: Literal["right", "left"] = "right",
    min_vis: float = 0.5,
    max_gap_s: float = 0.2,
) -> dict[str, float]:
    """Metrics from a (frames, 33, 3) landmark array. Cadence is both-feet
    steps/min and contact time the median, both from the foot with the most
    regular strikes. Pixel metrics carry a _px
    suffix; with m_per_px (see landmarks.pixel_scale) metre versions are
    added. facing says which way the runner moves in the image."""
    lm = np.asarray(landmarks, dtype=float)
    n = len(lm)
    good = np.ones(n, dtype=bool)
    for name in _KEY_JOINTS:
        good &= np.isfinite(joint_xy(lm, name)).all(axis=1) & (joint_visibility(lm, name) >= min_vis)
    if good.mean() < 0.5:
        raise ValueError(f"only {good.mean():.0%} of frames have the key joints visible")

    max_gap = int(round(max_gap_s * fps))
    long_gap = np.zeros(n, dtype=bool)
    for a, b in _runs(~good):
        if b - a > max_gap:
            long_gap[a:b] = True
    idx = np.arange(n)

    def series(name: str) -> NDArray[np.float64]:
        xy = joint_xy(lm, name)
        return np.column_stack([np.interp(idx, idx[good], xy[good, c]) for c in (0, 1)])

    sign = 1.0 if facing == "right" else -1.0
    hip = (series("left_hip") + series("right_hip")) / 2
    hip_s = k.smooth(hip[:, 1], fps)
    keep = ~long_gap

    strikes, contacts, overstride, flex_strike, flex_peak = [], [], [], [], []
    for side in ("left", "right"):
        heel, knee, ankle, hip_side = (series(f"{side}_{j}") for j in ("heel", "knee", "ankle", "hip"))
        heel_y = k.smooth(heel[:, 1], fps)
        s, t = k.detect_foot_strikes(heel_y, fps, vel_frac=VEL_FRAC), k.detect_toe_offs(heel_y, fps, vel_frac=VEL_FRAC)
        si, ti = np.round(s * fps).astype(int), np.round(t * fps).astype(int)
        ok = np.array([not long_gap[max(a - 1, 0): b + 2].any() for a, b in zip(si, ti)], dtype=bool)
        s, t, si = s[ok], t[ok], si[ok]
        strikes.append(s)
        contacts.append(k.ground_contact_times(s, t))
        overstride.append(sign * k.overstride(heel[:, 0], hip_side[:, 0], si))
        if len(si) >= 2:
            flex = k.knee_flexion(hip_side, knee, ankle)
            at, peak = k.knee_flexion_per_stride(flex, si)
            flex_strike.append(at)
            flex_peak.append(peak)

    all_strikes = np.sort(np.concatenate(strikes))
    # Cadence and contact time come from the foot with the most regular
    # strikes: a leg hidden behind something can have a junk heel track
    # while still reporting high visibility.
    scored = [_stride_cadence(s) for s in strikes]
    best = int(np.argmax([score for _, score in scored]))
    if scored[best][1] == 0.0:
        raise ValueError("no gait events found (need at least three foot strikes on one foot)")

    shoulder = (series("left_shoulder") + series("right_shoulder")) / 2
    report = {
        "duration_s": n / fps,
        "n_strikes": float(len(all_strikes)),
        "cadence_spm": scored[best][0],
        "ground_contact_time_s": float(np.median(contacts[best])),
        "vertical_oscillation_px": k.vertical_oscillation(hip_s[keep]),
        "overstride_px": float(np.mean(np.concatenate(overstride))),
        "trunk_lean_deg": float(sign * np.mean(k.trunk_lean(shoulder[keep], hip[keep]))),
        "frames_interpolated": float((~good & ~long_gap).sum()),
        "frames_unusable": float(long_gap.sum()),
    }
    if flex_strike:
        report["knee_flexion_strike_deg"] = float(np.mean(np.concatenate(flex_strike)))
        report["knee_flexion_peak_deg"] = float(np.mean(np.concatenate(flex_peak)))
    if m_per_px is not None:
        report["vertical_oscillation_m"] = report["vertical_oscillation_px"] * m_per_px
        report["overstride_m"] = report["overstride_px"] * m_per_px
    return report


def format_report(report: dict[str, float]) -> str:
    return "\n".join(f"{key:28s} {value:10.3f}" for key, value in report.items())
