"""Gait events and metrics computed from joint-position arrays.

Everything here takes plain numpy arrays (one or more per-frame (x, y)
series), not the full landmark array -- landmarks.py is where you pull a
named joint's series out of a (frames, 33, 3) array; this module does not
know MediaPipe's landmark indices.

Convention: image y grows downward, as in every image library. "Near the
ground" is therefore the *largest* y a joint reaches, not the smallest, and
"lifted up" means a smaller y. detect_foot_strikes and vertical_oscillation
both rely on this and it is easy to get backwards, so it is called out again
at each use.
"""

import numpy as np
from scipy.signal import savgol_filter


def joint_angle(a, b, c):
    """Angle at vertex b, in degrees, formed by points a-b-c.

    a, b, c: (2,) or (frames, 2) arrays of (x, y) points. All three must
    share a shape. Used for knee flexion (hip-knee-ankle) and similar.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    if np.any(norm == 0):
        raise ValueError("joint_angle is undefined when a or c sits on top of b")

    cosine = np.sum(ba * bc, axis=-1) / norm
    cosine = np.clip(cosine, -1.0, 1.0)  # guard float drift past +/-1 before arccos
    return np.degrees(np.arccos(cosine))


def smooth(series, fps, method="savgol", window_s=0.1, polyorder=2,
           min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    """Smooth a 1-D per-frame series.

    method="savgol": Savitzky-Golay. Window length is window_s seconds,
        rounded to the nearest odd frame count, so it scales with fps
        instead of being a fixed number of frames.
    method="one_euro": the One Euro Filter (Casiez, Pietriga & Roussel,
        2012) -- a causal filter whose cutoff rises with speed, which
        matters here because foot-strike timing comes from exactly the
        fast part of the signal a naive low-pass would blur.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("smooth expects a 1-D series")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    if method == "savgol":
        window = int(round(window_s * fps))
        window = max(window, polyorder + 2)
        if window % 2 == 0:
            window += 1
        max_window = len(series) if len(series) % 2 else len(series) - 1
        window = min(window, max_window)
        if window < polyorder + 2:
            # Too short a clip to smooth meaningfully -- hand it back as is
            # rather than raise, since a one-second test clip is a normal
            # input, not a bug.
            return series.copy()
        return savgol_filter(series, window, polyorder)

    if method == "one_euro":
        return _one_euro_filter(series, fps, min_cutoff, beta, d_cutoff)

    raise ValueError(f"unknown smoothing method: {method!r}")


def _one_euro_filter(series, fps, min_cutoff, beta, d_cutoff):
    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    out = np.empty_like(series)
    out[0] = series[0]
    x_prev = series[0]
    dx_prev = 0.0
    a_d = alpha(d_cutoff)
    for i in range(1, len(series)):
        dx = (series[i] - x_prev) * fps
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = alpha(cutoff)
        x_hat = a * series[i] + (1 - a) * x_prev
        out[i] = x_hat
        x_prev, dx_prev = x_hat, dx_hat
    return out


def _contact_runs(heel_y, fps, ground_frac=0.25, vel_frac=0.5, min_run_frames=2):
    """Frame ranges where the heel is both near its lowest screen position
    (largest y -- see the module docstring) and nearly stationary. Each
    run is one ground-contact phase: its first frame is a foot strike, its
    one-past-the-end frame is the matching toe-off.
    """
    heel_y = np.asarray(heel_y, dtype=float)
    if heel_y.ndim != 1:
        raise ValueError("heel_y must be a 1-D array")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if len(heel_y) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    vy = np.gradient(heel_y) * fps
    lo, hi = np.percentile(heel_y, 5), np.percentile(heel_y, 95)
    rng = hi - lo
    if rng == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    near_ground = heel_y >= hi - ground_frac * rng
    vel_scale = np.std(vy)
    still = np.abs(vy) <= vel_frac * vel_scale if vel_scale > 0 else np.ones_like(heel_y, dtype=bool)
    contact = near_ground & still

    contact_i = contact.astype(int)
    edges = np.diff(contact_i)
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if contact[0]:
        starts = np.concatenate(([0], starts))
    # A run still open at the last frame has no matching "end" transition
    # and is dropped by the n = min(...) truncation below. That is
    # deliberate: a contact phase truncated by the edge of the clip is one
    # we never see finish, so we cannot say it is a genuine completed
    # ground-contact phase rather than a swing that happened to end near
    # the ground as the recording cut off.
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    keep = (ends - starts) >= min_run_frames
    return starts[keep], ends[keep]


def detect_foot_strikes(heel_y, fps, ground_frac=0.25, vel_frac=0.5, min_run_frames=2):
    """Foot-strike times (seconds), from a single foot's heel-y series.

    A strike is the start of a ground-contact phase: the heel is near its
    largest y (closest to the ground -- image y grows downward) and its
    vertical velocity is small relative to the rest of the clip. See
    detect_toe_offs for the matching end-of-contact times.
    """
    starts, _ = _contact_runs(heel_y, fps, ground_frac, vel_frac, min_run_frames)
    return starts / fps


def detect_toe_offs(heel_y, fps, ground_frac=0.25, vel_frac=0.5, min_run_frames=2):
    """Toe-off times (seconds): the end of each ground-contact phase found
    the same way as detect_foot_strikes. Call with the same heel_y, fps,
    and thresholds so strikes and toe-offs pair up index-for-index.
    """
    _, ends = _contact_runs(heel_y, fps, ground_frac, vel_frac, min_run_frames)
    return ends / fps


def cadence_spm(strike_times):
    """Steps per minute, from consecutive strike times of one tracked foot.

    This counts only the strikes given to it. A single visible foot's own
    strike rate is half the full (both-feet) cadence a GPS watch reports --
    multiply by 2 for that number, or merge both feet's strike times first.
    """
    strike_times = np.asarray(strike_times, dtype=float)
    if len(strike_times) < 2:
        raise ValueError("need at least two strikes to compute a cadence")
    duration = strike_times[-1] - strike_times[0]
    if duration <= 0:
        raise ValueError("strike_times must be increasing")
    steps = len(strike_times) - 1
    return 60.0 * steps / duration


def ground_contact_times(strikes, toe_offs):
    """Contact duration per strike (seconds): toe_offs - strikes.

    strikes and toe_offs must already be paired one-to-one, in order --
    exactly what detect_foot_strikes/detect_toe_offs return when called on
    the same heel_y.
    """
    strikes = np.asarray(strikes, dtype=float)
    toe_offs = np.asarray(toe_offs, dtype=float)
    if strikes.shape != toe_offs.shape:
        raise ValueError(
            "strikes and toe_offs must pair up one-to-one "
            f"(got {strikes.shape[0]} strikes and {toe_offs.shape[0]} toe-offs)")
    if np.any(toe_offs <= strikes):
        raise ValueError("every toe-off must come after its matching strike")
    return toe_offs - strikes


def vertical_oscillation(hip_y, scale=1.0):
    """Vertical bounce amplitude: half the peak-to-trough range of hip_y,
    converted from pixels to metres by scale (see landmarks.pixel_scale).
    This matches how running watches report vertical oscillation -- as an
    amplitude, not the full excursion.
    """
    hip_y = np.asarray(hip_y, dtype=float)
    amplitude_px = (np.max(hip_y) - np.min(hip_y)) / 2.0
    return amplitude_px * scale


def overstride(heel_x, hip_x, strike_indices, scale=1.0):
    """Horizontal heel-ahead-of-hip distance at each strike.

    heel_x, hip_x: full per-frame series. strike_indices: frame indices
    (not times -- multiply a strike time by fps and round, if that is what
    you have) at which to measure. Positive means the heel landed ahead of
    the hip in the direction of increasing x, which is what "overstride"
    means if the runner is moving in that direction.
    """
    heel_x = np.asarray(heel_x, dtype=float)
    hip_x = np.asarray(hip_x, dtype=float)
    idx = np.asarray(strike_indices, dtype=int)
    return (heel_x[idx] - hip_x[idx]) * scale


def trunk_lean(shoulder, hip):
    """Forward trunk lean, in degrees, per frame.

    shoulder, hip: (2,) or (frames, 2) arrays of (x, y) pixel points.
    Image y grows downward, so "upright" is the shoulder directly above
    the hip (smaller y, equal x), which this returns as 0 degrees. A
    positive angle means the shoulder is ahead of the hip in the direction
    of increasing x.
    """
    shoulder = np.asarray(shoulder, dtype=float)
    hip = np.asarray(hip, dtype=float)
    dx = shoulder[..., 0] - hip[..., 0]
    dy = shoulder[..., 1] - hip[..., 1]  # negative when shoulder is above the hip
    return np.degrees(np.arctan2(dx, -dy))


def knee_flexion(hip, knee, ankle):
    """Knee flexion in degrees: 0 for a straight leg, growing as it bends.
    The supplement of the hip-knee-ankle interior angle (joint_angle)."""
    return 180.0 - joint_angle(hip, knee, ankle)


def knee_flexion_per_stride(flexion, strike_indices):
    """(flexion at each strike, peak flexion between each strike and the
    next). flexion is a per-frame series; strike_indices are frame indices.
    The last strike has no following strike, so peak has one fewer entry.
    """
    flexion = np.asarray(flexion, dtype=float)
    idx = np.asarray(strike_indices, dtype=int)
    at_strike = flexion[idx]
    peak = np.array([flexion[a:b].max() for a, b in zip(idx[:-1], idx[1:])])
    return at_strike, peak
