"""Synthetic pose-landmark sequences with known ground-truth gait events.

Lets kinematics.py be tested without any real video or MediaPipe model.
The geometry is deliberately simple -- flat ground-contact segments and a
half-sine swing arc for each heel, a cosine bounce for the hip -- so the
ground truth (strike times, toe-off times, contact time, vertical
oscillation amplitude) is exact, and the recovery error in the tests
measures the estimator, not the data.
"""

import numpy as np

from landmarks import NUM_LANDMARKS, POSE_LANDMARKS


def make_synthetic_gait(cadence_spm, contact_time_s, fps, seconds, noise=0.0,
                         seed=None, hip_y0=400.0, heel_y0=600.0,
                         vertical_osc_px=15.0, swing_height_px=120.0,
                         speed_px_s=150.0, leg_px=180.0):
    """Build a synthetic (frames, 33, 3) landmark array of a runner moving
    in the +x pixel direction, viewed from the side.

    cadence_spm: full running cadence, both feet, steps per minute --
        this is the number a GPS watch would report.
    contact_time_s: ground contact time per strike, seconds, same for
        both feet.
    fps, seconds: frame rate and clip length.
    noise: std dev, pixels, of Gaussian noise added to every landmark
        position. 0 (default) gives an exact signal.
    seed: RNG seed for the noise, for reproducible tests.

    Returns (landmarks, truth):
    landmarks: (frames, 33, 3) array, channels (x_px, y_px, visibility).
    truth: dict of the exact generating values -- "strike_times_left",
        "strike_times_right", "strike_times" (both feet, merged and
        sorted), "toe_off_times_left", "toe_off_times_right",
        "step_time_s" (time between any strike, either foot),
        "stride_time_s" (time between one foot's own strikes), "fps",
        "vertical_osc_px".
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    if cadence_spm <= 0:
        raise ValueError(f"cadence_spm must be positive, got {cadence_spm}")

    step_time_s = 60.0 / cadence_spm     # time between ANY strike, either foot
    stride_time_s = 2.0 * step_time_s    # time between the SAME foot's strikes
    if not (0 < contact_time_s < stride_time_s):
        raise ValueError(
            f"contact_time_s ({contact_time_s}) must be in (0, {stride_time_s}) "
            "given this cadence")

    n_frames = int(round(seconds * fps))
    t = np.arange(n_frames) / fps

    hip_x = 500.0 + speed_px_s * t
    hip_y = hip_y0 - vertical_osc_px * np.cos(2 * np.pi * t / step_time_s)

    def heel_trajectory(phase_offset):
        tau = (t - phase_offset) % stride_time_s
        in_contact = tau < contact_time_s
        swing_frac = np.where(
            in_contact, 0.0,
            (tau - contact_time_s) / (stride_time_s - contact_time_s))
        # Flat at heel_y0 during contact; a smooth lift-and-return arc
        # during swing (0 at both ends of swing, so it is continuous with
        # the flat segments either side).
        y = heel_y0 - swing_height_px * np.sin(np.pi * swing_frac)
        # Ahead of the hip at the start of its cycle (landing, overstride),
        # trailing by mid-swing. Not biomechanically exact -- just enough
        # structure for overstride() to have something non-trivial to measure.
        x = hip_x + leg_px * np.cos(2 * np.pi * tau / stride_time_s)
        return x, y

    def event_times(phase_offset):
        n_strides = int(np.ceil(seconds / stride_time_s)) + 1
        strikes = phase_offset + np.arange(n_strides) * stride_time_s
        strikes = strikes[(strikes >= 0) & (strikes <= seconds - contact_time_s)]
        return strikes, strikes + contact_time_s

    left_x, left_y = heel_trajectory(0.0)
    right_x, right_y = heel_trajectory(step_time_s)
    strikes_left, toe_offs_left = event_times(0.0)
    strikes_right, toe_offs_right = event_times(step_time_s)

    shoulder_y = hip_y - 1.1 * leg_px
    shoulder_x = hip_x  # upright trunk by construction

    landmarks = np.zeros((n_frames, NUM_LANDMARKS, 3), dtype=float)
    landmarks[:, :, 2] = 1.0  # visibility

    animated = {
        "left_hip": (hip_x, hip_y),
        "right_hip": (hip_x, hip_y),
        "left_heel": (left_x, left_y),
        "right_heel": (right_x, right_y),
        "left_ankle": (left_x, left_y - 20.0),
        "right_ankle": (right_x, right_y - 20.0),
        "left_knee": (0.5 * (hip_x + left_x), 0.5 * (hip_y + left_y)),
        "right_knee": (0.5 * (hip_x + right_x), 0.5 * (hip_y + right_y)),
        "left_shoulder": (shoulder_x, shoulder_y),
        "right_shoulder": (shoulder_x, shoulder_y),
    }
    for name, (x, y) in animated.items():
        i = POSE_LANDMARKS[name]
        landmarks[:, i, 0] = x
        landmarks[:, i, 1] = y

    # Everything not simulated above (face, arms, other foot points) is
    # parked at the hip so the array has no undefined positions. No metric
    # in this scaffold reads these.
    unused = set(POSE_LANDMARKS.values()) - {POSE_LANDMARKS[n] for n in animated}
    for i in unused:
        landmarks[:, i, 0] = hip_x
        landmarks[:, i, 1] = hip_y

    if noise:
        rng = np.random.default_rng(seed)
        landmarks[:, :, :2] += rng.normal(0.0, noise, size=landmarks[:, :, :2].shape)

    truth = {
        "strike_times_left": strikes_left,
        "strike_times_right": strikes_right,
        "strike_times": np.sort(np.concatenate([strikes_left, strikes_right])),
        "toe_off_times_left": toe_offs_left,
        "toe_off_times_right": toe_offs_right,
        "step_time_s": step_time_s,
        "stride_time_s": stride_time_s,
        "fps": fps,
        "vertical_osc_px": vertical_osc_px,
    }
    return landmarks, truth
