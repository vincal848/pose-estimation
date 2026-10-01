"""Video -> landmark extraction via MediaPipe Pose.

Not implemented yet (M2 on the README's milestone list). The intended API
is documented here so kinematics.py and the tests can be written against
it now. mediapipe and opencv-python are optional dependencies
(requirements-video.txt) and are imported lazily, inside extract_landmarks,
so importing this module -- or anything else in the package -- never
requires them.

Intended API
------------
extract_landmarks(path, model_complexity=1) -> (landmarks, fps)
    landmarks: an (frames, 33, 3) array, channels (x_px, y_px, visibility),
        matching the convention in landmarks.py.
    fps: the video's frame rate, read from its container, as a float.

    model_complexity is passed straight through to mediapipe.solutions.pose
    (0 = fastest/least accurate, 2 = slowest/most accurate).
"""


def extract_landmarks(path, model_complexity=1):
    import cv2  # noqa: F401  (requirements-video.txt; lazy on purpose)
    import mediapipe as mp  # noqa: F401

    raise NotImplementedError(
        "Video landmark extraction lands in M2. For now, use "
        "synth.make_synthetic_gait to get a (frames, 33, 3) array with "
        "known ground truth, and feed that to kinematics.py directly."
    )
