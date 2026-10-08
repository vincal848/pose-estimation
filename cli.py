"""Command line: video (or saved landmarks) -> metrics report.

    python cli.py run.mp4 --model pose_landmarker_full.task --save-landmarks run.npz
    python cli.py run.npz --m-per-px 0.002
"""

import argparse
from collections.abc import Sequence

import video
from analyze import analyze_landmarks, format_report


def main(argv: Sequence[str] | None = None, detect: video.Detector | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video file, or a .npz saved with --save-landmarks")
    p.add_argument("--model", help="MediaPipe pose_landmarker .task file (video input)")
    p.add_argument("--save-landmarks", help="write the extracted landmarks to this .npz")
    p.add_argument("--m-per-px", type=float, help="metres per pixel, for metre-valued metrics")
    p.add_argument("--facing", choices=["right", "left"], default="right")
    args = p.parse_args(argv)

    if args.input.endswith(".npz"):
        landmarks, fps = video.load_landmarks(args.input)
    else:
        landmarks, fps = video.extract_landmarks(args.input, model_path=args.model, detect=detect)
        if args.save_landmarks:
            video.save_landmarks(args.save_landmarks, landmarks, fps)
    print(format_report(analyze_landmarks(landmarks, fps, m_per_px=args.m_per_px, facing=args.facing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
