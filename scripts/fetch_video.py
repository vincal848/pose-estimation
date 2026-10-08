"""Download the validation clip and the MediaPipe model into data/ (git-ignored).

    python scripts/fetch_video.py            # both
    python scripts/fetch_video.py video|model

Refuses any file whose checksum does not match. See CREDITS.md for licences.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# name -> (url, hash algorithm, expected hex digest)
FILES = {
    "video": (
        "https://upload.wikimedia.org/wikipedia/commons/9/94/"
        "Effect-of-Running-Speed-and-Leg-Prostheses-on-Mediolateral-Foot-Placement-and-Its-Variability-pone.0115637.s003.ogv",
        "sha1",
        "7bebb7696c158e29d5e8f1df234ced95130216e7",
    ),
    "model": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "sha256",
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    ),
}


def fetch(name: str) -> Path:
    url, algo, digest = FILES[name]
    dest = DATA / url.rsplit("/", 1)[1]
    if not dest.exists():
        DATA.mkdir(exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "pose-estimation/0.1"})
        with urllib.request.urlopen(req) as r:
            dest.write_bytes(r.read())
    got = hashlib.new(algo, dest.read_bytes()).hexdigest()
    if got != digest:
        dest.unlink()
        raise SystemExit(f"{dest.name}: {algo} {got} != expected {digest}; file removed")
    print(dest)
    return dest


if __name__ == "__main__":
    for n in sys.argv[1:] or FILES:
        fetch(n)
