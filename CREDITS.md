# Credits

## Validation video

`Effect-of-Running-Speed-and-Leg-Prostheses-on-Mediolateral-Foot-Placement-and-Its-Variability-pone.0115637.s003.ogv`
(side view of a non-amputee sprinter on a treadmill at 3.0 m/s)

- Authors: Arellano C, McDermott W, Kram R, Grabowski A (2015), "Effect of
  Running Speed and Leg Prostheses on Mediolateral Foot Placement and Its
  Variability", PLOS ONE, doi:10.1371/journal.pone.0115637 (S1 video).
- Source: https://commons.wikimedia.org/wiki/File:Effect-of-Running-Speed-and-Leg-Prostheses-on-Mediolateral-Foot-Placement-and-Its-Variability-pone.0115637.s003.ogv
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Changes: none to the file. Landmarks and metrics derived from it by this
  repo are our own analysis; the video is not redistributed here, it is
  downloaded by `scripts/fetch_video.py` into the git-ignored `data/`.

## Held-out validation video

`Effect-of-Running-Speed-and-Leg-Prostheses-on-Mediolateral-Foot-Placement-and-Its-Variability-pone.0115637.s005.ogv`
(side view of the same non-amputee sprinter at his maximum speed, 9.0 m/s).
Same authors, paper, licence (CC BY 4.0) and no-changes statement as above;
SHA1 e87cfda31eec61a7695414f119c25fab4ddb33d5. The paper's s007 and s009
videos show a sprinter with a transtibial prosthesis and were not used.

## Third (held-out) video

`Jogging - near arakawa river - tokyo japan - 2022 may 3.webm`, by Nesnad
(own work), CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ ;
https://commons.wikimedia.org/wiki/File:Jogging_-_near_arakawa_river_-_tokyo_japan_-_2022_may_3.webm ;
SHA1 4abc01befcc367d62d1839019153ed974f3ebe07. Unchanged; not redistributed.

## MediaPipe model

`pose_landmarker_full.task` from Google's MediaPipe model storage
(https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task),
licensed under Apache License 2.0 per Google's "Model Card BlazePose GHUM 3D"
(https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf,
checked 2026-10-08: "LICENSED UNDER Apache License, Version 2.0"). Downloaded by
the same script, not committed.
