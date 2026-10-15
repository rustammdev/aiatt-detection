# aiatt-detection

Person detection and tracking used by the ai-attendance analyzer worker.

It is published under the GNU Affero General Public License v3.0 because it
uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is
AGPL-3.0. The rest of the product talks to this package only through the
values it returns and is not part of it.

| File | What it holds |
|---|---|
| `app/detection/yolo.py` | `PersonDetector`: YOLO restricted to the COCO `person` class, with keypoints when a `*-pose` checkpoint is loaded |
| `app/detection/__init__.py` | `PersonDetections`, `build_tracker` (ByteTrack), `resolve_device` |

Every tag is the version of a worker release built from exactly this source.
The worker image names it in the label `ai.aiatt.detector-source`.

## Use

```python
import cv2
from app.detection.yolo import PersonDetector

detector = PersonDetector(
    model="yolo26s-pose.pt",
    device="auto",
    confidence=0.20,
    iou=0.5,
    image_size=1600,
    half_precision=True,
)
people = detector.detect(cv2.imread("frame.jpg"))
print(len(people.detections), people.has_pose)
```

## Requirements

Python 3.12 and:

| Package | Licence |
|---|---|
| ultralytics ≥ 8.4 | AGPL-3.0 |
| supervision ≥ 0.29 | MIT |
| trackers ≥ 2.5 | Apache-2.0 |
| torch | BSD-style |
| numpy | BSD-3-Clause |

The model weights (`yolo26s-pose.pt`) are Ultralytics' own and are distributed
under AGPL-3.0 by Ultralytics.

## Licence

AGPL-3.0-only. See [LICENSE](LICENSE).
