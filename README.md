# IOU Track

[![PyPI](https://img.shields.io/pypi/v/ioutrack_rs.svg?style=flat-square)](https://pypi.org/project/ioutrack_rs/)

Python package for IOU-based tracking ([SORT](https://arxiv.org/abs/1602.00763) & [ByteTrack](https://arxiv.org/abs/2110.06864)) written in Rust by [PaulKlinger](https://github.com/PaulKlinger). This is a modified version that can also return the indices of the boxes kept, so that it can be used with Ultralytics.

## Usage in Ultralytics

1. Install `pip install ioutracker_rs`
2. Patch Ultralytics tracker to use:

```python
from ultralytics import YOLO, ASSETS
import numpy as np
from ioutrack import ByteTrack
import ultralytics.trackers.byte_tracker as byte_tracker

# Monkey-patch with Rust based tracker
tracker = ByteTrack(max_age=5, min_hits=2, init_tracker_min_score=0.25)
def update(self, dets, img):
    boxes, cls = dets.data[:,:5], dets.data[:, -1:]
    tracks = tracker.update(boxes, return_indices=True)
    idxs = tracks[:, -1:].astype(int)
    confs = boxes[idxs.flatten(), 4:5]
    tracks = np.hstack((tracks[:, :-1], confs, cls[idxs.flatten()], idxs))
    return tracks
byte_tracker.BYTETracker.update = update

# Use normally after this. No need to run previous steps again
model = YOLO("yolo11n.pt")
results = model.track(ASSETS / "bus.jpg", tracker="bytetrack.yaml", persist=True, verbose=False)
```

## Latency comparison

```python
import cv2
import time
from ultralytics import YOLO, ASSETS
from ioutrack import ByteTrack
import ultralytics.trackers.byte_tracker as byte_tracker
import numpy as np

def update(self, dets, img):
    global tracker
    boxes, cls = dets.data[:,:5], dets.data[:, -1:]
    tracks = tracker.update(boxes, return_indices=True)
    idxs = tracks[:, -1:].astype(int)
    confs = boxes[idxs.flatten(), 4:5]
    tracks = np.hstack((tracks[:, :-1], confs, cls[idxs.flatten()], idxs))
    return tracks

img = cv2.imread(ASSETS / "bus.jpg")
original_update = byte_tracker.BYTETracker.update 

def test_latency(patch=False):
    model = YOLO("yolo11n.pt")
    if patch:
        global tracker
        tracker = ByteTrack(max_age=5, min_hits=2, init_tracker_min_score=0.25)
        byte_tracker.BYTETracker.update = update
    else:
        byte_tracker.BYTETracker.update  = original_update
    model.track(img, tracker="bytetrack.yaml", persist=True, verbose=False) # warmup
    s = time.perf_counter()
    results = model.track(img, tracker="bytetrack.yaml", persist=True, verbose=False)
    e = time.perf_counter()
    infer = sum(results[0].speed.values())
    print(results[0].speed)
    print("Infer:", infer)
    print("Tracker:", ((e - s) * 1000) - infer)
    print("End2End:", (e - s) * 1000)

print("Ultralytics ByteTrack")
test_latency(False)
print("Rust ByteTrack")
test_latency(True)
```

| Tracker Type       | Tracker (ms) |
|--------------------|--------------|
| Default ByteTrack  | 2.2130       |
| Rust ByteTrack     | 0.7089       |
