# Dataset Documentation

Dataset Name: Hard Hat Workers (Roboflow export)

Dataset Source: Roboflow (exported by user `joseph-nelson`)

Roboflow URL: https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/2 (placeholder)

License: Public Domain (as indicated in the dataset metadata)

Number of Images:

- Train (original): 5209 images
- Test (original): 1766 images (provided and preserved)
- Validation (to be created): ~15% of train (≈ 781–782 images)

Train/Validation/Test Distribution (expected after split):

- Train: ~4427 images
- Validation: ~782 images
- Test: 1766 images

Classes:

- head
- helmet
- person

Annotation Format:

- YOLO TXT per-image annotations (one object per line):
  class_id x_center y_center width height

- Coordinates are normalized to [0, 1] relative to image width/height.

Reason for choosing this dataset:

- High-quality, curated dataset for safety-critical detection (hard hat/helmet detection) with
  a realistic variety of scenes and at least 500 images per class.

Folder structure (root: `data/`):

```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Usage:

- Create deterministic validation split:

```
python src/split_dataset.py --data-dir data --valid-ratio 0.15 --seed 42
```

- Verify dataset integrity and annotations:

```
python src/verify_dataset.py --data-dir data --config data/data.yaml
```

Notes:

- The provided test split must never be modified; `split_dataset.py` only operates on the
  training set to create a validation set.
- The verification utility enforces strict YOLO annotation formatting and will exit with
  a non-zero status if validation problems are found.
