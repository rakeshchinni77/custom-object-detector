"""Production-grade dataset verification utility for YOLO datasets.

Validates directory structure, annotation format, label/image correspondence,
and consistency with `data/data.yaml`.

Exit codes:
    0 - Validation passed
    1 - Validation issues found
    2 - Fatal error (missing folders, invalid YAML, etc.)

Usage:
    python src/verify_dataset.py --data-dir data --config data/data.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - exercised when dependency is missing
    yaml = None


logger = logging.getLogger("verify_dataset")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML content from `path`.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If YAML content cannot be parsed.
    """
    if yaml is None:
        raise ImportError("PyYAML is not installed.\nInstall it using:\npython -m pip install pyyaml")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("YAML content must decode to a dictionary")
    return data


def gather_counts(images_dir: Path, labels_dir: Path) -> Tuple[int, int, List[Path], List[Path]]:
    """Return image and label counts and file lists for a split."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in exts]) if images_dir.exists() else []
    label_paths = sorted([path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]) if labels_dir.exists() else []
    return len(image_paths), len(label_paths), image_paths, label_paths


def validate_annotation_file(path: Path, num_classes: int) -> Tuple[bool, List[str]]:
    """Validate the content of a single YOLO annotation file."""
    errors: List[str] = []
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return False, [f"Could not read file: {exc}"]

    if not text:
        return True, []

    for index, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"Line {index}: expected 5 values, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"Line {index}: class id is not an integer: {parts[0]}")
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(f"Line {index}: class id {class_id} out of range [0, {num_classes - 1}]")

        try:
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"Line {index}: bounding box values must be numeric")
            continue

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            errors.append(f"Line {index}: center coordinates must be normalized in [0, 1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            errors.append(f"Line {index}: width/height must be in (0, 1]")

    return len(errors) == 0, errors


def analyze_split(images: List[Path], labels: List[Path]) -> Tuple[int, int, List[str], List[str]]:
    """Compute missing-label and orphan-label counts for a split."""
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = sorted([path.name for path in images if path.stem not in label_stems])
    orphan_labels = sorted([path.name for path in labels if path.stem not in image_stems])
    return len(images), len(labels), missing_labels, orphan_labels


def produce_report(data_dir: Path, yaml_cfg: Dict[str, Any]) -> int:
    """Run validation checks and print a structured report."""
    required_paths = [
        (data_dir / "train" / "images", data_dir / "train" / "labels"),
        (data_dir / "valid" / "images", data_dir / "valid" / "labels"),
        (data_dir / "test" / "images", data_dir / "test" / "labels"),
    ]

    num_classes: int | None = None
    class_names: List[str] | None = None
    if isinstance(yaml_cfg, dict):
        if "names" in yaml_cfg and isinstance(yaml_cfg["names"], list):
            class_names = [str(name) for name in yaml_cfg["names"]]
            num_classes = len(class_names)
        elif "nc" in yaml_cfg and isinstance(yaml_cfg["nc"], int):
            num_classes = yaml_cfg["nc"]

    if num_classes is None:
        logger.error("Unable to determine class count from data YAML")
        return 2

    summary = {
        "train_images": 0,
        "train_labels": 0,
        "valid_images": 0,
        "valid_labels": 0,
        "test_images": 0,
        "test_labels": 0,
        "missing_labels": 0,
        "orphan_labels": 0,
        "invalid_labels": 0,
    }

    invalid_label_files: List[str] = []

    for (img_dir, lbl_dir), split_name in zip(required_paths, ["train", "valid", "test"]):
        if not img_dir.exists() or not lbl_dir.exists():
            logger.error("Required folders missing for %s split: %s or %s", split_name, img_dir, lbl_dir)
            return 2

        n_images, n_labels, images, labels = gather_counts(img_dir, lbl_dir)
        if split_name == "train":
            summary["train_images"] = n_images
            summary["train_labels"] = n_labels
        elif split_name == "valid":
            summary["valid_images"] = n_images
            summary["valid_labels"] = n_labels
        else:
            summary["test_images"] = n_images
            summary["test_labels"] = n_labels

        _, _, missing_labels, orphan_labels = analyze_split(images, labels)
        summary["missing_labels"] += len(missing_labels)
        summary["orphan_labels"] += len(orphan_labels)

        for label_path in labels:
            ok, errors = validate_annotation_file(label_path, num_classes)
            if not ok:
                summary["invalid_labels"] += 1
                invalid_label_files.append(f"{label_path}: {'; '.join(errors)}")

    report_lines = [
        "==========================",
        "DATASET REPORT",
        "==========================",
        "",
        f"Train Images: {summary['train_images']}",
        f"Train Labels: {summary['train_labels']}",
        "",
        f"Validation Images: {summary['valid_images']}",
        f"Validation Labels: {summary['valid_labels']}",
        "",
        f"Test Images: {summary['test_images']}",
        f"Test Labels: {summary['test_labels']}",
        "",
        f"Classes: {class_names if class_names else num_classes}",
        f"Missing Labels: {summary['missing_labels']}",
        f"Orphan Labels: {summary['orphan_labels']}",
        f"Invalid Labels: {summary['invalid_labels']}",
        "",
    ]

    status = "PASSED"
    if summary["missing_labels"] > 0 or summary["orphan_labels"] > 0 or summary["invalid_labels"] > 0:
        status = "FAILED"
    report_lines.append(f"Status: {status}")
    report_lines.append("==========================")

    print("\n".join(report_lines))

    if invalid_label_files:
        logger.error("Examples of invalid labels:")
        for line in invalid_label_files[:10]:
            logger.error(line)

    return 0 if status == "PASSED" else 1


def main(argv: List[str] | None = None) -> int:
    """Parse CLI arguments and run dataset verification."""
    parser = argparse.ArgumentParser(description="Verify a YOLO dataset structure and annotations")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--config", type=str, default="data/data.yaml", help="Path to data.yaml")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Config YAML not found: %s", config_path)
        return 2

    try:
        cfg = load_yaml(config_path)
    except ImportError as exc:
        print(str(exc))
        return 2
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load YAML: %s", exc)
        return 2

    return produce_report(data_dir, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
