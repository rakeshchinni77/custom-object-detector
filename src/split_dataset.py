"""Deterministic dataset splitting utility for YOLO-style datasets.

This module moves a deterministic validation subset from the training split into
`data/valid` without touching the test split. The split is idempotent: if the
validation set already exists, the script logs and exits successfully.

Usage:
    python src/split_dataset.py --data-dir data --seed 42 --valid-ratio 0.15
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple


logger = logging.getLogger("split_dataset")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def list_image_files(image_dir: Path) -> List[Path]:
    """Return sorted image files in `image_dir`.

    Args:
        image_dir: Directory containing images.

    Returns:
        A sorted list of image file paths.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in exts])


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not exist.

    Args:
        *paths: One or more directories to create.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_training_pairs(train_img_dir: Path, train_lbl_dir: Path) -> List[Tuple[Path, Path]]:
    """Build image/label pairs from the training split.

    Args:
        train_img_dir: Directory containing training images.
        train_lbl_dir: Directory containing training labels.

    Returns:
        A list of `(image_path, label_path)` pairs that have both files.
    """
    pairs: List[Tuple[Path, Path]] = []
    for image_path in list_image_files(train_img_dir):
        label_path = train_lbl_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            logger.warning("Label not found for image %s; skipping from split selection", image_path.name)
            continue
        pairs.append((image_path, label_path))
    return pairs


def deterministic_split(pairs: Sequence[Tuple[Path, Path]], valid_ratio: float, seed: int) -> List[Tuple[Path, Path]]:
    """Return a deterministic validation subset from `pairs`.

    Args:
        pairs: Training image/label pairs.
        valid_ratio: Fraction of the training set to allocate to validation.
        seed: Deterministic random seed.

    Returns:
        A list of selected pairs.
    """
    if not pairs:
        return []

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_valid = int(round(len(shuffled) * valid_ratio))
    if n_valid <= 0:
        n_valid = 1
    return shuffled[:n_valid]


def move_pairs(selected_pairs: Sequence[Tuple[Path, Path]], valid_img_dir: Path, valid_lbl_dir: Path) -> Tuple[int, int]:
    """Move selected image/label pairs into the validation directories.

    Args:
        selected_pairs: Image/label pairs to move.
        valid_img_dir: Validation image destination.
        valid_lbl_dir: Validation label destination.

    Returns:
        Number of moved images and labels.
    """
    moved_images = 0
    moved_labels = 0

    for image_path, label_path in selected_pairs:
        dst_img = valid_img_dir / image_path.name
        dst_lbl = valid_lbl_dir / label_path.name

        if dst_img.exists() or dst_lbl.exists():
            logger.warning("Validation destination already exists for %s; skipping", image_path.name)
            continue

        shutil.move(str(image_path), str(dst_img))
        shutil.move(str(label_path), str(dst_lbl))
        moved_images += 1
        moved_labels += 1

    return moved_images, moved_labels


def count_files(directory: Path) -> int:
    """Count files in a directory, returning 0 if the directory does not exist."""
    if not directory.exists():
        return 0
    return len([path for path in directory.iterdir() if path.is_file()])


def main(argv: List[str] | None = None) -> int:
    """Command-line entry point for dataset splitting.

    Returns:
        0 on success, 2 on fatal configuration issues.
    """
    parser = argparse.ArgumentParser(description="Create a deterministic validation split for a YOLO dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Fraction of the training set to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "train" / "images"
    train_lbl_dir = data_dir / "train" / "labels"
    valid_img_dir = data_dir / "valid" / "images"
    valid_lbl_dir = data_dir / "valid" / "labels"

    logger.info("Loading training dataset...")
    if not train_img_dir.exists() or not train_lbl_dir.exists():
        logger.error("Training folders not found. Expected %s and %s", train_img_dir, train_lbl_dir)
        return 2

    ensure_dirs(valid_img_dir, valid_lbl_dir)

    if count_files(valid_img_dir) > 0 or count_files(valid_lbl_dir) > 0:
        logger.info("Validation dataset already exists. Dataset split previously completed. Skipping split.")
        logger.info("Final counts - Train: %d, Valid: %d, Test: %d", count_files(train_img_dir), count_files(valid_img_dir), count_files(data_dir / "test" / "images"))
        return 0

    train_pairs = load_training_pairs(train_img_dir, train_lbl_dir)
    logger.info("Found %d training images.", len(train_pairs))
    logger.info("Validation ratio: %.2f%%", args.valid_ratio * 100)

    selected_pairs = deterministic_split(train_pairs, args.valid_ratio, args.seed)
    logger.info("Selected %d validation images.", len(selected_pairs))
    logger.info("Moving validation images...")
    moved_images, moved_labels = move_pairs(selected_pairs, valid_img_dir, valid_lbl_dir)
    logger.info("Moving validation labels...")
    logger.info("Dataset split completed successfully.")
    logger.info("Final counts")
    logger.info("Train : %d", count_files(train_img_dir))
    logger.info("Valid : %d", count_files(valid_img_dir))
    logger.info("Test : %d", count_files(data_dir / "test" / "images"))
    logger.info("Moved images: %d, moved labels: %d", moved_images, moved_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
