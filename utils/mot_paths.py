"""Resolve MOT sequence folders (standard MOTChallenge and custom layouts)."""
import glob
import os


# Folder names on Google Drive may differ slightly from benchmark names.
SEQUENCE_ALIASES = {
    "KITTI-17": ["KITTI-17", "Kitti-17", "kitti-17"],
    "TUD-Campus": ["TUD-Campus", "TUD-campus", "tud-campus"],
    "TUD-Stadtmitte": ["TUD-Stadtmitte", "TUD-stadtmitte"],
    "PETS09-S2L1": ["PETS09-S2L1", "PETS09-s2l1"],
    "MOT16-09": ["MOT16-09", "mot16-09"],
    "MOT16-11": ["MOT16-11", "mot16-11"],
}


def get_image_dir(sequence_dir):
    """Return image folder path (supports img1, img, images)."""
    for name in ("img1", "img", "images"):
        path = os.path.join(sequence_dir, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "No image folder found in %s (expected img1/, img/ or images/)" % sequence_dir)


def _first_txt_in_dir(directory):
    if not os.path.isdir(directory):
        return None
    txts = sorted(glob.glob(os.path.join(directory, "*.txt")))
    return txts[0] if txts else None


def get_gt_file(sequence_dir):
    """Return ground-truth file path."""
    default = os.path.join(sequence_dir, "gt", "gt.txt")
    if os.path.exists(default):
        return default
    found = _first_txt_in_dir(os.path.join(sequence_dir, "gt"))
    return found if found else default


def get_det_file(sequence_dir):
    """Return detection file path."""
    default = os.path.join(sequence_dir, "det", "det.txt")
    if os.path.exists(default):
        return default
    found = _first_txt_in_dir(os.path.join(sequence_dir, "det"))
    return found if found else default


def list_image_filenames(sequence_dir):
    """Map frame index -> absolute image path."""
    image_dir = get_image_dir(sequence_dir)
    image_filenames = {}
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        stem, _ = os.path.splitext(fname)
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        image_filenames[frame_idx] = os.path.join(image_dir, fname)
    return image_filenames


def is_sequence_dir(path):
    """Check if directory looks like a MOT sequence."""
    if not os.path.isdir(path):
        return False
    try:
        get_image_dir(path)
        return True
    except FileNotFoundError:
        return False


def find_sequence_dir(data_root, sequence_name):
    """
    Find a sequence folder under data_root.

    Supports:
    - MyDrive/Videos-CV/MOT16-09/   (flat custom layout)
    - MOT16/train/MOT16-09/         (standard MOTChallenge layout)
    """
    names_to_try = SEQUENCE_ALIASES.get(sequence_name, [sequence_name])
    if sequence_name not in names_to_try:
        names_to_try = [sequence_name] + names_to_try

    search_roots = [data_root]
    for sub in ("MOT15", "MOT16"):
        candidate = os.path.join(data_root, sub)
        if os.path.isdir(candidate):
            search_roots.append(candidate)

    for root in search_roots:
        for name in names_to_try:
            for split in ("train", "test", ""):
                if split:
                    candidate = os.path.join(root, split, name)
                else:
                    candidate = os.path.join(root, name)
                if is_sequence_dir(candidate):
                    return candidate

    tried = ", ".join(names_to_try)
    raise FileNotFoundError(
        "Sequence '%s' not found under %s (tried folder names: %s)" % (
            sequence_name, data_root, tried))
