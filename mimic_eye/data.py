import hashlib
import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from . import LABELS

REGIONS = (
    "cardiac silhouette", "left clavicle", "left costophrenic angle",
    "left hilar structures", "left lower lung zone", "left lung", "left mid lung zone",
    "left upper lung zone", "right clavicle", "right costophrenic angle",
    "right hilar structures", "right lower lung zone", "right lung", "right mid lung zone",
    "right upper lung zone", "trachea", "upper mediastinum",
)
EXCLUDED = (
    "Consolidation", "Pneumothorax", "Lung Lesion", "Enlarged Cardiomediastinum",
    "Fracture", "Pleural Other",
)
SPLITS = ("train", "val", "test")
ID_COLUMNS = ("dicom_id", "DICOM_ID", "study_id", "subject_id", "id")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path, columns=()):
    if not path.is_file():
        return None
    frame = pd.read_csv(path, dtype=dict.fromkeys(ID_COLUMNS, str))
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return frame


def label_vector(row):
    return [int(row.get(label) == 1) for label in LABELS]


def eye_gaze(frame):
    """Convert EyeGaze fixations and per-image pupil baselines to x, y, seconds, area."""
    frame = frame.sort_values("Time (in secs)")
    diameters = frame[["LPD", "RPD"]].to_numpy(dtype=float)
    area = (diameters ** 2).mean(axis=1) * np.pi / 4
    valid = (frame[["LPV", "RPV"]].to_numpy() == 1).all(axis=1)
    valid &= np.isfinite(area) & (diameters > 0).all(axis=1)
    time = frame["Time (in secs)"].to_numpy(dtype=float)
    area[~valid] = np.nan
    time_valid = valid & np.isfinite(time)
    baseline = area[time_valid & (time <= time[time_valid].min() + 2)] if time_valid.any() else []
    pupil = area / np.mean(baseline) if len(baseline) else np.full(len(area), np.nan)
    return clean_gaze(frame[["FPOGX", "FPOGY", "FPOGD"]].to_numpy(dtype=float), pupil)


def reflacx_gaze(folder):
    columns = (
        "x_position", "y_position", "xmin_shown_from_image", "xmax_shown_from_image",
        "ymin_shown_from_image", "ymax_shown_from_image", "timestamp_start_fixation",
        "timestamp_end_fixation", "pupil_area_normalized",
    )
    frame = read_table(folder / "fixations.csv", columns)
    if frame is None:
        return []
    frame = frame.sort_values("timestamp_start_fixation")
    x_span = frame.xmax_shown_from_image - frame.xmin_shown_from_image
    y_span = frame.ymax_shown_from_image - frame.ymin_shown_from_image
    x = (frame.x_position - frame.xmin_shown_from_image) / x_span.where(x_span > 0)
    y = (frame.y_position - frame.ymin_shown_from_image) / y_span.where(y_span > 0)
    duration = frame.timestamp_end_fixation - frame.timestamp_start_fixation
    return clean_gaze(np.column_stack((x, y, duration)), frame.pupil_area_normalized.to_numpy(float))


def clean_gaze(xyd, pupil):
    valid = np.isfinite(xyd).all(axis=1) & (xyd[:, 2] > 0)
    xyd = xyd[valid]
    xyd[:, :2] = xyd[:, :2].clip(0, 1)
    return [
        [*coords.tolist(), float(area) if np.isfinite(area) and area > 0 else None]
        for coords, area in zip(xyd, pupil[valid])
    ]


def read_transcript(eye, dicom_id, reflacx):
    path = eye / "audio_segmentation_transcripts" / dicom_id / "transcript.json"
    if path.is_file():
        value = read_json(path)
        text = value.get("full_text") or value.get("transcript")
        if isinstance(text, str) and text.strip():
            return text.strip()
        phrases = value.get("time_stamped_text", [])
        text = " ".join(phrase.get("phrase", "") for phrase in phrases).strip()
        if text:
            return text
    if reflacx is not None:
        words = read_table(reflacx / "timestamps_transcription.csv", ("word",))
        if words is not None:
            return " ".join(words.word.dropna().astype(str))
    return ""


def read_boxes(frame, size, source):
    if frame is None:
        return []
    width, height = size
    coords = ("x1", "y1", "x2", "y2") if source == "eyegaze" else ("xmin", "ymin", "xmax", "ymax")
    if set(coords) - set(frame):
        raise ValueError(f"Missing {source} box coordinates {coords}")
    boxes = []
    for row in frame.to_dict("records"):
        region = str(row.get("bbox_name", "expert lesion")).strip().lower()
        if source == "eyegaze" and region not in REGIONS:
            continue
        x1, y1, x2, y2 = (float(row[key]) for key in coords)
        bounds = [x1 / width, y1 / height, x2 / width, y2 / height]
        if 0 <= bounds[0] < bounds[2] <= 1 and 0 <= bounds[1] < bounds[3] <= 1:
            boxes.append({"region": region, "bounds": bounds, "source": source, "confidence": 1.0})
    return boxes


def group_by_id(frame, column):
    return dict(tuple(frame.groupby(column))) if frame is not None else {}


def patient_cases(patient, source_counts):
    cxr = patient / "CXR-JPG"
    meta = read_table(cxr / "cxr_meta.csv", ("dicom_id", "study_id"))
    labels = read_table(cxr / "cxr_chexpert.csv", ("study_id", *LABELS))
    if meta is None:
        return
    by_study = {} if labels is None else labels.drop_duplicates("study_id").set_index("study_id").to_dict("index")
    eye = patient / "EyeGaze"
    gaze = group_by_id(read_table(eye / "fixations.csv", (
        "DICOM_ID", "FPOGX", "FPOGY", "FPOGD", "LPD", "RPD", "LPV", "RPV", "Time (in secs)",
    )), "DICOM_ID")
    boxes = group_by_id(read_table(eye / "bounding_boxes.csv", (
        "dicom_id", "bbox_name", "x1", "y1", "x2", "y2",
    )), "dicom_id")
    reflacx = patient / "REFLACX"
    readings = group_by_id(read_table(reflacx / "metadata.csv", ("dicom_id", "id")), "dicom_id")

    for row in meta.drop_duplicates("dicom_id").to_dict("records"):
        dicom_id, study_id = row["dicom_id"], row["study_id"]
        if not isinstance(dicom_id, str) or not isinstance(study_id, str):
            continue
        image = cxr / f"s{study_id}" / f"{dicom_id}.jpg"
        if not image.is_file():
            continue
        in_eye, in_reflacx = dicom_id in gaze, dicom_id in readings
        source_counts["images"] += 1
        source_counts["eyegaze"] += in_eye
        source_counts["reflacx"] += in_reflacx
        source_counts["both"] += in_eye and in_reflacx
        source_counts["image_only"] += not (in_eye or in_reflacx)
        if study_id not in by_study:
            continue
        label_row = by_study[study_id]
        targets = label_vector(label_row)
        if not any(targets) or any(label_row.get(label) == 1 for label in EXCLUDED):
            continue
        reference = patient / "CXR-DICOM" / f"s{study_id}.txt"
        if not reference.is_file() or reference.stat().st_size == 0:
            continue
        try:
            with Image.open(image) as im:
                size = im.size
                im.load()
        except OSError:
            continue

        matches = readings.get(dicom_id)
        folders = [] if matches is None else [
            reflacx / "main_data" / reading for reading in sorted(matches.id.dropna().unique())
        ]
        fixes = eye_gaze(gaze[dicom_id]) if dicom_id in gaze else []
        chosen = None
        if not fixes:
            for folder in folders:
                fixes = reflacx_gaze(folder)
                if fixes:
                    chosen = folder
                    break
        if not fixes:
            continue
        annotations = read_boxes(boxes.get(dicom_id), size, "eyegaze")
        for folder in folders:
            ellipses = read_table(folder / "anomaly_location_ellipses.csv")
            annotations.extend(read_boxes(ellipses, size, "reflacx"))
        transcript = read_transcript(eye, dicom_id, chosen or (folders[0] if folders else None))
        subject_id = row.get("subject_id")
        yield {
            "id": dicom_id,
            "patient": subject_id if isinstance(subject_id, str) and subject_id else patient.name,
            "study": study_id,
            "image": str(image),
            "report": str(reference),
            "labels": targets,
            "boxes": annotations,
            "gaze": fixes,
            "transcript": transcript,
        }


def split_cases(cases, seed):
    groups = [case["patient"] for case in cases]
    patients = len(set(groups))
    if patients < 3:
        raise ValueError("At least three patients are required for train, validation, and test")
    indices = np.arange(len(cases))
    train, rest = next(GroupShuffleSplit(
        test_size=max(2, ceil(patients * 0.3)), random_state=seed,
    ).split(indices, groups=groups))
    val, test = next(GroupShuffleSplit(test_size=0.46, random_state=seed).split(
        rest, groups=np.asarray(groups)[rest],
    ))
    for split, subset in zip(SPLITS, (train, rest[val], rest[test])):
        for index in subset:
            cases[index]["split"] = split
    validate_splits(cases)


def validate_splits(cases):
    if not cases or {case["split"] for case in cases} != set(SPLITS):
        raise ValueError("Manifest must contain train, val, and test cases")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Manifest contains duplicate image IDs")
    for field in ("patient", "study"):
        assignments = {}
        for case in cases:
            if assignments.setdefault(case[field], case["split"]) != case["split"]:
                raise ValueError(f"Manifest contains a {field} shared across splits")


def gaze_stats(cases):
    """Fit duration and pupil statistics on training cases, with finite missing-value defaults."""
    result = []
    for feature in (2, 3):
        values = [
            fix[feature] for case in cases if case["split"] == "train"
            for fix in case["gaze"] if fix[feature] is not None
        ]
        result.append([float(np.mean(values)), float(np.std(values)) or 1.0] if values else [0.0, 1.0])
    return result


def region_bounds(cases):
    groups = defaultdict(list)
    for case in cases:
        if case["split"] != "test":
            for box in case["boxes"]:
                if box["region"] in REGIONS:
                    groups[box["region"]].append(box)
    bounds = {}
    for name, boxes in sorted(groups.items()):
        coords = np.asarray([box["bounds"] for box in boxes])
        median = np.median(coords, axis=0)
        if (median[2:] - median[:2] <= 1e-6).any():
            weights = [box["confidence"] for box in boxes]
            median = np.average(coords, weights=weights, axis=0) if sum(weights) else coords.mean(axis=0)
        bounds[name] = median.tolist()
    return bounds


def prepare(root, output, seed, detector_weights=None, device=None):
    root = Path(root).resolve()
    if (root / "mimic-eye").is_dir():
        root = root / "mimic-eye"
    source_counts = Counter()
    cases = [
        case for patient in sorted(root.glob("patient_*")) if patient.is_dir()
        for case in patient_cases(patient, source_counts)
    ]
    if not cases:
        raise ValueError(f"No usable MIMIC-Eye cases found under {root}")
    split_cases(cases, seed)
    if detector_weights:
        from .detector import complete_boxes
        complete_boxes(cases, detector_weights, device, Path(output).resolve().parent / "box-completion")
    save_manifest(output, cases, seed, source_counts)


def save_manifest(output, cases, seed, source_counts=None):
    summary = {
        split: {
            "images": sum(case["split"] == split for case in cases),
            "positive": dict(zip(LABELS, map(int, np.sum(
                [case["labels"] for case in cases if case["split"] == split], axis=0,
            )))),
            "box_sources": dict(Counter(
                box["source"] for case in cases if case["split"] == split for box in case["boxes"]
            )),
        }
        for split in SPLITS
    }
    write_json(output, {
        "seed": seed, "labels": LABELS, "gaze_stats": gaze_stats(cases),
        "regions": region_bounds(cases), "source_counts": source_counts, "summary": summary, "cases": cases,
    })
    print(json.dumps(summary, indent=2))


def load_manifest(path):
    manifest = read_json(path)
    if tuple(manifest["labels"]) != LABELS:
        raise ValueError("Manifest labels do not match the classifier label order")
    validate_splits(manifest["cases"])
    return manifest


def prepare_chexpert(root, output, seed):
    """Build a local pretraining manifest from CheXpert's original CSV labels."""
    root = Path(root).resolve()
    frame = read_table(root / "train.csv", ("Path", *LABELS))
    if frame is None:
        raise FileNotFoundError(root / "train.csv")
    cases = []
    for row in frame.to_dict("records"):
        path = Path(row["Path"])
        image = root.joinpath(*path.parts[1:]) if path.parts[0] == root.name else root / path
        if not image.is_file():
            raise FileNotFoundError(image)
        cases.append({
            "id": path.as_posix(), "patient": path.parts[-3],
            "study": "/".join(path.parts[-3:-1]), "image": str(image),
            "labels": label_vector(row), "boxes": [], "gaze": [], "transcript": "", "report": None,
        })
    split_cases(cases, seed)
    save_manifest(output, cases, seed)
