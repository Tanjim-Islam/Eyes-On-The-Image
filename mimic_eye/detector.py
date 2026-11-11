import os
import shutil
from pathlib import Path

from .data import REGIONS, SPLITS, load_manifest, write_json


def load_detector(weights, workspace):
    settings = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(workspace / "settings")))
    settings.expanduser().mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    return YOLO(str(weights))


def complete_boxes(cases, weights, device, workspace):
    model = load_detector(weights, workspace)
    for case in cases:
        present = {box["region"] for box in case["boxes"]}
        if set(REGIONS) <= present:
            continue
        result = model.predict(
            case["image"], conf=0.05, iou=0.5, device=device, verbose=False,
            project=str(workspace), name="boxes", save=False,
        )[0]
        best = {}
        for coords, score, label in zip(
            result.boxes.xyxyn.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(),
        ):
            region = result.names[int(label)].strip().lower()
            if region not in REGIONS or region in present:
                continue
            if region in best and best[region]["confidence"] >= score:
                continue
            x1, y1, x2, y2 = (max(0.0, min(value, 1.0)) for value in coords)
            if x1 < x2 and y1 < y2:
                best[region] = {
                    "region": region, "bounds": [x1, y1, x2, y2],
                    "source": "yolo", "confidence": score,
                }
        case["boxes"].extend(best.values())


def train_detector(manifest, output, device, epochs=50, batch=16):
    data = load_manifest(manifest)
    cases = [
        (case, [box for box in case["boxes"] if box["source"] == "eyegaze"])
        for case in data["cases"]
    ]
    if {case["split"] for case, boxes in cases if boxes} != set(SPLITS):
        raise ValueError("Each split needs at least one image with expert EyeGaze boxes")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = load_detector(output / "yolov8n.pt", output)
    for index, (case, boxes) in enumerate(cases):
        if not boxes:
            continue
        image = Path(case["image"])
        dst = output / "images" / case["split"] / f"{index}{image.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.hardlink_to(image)
        except OSError:
            shutil.copyfile(image, dst)
        label = output / "labels" / case["split"] / f"{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for box in boxes:
            x1, y1, x2, y2 = box["bounds"]
            values = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
            lines.append(f"{REGIONS.index(box['region'])} " + " ".join(f"{v:.8f}" for v in values))
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = output / "data.yaml"
    write_json(config, {
        "path": output.as_posix(), **{split: f"images/{split}" for split in SPLITS},
        "names": list(REGIONS),
    })
    model.train(
        data=str(config), imgsz=640, epochs=epochs, batch=batch, device=device,
        seed=data["seed"], project=str(output), name="fit",
    )
    weights = output / "fit" / "weights" / "best.pt"
    model = load_detector(weights, output)
    metrics = {}
    for split in SPLITS:
        box = model.val(
            data=str(config), split=split, imgsz=640, device=device,
            project=str(output), name=split, plots=False,
        ).box
        metrics[split] = {
            "precision": float(box.mp), "recall": float(box.mr),
            "f1": float(box.f1.mean()) if len(box.f1) else 0.0,
            "map50": float(box.map50), "map": float(box.map),
            "regions": {
                REGIONS[int(index)]: {"precision": float(p), "recall": float(r), "ap50": float(ap50), "ap": float(ap)}
                for index, p, r, ap50, ap in zip(box.ap_class_index, box.p, box.r, box.ap50, box.ap)
            },
        }
    write_json(output / "metrics.json", metrics)
    print(f"Detector checkpoint at {weights}")
