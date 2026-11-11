import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from . import LABELS, PAPER
from .data import load_manifest, manifest_digest, write_json
from .gaze import alignment_loss, gaze_metrics
from .model import Classifier, classification_loss, contrastive_loss, load_weights


class Cases(Dataset):
    def __init__(self, rows, stats, spec, modalities):
        self.rows = rows
        self.stats = torch.tensor(stats)
        self.spec = spec
        self.modalities = modalities

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        size = self.spec["image_size"]
        with Image.open(row["image"]) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
        if self.spec["image_backbone"] == "biovil-t":
            pixels = pixels.mean(dim=0, keepdim=True).expand(3, -1, -1)
        elif self.spec["image_backbone"] == "cxr-clip":
            pixels = (pixels - pixels.new_tensor([0.485, 0.456, 0.406])[:, None, None]) / pixels.new_tensor(
                [0.229, 0.224, 0.225],
            )[:, None, None]
        else:
            pixels = pixels * 2 - 1
        sample = {"image": pixels, "labels": torch.tensor(row["labels"], dtype=torch.float32)}
        if not self.modalities:
            return sample

        if self.spec["box"]:
            mask = Image.new("L", (size, size))
            draw = ImageDraw.Draw(mask)
            for box in row["boxes"]:
                bounds = [v * (size - 1) for v in box["bounds"]]
                if box["source"] == "reflacx":
                    draw.ellipse(bounds, fill=255)
                else:
                    draw.rectangle(bounds, fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
            sample["box"] = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0) / 255
        if self.spec["gaze"]:
            length = PAPER["max_fixations"]
            seq = torch.zeros(length, 4)
            raw = torch.zeros_like(seq)
            mask = torch.zeros(length, dtype=torch.bool)
            fixes = row["gaze"][:length]
            if fixes:
                values = [f[:3] + [self.stats[1, 0].item() if f[3] is None else f[3]] for f in fixes]
                raw[:len(fixes)] = torch.tensor(values)
                seq.copy_(raw)
                seq[:len(fixes), 2:] = (raw[:len(fixes), 2:] - self.stats[:, 0]) / self.stats[:, 1]
                mask[:len(fixes)] = True
            sample.update(gaze=seq, gaze_mask=mask, raw_gaze=raw)
            sample["gaze_quality"] = torch.tensor(sum(fix[3] is not None for fix in fixes) / max(1, len(fixes)))
        if self.spec["text"]:
            sample["transcript"] = row["transcript"]
        return sample


def to_model_inputs(batch, model, tokenizer, device):
    inputs = {key: batch[key].to(device) for key in ("image", "box", "gaze", "gaze_mask") if key in batch}
    if "transcript" in batch:
        tokens = tokenizer(
            batch["transcript"], padding=True, truncation=True,
            max_length=model.spec["text_tokens"], return_tensors="pt",
        )
        inputs["text"] = {key: val.to(device) for key, val in tokens.items()}
        inputs["text_present"] = torch.tensor([bool(text.strip()) for text in batch["transcript"]], device=device)
    return inputs


def choose_device(value):
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def probabilities(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            scores.append(model(batch["image"].to(device))[0].float().sigmoid().cpu())
            targets.append(batch["labels"])
    return torch.cat(scores).numpy(), torch.cat(targets).numpy()


def fit_thresholds(scores, labels):
    thresholds = []
    grid = [0.5, *np.arange(0.38, 0.78, 0.02)]
    for index in range(labels.shape[1]):
        target = labels[:, index]
        if len(np.unique(target)) < 2:
            thresholds.append(0.5)
            continue
        threshold = max(grid, key=lambda t: f1_score(target, scores[:, index] > t, zero_division=0))
        thresholds.append(float(threshold))
    return thresholds


def classification_metrics(scores, labels, thresholds):
    predicted = scores > np.asarray(thresholds)
    per_label = {}
    for i, label in enumerate(LABELS):
        target = labels[:, i]
        auc = float(roc_auc_score(target, scores[:, i])) if len(np.unique(target)) == 2 else None
        per_label[label] = {
            "support": int(target.sum()), "auc": auc,
            "f1": float(f1_score(target, predicted[:, i], zero_division=0)),
        }
    aucs = [row["auc"] for row in per_label.values() if row["auc"] is not None]
    return {
        "auc": float(np.mean(aucs)) if aucs else None,
        "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, predicted, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(labels, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predicted, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(labels, predicted, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(labels, predicted, average="micro", zero_division=0)),
        "per_label": per_label,
    }


def training_loss(model, batch, tokenizer, device, pos_weight):
    inputs = to_model_inputs(batch, model, tokenizer, device)
    logits, saliency, features, present = model(**inputs)
    loss = classification_loss(logits, batch["labels"].to(device), pos_weight)
    if model.spec["enhanced"]:
        nce = sum(contrastive_loss(features[0], features[i], present[i]) for i in (2, 3)) / 2
        gaze = alignment_loss(
            saliency, batch["raw_gaze"].to(device), inputs["gaze_mask"],
            inputs["box"], batch["gaze_quality"].to(device),
        )
        loss = loss + PAPER["contrastive_weight"] * nce + PAPER["gaze_weight"] * gaze
    return loss


def fit(model, train_loader, val_loader, tokenizer, device, schedule, output, thresholds=None):
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=schedule["lr"], betas=(0.9, 0.98), weight_decay=PAPER["weight_decay"],
    )
    epochs = schedule["epochs"]
    steps = len(train_loader) * epochs
    warmup = int(steps * schedule["warmup"])

    def learning_rate(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=4096, enabled=amp)
    labels = torch.tensor([case["labels"] for case in train_loader.dataset.rows], device=device).float()
    positives = labels.sum(dim=0)
    weights = ((len(labels) - positives) / positives.clamp_min(1)).clamp(1, 20)
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        if model.spec.get("freeze_image_box"):
            model.image_encoder.eval()
            model.box_encoder.eval()
        mean_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = training_loss(model, batch, tokenizer, device, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale:
                scheduler.step()
            mean_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} train loss {mean_loss / len(train_loader):.4f}", flush=True)
        if val_loader is not None:
            scores, targets = probabilities(model, val_loader, device)
            val_loss = F.binary_cross_entropy(torch.from_numpy(scores), torch.from_numpy(targets)).item()
            print(f"Validation loss {val_loss:.4f}", flush=True)
            if val_loss >= best_loss:
                continue
            best_loss, best_epoch = val_loss, epoch
            thresholds = fit_thresholds(scores, targets)
        elif epoch != epochs:
            continue
        temp = output.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "spec": model.spec, "thresholds": thresholds, "epoch": epoch}, temp)
        temp.replace(output)
    return best_epoch


def train(args):
    data = load_manifest(args.manifest)
    schedule = dict(PAPER["experiments"][args.experiment])
    schedule["epochs"] = args.epochs or schedule["epochs"]
    batch = args.batch or schedule["batch"]
    if schedule["enhanced"] and args.image_backbone != "vit-chexpert":
        raise ValueError("The enhanced gaze variant uses ViT patch attention")
    if (args.image_backbone == "cxr-clip" or (
        args.image_backbone == "vit-chexpert" and args.experiment != "vit-only"
    )) and not args.encoder_weights:
        raise ValueError("Supply --encoder-weights for the selected pretrained backbone")
    if args.encoder_weights and args.image_backbone == "biovil-t":
        raise ValueError("BioViL-T loads its official pretrained encoder")
    if not schedule["box"] and args.box_backbone != "cnn":
        raise ValueError("The selected experiment does not use a box encoder")
    if schedule.get("freeze_image_box") and (args.image_backbone != "vit-chexpert" or args.box_backbone != "cnn"):
        raise ValueError("The text encoder comparison uses ViT-B and the CNN box encoder")

    spec = {
        **{key: schedule[key] for key in ("box", "text", "gaze", "enhanced")},
        "experiment": args.experiment, "image_backbone": args.image_backbone, "box_backbone": args.box_backbone,
        "freeze_image_box": schedule.get("freeze_image_box", False),
        "vision_model": args.vision_model or PAPER["vision_model"],
        "text_model": args.text_model or PAPER["text_models"][args.text_backbone],
        "width": PAPER["width"], "gaze_hidden": 384,
        "image_size": PAPER["image_size"], "text_tokens": PAPER["max_text_tokens"],
    }
    device = choose_device(args.device)
    stats = data["gaze_stats"]
    rows = data["cases"]
    train_rows = [case for case in rows if case["split"] == "train"]
    val_rows = [case for case in rows if case["split"] == "val"]
    torch.manual_seed(data["seed"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = None
    if spec["text"]:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(spec["text_model"])
    model = Classifier(spec, initialize=True, encoder_weights=args.encoder_weights).to(device)
    if tokenizer:
        model.spec["text_tokens"] = min(PAPER["max_text_tokens"], tokenizer.model_max_length)
    train_loader = DataLoader(Cases(train_rows, stats, spec, True), batch_size=batch, shuffle=True)
    val_loader = DataLoader(Cases(val_rows, stats, spec, False), batch_size=batch)
    checkpoint = output / "model.pt"
    epoch = fit(model, train_loader, val_loader, tokenizer, device, schedule, checkpoint)
    if args.refit:
        thresholds = load_weights(checkpoint)["thresholds"]
        del model
        torch.manual_seed(data["seed"])
        model = Classifier(spec, initialize=True, encoder_weights=args.encoder_weights).to(device)
        if tokenizer:
            model.spec["text_tokens"] = min(PAPER["max_text_tokens"], tokenizer.model_max_length)
        combined = DataLoader(Cases(train_rows + val_rows, stats, spec, True), batch_size=batch, shuffle=True)
        fit(model, combined, None, tokenizer, device, {**schedule, "epochs": epoch}, checkpoint, thresholds)
    write_json(output / "run.json", {
        "experiment": args.experiment, "seed": data["seed"], "schedule": schedule,
        "manifest_sha256": manifest_digest(args.manifest),
        "refit_train_val": args.refit, "selected_epoch": epoch,
        "threshold_split": "val", "checkpoint": checkpoint.name,
        "image_backbone": spec["image_backbone"], "box_backbone": spec["box_backbone"] if spec["box"] else None,
        "vision_model": spec["vision_model"], "text_model": spec["text_model"] if spec["text"] else None,
        "batch": batch, "torch_version": str(torch.__version__),
    })
    print(f"Classifier checkpoint at {checkpoint}")


def predict(args):
    checkpoint = load_weights(args.checkpoint)
    spec = checkpoint["spec"]
    device = choose_device(args.device)
    model = Classifier(spec).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.manifest:
        data = load_manifest(args.manifest)
        rows = [case for case in data["cases"] if case["split"] == args.split]
        stats = data["gaze_stats"]
    else:
        rows = [{"id": Path(args.image).stem, "image": str(args.image), "labels": [0] * len(LABELS), "gaze": []}]
        stats = [[0, 1], [0, 1]]
    loader = DataLoader(Cases(rows, stats, spec, False), batch_size=args.batch)
    predictions, all_scores, alignment = [], [], []
    index = 0
    for batch in loader:
        images = batch["image"].to(device)
        gaze_available = args.score_gaze and any(row["gaze"] for row in rows[index:index + len(images)])
        gradient_map = gaze_available and not spec["enhanced"]
        with torch.set_grad_enabled(gradient_map):
            images.requires_grad_(gradient_map)
            logits, saliency, _, _ = model(images)
            if gradient_map:
                gradients, = torch.autograd.grad(logits.amax(dim=1).sum(), images)
                saliency = gradients.abs().mean(dim=1)
            elif saliency is not None:
                saliency = saliency.sigmoid()
        scores = logits.detach().float().sigmoid().cpu().numpy()
        for i, score in enumerate(scores):
            row = rows[index + i]
            result = {"id": row["id"], "probabilities": dict(zip(LABELS, map(float, score)))}
            if gaze_available and row["gaze"]:
                result["gaze"] = gaze_metrics(saliency[i].detach().cpu().numpy(), row["gaze"])
                alignment.append(result["gaze"])
            predictions.append(result)
        index += len(images)
        all_scores.extend(scores)
    result = {"input_mode": "image", "thresholds": checkpoint["thresholds"], "predictions": predictions}
    if args.manifest:
        result["split"] = args.split
        result["manifest_sha256"] = manifest_digest(args.manifest)
        result["metrics"] = classification_metrics(
            np.asarray(all_scores), np.asarray([row["labels"] for row in rows]), checkpoint["thresholds"],
        )
    if alignment:
        result["gaze_metrics"] = {
            name: {"mean": float(np.mean([item[name] for item in alignment])),
                   "std": float(np.std([item[name] for item in alignment]))}
            for name in alignment[0]
        }
        result["gaze_metrics"]["samples"] = len(alignment)
    write_json(args.output, result)
