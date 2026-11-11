import json
from pathlib import Path

LABELS = (
    "Atelectasis", "Cardiomegaly", "Edema", "Lung Opacity", "No Finding",
    "Pleural Effusion", "Pneumonia", "Support Devices",
)
PAPER = json.loads(Path(__file__).with_name("paper.json").read_text(encoding="utf-8"))
