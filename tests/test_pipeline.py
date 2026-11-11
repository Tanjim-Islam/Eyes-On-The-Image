import csv
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast, ViTConfig, ViTModel

from mimic_eye import LABELS
from mimic_eye.__main__ import main
from mimic_eye.data import load_manifest, read_json, validate_splits
from mimic_eye.evaluate import chexpert_labels
from mimic_eye.gaze import gaze_metrics
from mimic_eye.model import load_weights


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_dataset(root):
    for patient_index in range(6):
        patient = root / f"patient_{patient_index}"
        study = str(100 + patient_index)
        ids = [f"sample-{patient_index}"] if patient_index else ["sample-0", "sample-0-b"]
        cxr = patient / "CXR-JPG"
        csv_file(cxr / "cxr_meta.csv", [
            {"dicom_id": key, "study_id": study, "subject_id": patient_index} for key in ids
        ])
        csv_file(cxr / "cxr_chexpert.csv", [
            {"study_id": study, **{label: int(i == patient_index) for i, label in enumerate(LABELS)}}
        ])
        for key in ids:
            path = cxr / f"s{study}" / f"{key}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.tile(np.arange(48, dtype=np.uint8) + patient_index, (48, 1))
            Image.fromarray(pixels).save(path)
        report = patient / "CXR-DICOM" / f"s{study}.txt"
        report.parent.mkdir()
        report.write_text(f"FINDINGS:\nSample token-{patient_index}. No clear evidence of pneumonia.\n"
                          "IMPRESSION:\nMild atelectasis.\n", encoding="utf-8")

        eye = patient / "EyeGaze"
        csv_file(eye / "bounding_boxes.csv", [
            {"dicom_id": key, "bbox_name": "left lower lung zone", "x1": 4, "y1": 16, "x2": 20, "y2": 38}
            for key in ids
        ])
        if patient_index == 5:
            reflacx = patient / "REFLACX"
            csv_file(reflacx / "metadata.csv", [{"dicom_id": ids[0], "id": "reading-1"}])
            reading = reflacx / "main_data" / "reading-1"
            csv_file(reading / "fixations.csv", [
                {"x_position": 6, "y_position": 20, "xmin_shown_from_image": 2, "xmax_shown_from_image": 42,
                 "ymin_shown_from_image": 4, "ymax_shown_from_image": 44, "timestamp_start_fixation": 0.0,
                 "timestamp_end_fixation": 0.2, "pupil_area_normalized": 1.0},
            ])
            csv_file(reading / "anomaly_location_ellipses.csv", [
                {"xmin": 3, "xmax": 10, "ymin": 8, "ymax": 16, "certainty": 1},
            ])
            csv_file(reading / "timestamps_transcription.csv", [{"word": "atelectasis"}])
            continue
        csv_file(eye / "fixations.csv", [
            {"DICOM_ID": key, "FPOGX": 0.2 + i * 0.1, "FPOGY": 0.65, "FPOGD": 0.2,
             "LPD": 3.0 + i * 0.1, "RPD": 3.0, "LPV": 1, "RPV": 1, "Time (in secs)": i * 0.25}
            for key in ids for i in range(3)
        ])
        for key in ids:
            transcript = eye / "audio_segmentation_transcripts" / key / "transcript.json"
            transcript.parent.mkdir(parents=True)
            transcript.write_text('{"full_text":"mild atelectasis"}', encoding="utf-8")


def tiny_backbones(root):
    vision, text = root / "vit", root / "bert"
    ViTModel(ViTConfig(hidden_size=24, intermediate_size=32, num_attention_heads=4, num_hidden_layers=1)).save_pretrained(vision)
    words = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "mild", "atelectasis")
    backend = Tokenizer(WordPiece(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", model_max_length=32,
    )
    BertModel(BertConfig(vocab_size=len(words), hidden_size=24, intermediate_size=32,
                         num_attention_heads=4, num_hidden_layers=1)).save_pretrained(text)
    tokenizer.save_pretrained(text)
    return vision, text


class SyntheticLLM(BaseHTTPRequestHandler):
    prompts = None

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = data["messages"][0]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("Extract"):
            answer = {"Atelectasis": {"high": ["left lower lobe atelectasis"], "medium": [], "low": []}}
        elif prompt.startswith("Check"):
            answer = [True] * len(json.loads(prompt.rsplit("\n", 1)[-1]))
        else:
            answer = {"findings": "No clear evidence of pneumonia.", "impression": "Mild atelectasis."}
        payload = json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class PipelineTest(unittest.TestCase):
    def test_prepare_train_predict_generate_evaluate(self):
        torch.set_num_threads(2)
        device = os.environ.get("MIMIC_TEST_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            synthetic_dataset(raw)
            manifest = root / "manifest.json"
            main(["prepare", "--root", str(raw), "--output", str(manifest)])
            data = load_manifest(manifest)
            self.assertEqual(data["source_counts"]["images"], 7)
            test_ids = {case["id"] for case in data["cases"] if case["split"] == "test"}
            reflacx = next(case for case in data["cases"] if case["id"] == "sample-5")
            self.assertAlmostEqual(reflacx["gaze"][0][0], 0.1)
            self.assertTrue(any(box["source"] == "reflacx" for box in reflacx["boxes"]))
            duplicate = [case for case in data["cases"] if case["patient"] == "0"]
            self.assertEqual(len({case["split"] for case in duplicate}), 1)
            duplicate[0]["split"] = "test" if duplicate[1]["split"] != "test" else "train"
            with self.assertRaisesRegex(ValueError, "shared across splits"):
                validate_splits(data["cases"])

            vision, text = tiny_backbones(root)
            common = ["--manifest", str(manifest), "--vision-model", str(vision), "--device", device,
                      "--batch", "2", "--epochs", "1"]
            pretraining = root / "pretraining"
            main(["train", "--experiment", "vit-only", *common, "--output", str(pretraining)])
            image_box, text_comparison = root / "image-box", root / "text-comparison"
            main(["train", "--experiment", "image-box", *common, "--encoder-weights",
                  str(pretraining / "model.pt"), "--output", str(image_box)])
            main(["train", "--experiment", "text-encoder", *common, "--text-model", str(text),
                  "--encoder-weights", str(image_box / "model.pt"), "--output", str(text_comparison)])
            old = load_weights(image_box / "model.pt")["model"]
            comparison = load_weights(text_comparison / "model.pt")["model"]
            for key in old:
                if key.startswith(("image_encoder.", "box_encoder.")):
                    torch.testing.assert_close(old[key], comparison[key], rtol=0, atol=0)
            model = root / "enhanced"
            main(["train", "--experiment", "enhanced", *common, "--text-model", str(text), "--refit",
                  "--encoder-weights", str(pretraining / "model.pt"), "--output", str(model)])
            predictions = root / "predictions.json"
            main(["predict", "--manifest", str(manifest), "--checkpoint", str(model / "model.pt"),
                  "--device", device, "--score-gaze", "--output", str(predictions)])
            scored = read_json(predictions)
            self.assertEqual(scored["input_mode"], "image")
            self.assertEqual({case["id"] for case in scored["predictions"]}, test_ids)
            self.assertEqual(scored["gaze_metrics"]["samples"], len(test_ids))

            SyntheticLLM.prompts = []
            with ThreadingHTTPServer(("127.0.0.1", 0), SyntheticLLM) as server:
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
                    keywords, reports = root / "keywords.json", root / "reports.json"
                    options = ["--model", "synthetic", "--endpoint", endpoint]
                    main(["keywords", "--manifest", str(manifest), "--output", str(keywords), *options])
                    main(["generate", "--predictions", str(predictions), "--keywords", str(keywords),
                          "--output", str(reports), *options])
                finally:
                    server.shutdown()
                    worker.join()
            for case in load_manifest(manifest)["cases"]:
                if case["split"] == "test":
                    self.assertNotIn(f"token-{case['patient']}.", SyntheticLLM.prompts[0])
            results = root / "results.json"
            main(["evaluate", "--manifest", str(manifest), "--reports", str(reports), "--output", str(results),
                  "--metrics", "chexpert_f1", "medical"])
            output = read_json(results)
            self.assertEqual(output["scored"], len(test_ids))
            self.assertEqual(output["fallbacks"], 0)
            self.assertEqual(output["metrics"]["chexpert_f1"]["mean"], 1.0)
            self.assertTrue(results.with_suffix(".csv").is_file())

    def test_attention_and_negation_use_reference_evidence(self):
        self.assertNotIn("Pneumonia", chexpert_labels("No clear evidence of pneumonia."))
        saliency = np.zeros((8, 8))
        saliency[0, 0] = 1
        self.assertGreater(gaze_metrics(saliency, [[0, 0, 1, None]])["nss"], 0)
        self.assertLess(gaze_metrics(saliency, [[1, 1, 1, None]])["nss"], 0)
