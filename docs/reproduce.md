# Reproduce the experiments

## Data and checkpoints

Obtain [MIMIC-Eye v1.0.0](https://physionet.org/content/mimic-eye-multimodal-datasets/1.0.0/) through PhysioNet. Point `--root` at the extracted directory containing `patient_*` folders. The command reads the original EyeGaze, REFLACX, and CXR tables. It keeps readable radiographs with fixations, a reference report, and at least one of the eight retained labels. It writes a patient-grouped split, the training gaze statistics, and region bounds computed from train and validation data.

```sh
python -m mimic_eye prepare --root /path/to/mimic-eye --output work/cases.json
python -m mimic_eye train-detector --manifest work/cases.json --output work/detector
python -m mimic_eye prepare --root /path/to/mimic-eye --detector-weights work/detector/fit/weights/best.pt --output work/cases.json
```

The second command trains YOLOv8n on expert EyeGaze boxes and scores each split. With detector weights, preparation adds the highest-confidence prediction for each missing region and retains expert boxes and ellipses. If you already have the detector weights, run the last command directly.

The classifier takes CheXpert ViT weights via `--encoder-weights`. To train them locally, use the original [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) CSV and images.

```sh
python -m mimic_eye prepare --dataset chexpert --root /path/to/CheXpert-v1.0-small --output work/chexpert.json
python -m mimic_eye train --experiment vit-only --manifest work/chexpert.json --output work/pretrained
```

## Classify and generate reports

```sh
python -m mimic_eye train --experiment enhanced --manifest work/cases.json --encoder-weights work/pretrained/model.pt --output work/enhanced --refit
python -m mimic_eye predict --checkpoint work/enhanced/model.pt --manifest work/cases.json --score-gaze --output work/predictions.json
python -m mimic_eye keywords --manifest work/cases.json --model LOCAL_MODEL --output work/keywords.json
python -m mimic_eye generate --predictions work/predictions.json --keywords work/keywords.json --model LOCAL_MODEL --output work/reports.json
python -m nltk.downloader wordnet
python -m mimic_eye evaluate --manifest work/cases.json --reports work/reports.json --output work/scores.json
```

The selected experiment settings are in [paper.json](../mimic_eye/paper.json). `--refit` selects the epoch and thresholds on validation data, then trains for that many epochs on train and validation together. Use `--batch 8` for a smaller GPU. Prediction defaults to the test split and takes only the image as model input. `--score-gaze` compares saliency with recorded fixations after prediction. Use `--image` in place of `--manifest` for a single radiograph.

Set `LOCAL_MODEL` to the name served by an OpenAI-compatible local LLM. The default endpoint is `http://127.0.0.1:1234/v1`. Keyword extraction reads train and validation reports at a loopback endpoint.

For Gemini, copy `.env.example` to `.env` in your working directory and fill in `GEMINI_API_KEY`. An environment variable takes precedence over the file. Run `generate` with `--provider gemini --model gemini-2.5-pro`. The prompt contains classifier findings, terms, and region matches. Apply [PhysioNet's guidance on MIMIC data and LLMs](https://physionet.org/news/post/llm-responsible-use/) when using external services.

| Paper tables | Commands and settings |
| --- | --- |
| 3–6 | `prepare` writes cohort summaries. `train-detector` writes aggregate and per-region detection scores. |
| 7 and 10 | Train `image-box`, `image-box-text`, `raw-gaze`, and `enhanced`. Use `predict --score-gaze` to collect classifier and alignment scores. |
| 8 | Use `--experiment image-box`. Select `vit-chexpert`, `biovil-t`, or `cxr-clip` for the image backbone and `cnn`, `convnext-t`, `resnet-50`, or `efficientnetv2-s` for the box encoder. For [BioViL-T](https://github.com/microsoft/hi-ml), install `".[biovil]"`. For [CXR-CLIP](https://github.com/Soombit-ai/cxr-clip), supply the official ResNet-50 checkpoint as `--encoder-weights`. |
| 9 | Use `--experiment text-encoder` with the trained `image-box` checkpoint as `--encoder-weights`. Select `bioclinicalbert`, `pubmedbert`, or `biomed-roberta` with `--text-backbone`. |
| 11 | Generate and score reports separately for `enhanced`, `raw-gaze`, and `image-box-text` with each loaded report model. |
| 14 | To score external VLM runs, pass JSON containing `{"reports":[{"id":"image-id","text":"report text"}]}` to `evaluate`. Include each image in the selected split once. The original Look & Mark runner is unavailable in this checkout. |

The evaluator reports BLEU-4, ROUGE-L, METEOR, BERTScore, CheXpert category agreement, RadGraph relation F1, and MEDICAL. CheXpert category agreement uses a term and negation heuristic. Report failures are marked as fallbacks and counted separately. The evaluator writes per-image scores in JSON and CSV.

The original run manifests, checkpoints, and report outputs are required to verify the accepted numerical results. This implementation uses patient-disjoint splits, validation-only thresholds, image-only prediction, and NSS at observed fixation locations. These changes require fresh runs and can change scores. The previous CXR-CLIP option pointed to a generic CLIP model. The current option uses CXR-CLIP weights. Classifier checkpoints produced by this package carry the model configuration used by prediction.
