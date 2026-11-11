import argparse
from pathlib import Path

from . import PAPER


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parser():
    cli = argparse.ArgumentParser(prog="python -m mimic_eye", description="MIMIC-Eye paper pipeline")
    commands = cli.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="Build local patient-grouped dataset manifests")
    prep.add_argument("--root", required=True, type=Path)
    prep.add_argument("--dataset", choices=("mimic-eye", "chexpert"), default="mimic-eye")
    prep.add_argument("--detector-weights", type=Path)
    prep.add_argument("--seed", default=42, type=int)
    prep.add_argument("--device")

    detector = commands.add_parser("train-detector", help="Fit YOLOv8n on expert EyeGaze boxes")
    detector.add_argument("--epochs", type=positive, default=50)
    detector.add_argument("--batch", type=positive, default=16)
    detector.add_argument("--device")

    train = commands.add_parser("train", help="Fit the classifier for a paper experiment")
    train.add_argument("--experiment", choices=PAPER["experiments"], default="enhanced")
    train.add_argument("--image-backbone", choices=("vit-chexpert", "biovil-t", "cxr-clip"), default="vit-chexpert")
    train.add_argument("--box-backbone", choices=("cnn", "convnext-t", "resnet-50", "efficientnetv2-s"), default="cnn")
    train.add_argument("--text-backbone", choices=PAPER["text_models"], default="bioclinicalbert")
    train.add_argument("--encoder-weights", type=Path)
    train.add_argument("--vision-model", help="Override the initial ViT model identifier or local path")
    train.add_argument("--text-model", help="Override the text model identifier or local path")
    train.add_argument("--epochs", type=positive)
    train.add_argument("--batch", type=positive)
    train.add_argument("--device")
    train.add_argument("--refit", action="store_true", help="Refit on train and val for the selected number of epochs")

    predict = commands.add_parser("predict", help="Predict with radiographs only")
    predict.add_argument("--checkpoint", type=Path, required=True)
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--manifest", type=Path)
    predict.add_argument("--batch", type=positive, default=8)
    predict.add_argument("--device")
    predict.add_argument("--score-gaze", action="store_true", help="Score saliency against held-out fixations")

    keywords = commands.add_parser("keywords", help="Build a development vocabulary with a local LLM")
    generate = commands.add_parser("generate", help="Generate reports from classifier predictions")
    generate.add_argument("--predictions", type=Path, required=True)
    generate.add_argument("--keywords", type=Path, required=True)
    generate.add_argument("--provider", choices=("local", "gemini"), default="local")
    for command in (keywords, generate):
        command.add_argument("--model", required=True)
        command.add_argument("--endpoint", help="Local OpenAI-compatible base URL, defaults to http://127.0.0.1:1234/v1")

    evaluate = commands.add_parser("evaluate", help="Score generated reports against local references")
    evaluate.add_argument("--reports", type=Path, required=True)
    metrics = ("bleu", "rouge_l", "meteor", "bertscore", "chexpert_f1", "radgraph_f1", "medical")
    evaluate.add_argument("--metrics", nargs="+", choices=metrics, default=metrics)
    evaluate.add_argument("--radgraph-model", choices=("radgraph", "radgraph-xl"), default="radgraph")

    for command in (detector, train, keywords, evaluate):
        command.add_argument("--manifest", type=Path, required=True)
    for command in (predict, evaluate):
        command.add_argument("--split", choices=("train", "val", "test"), default="test")
    for command in (prep, detector, train, predict, keywords, generate, evaluate):
        command.add_argument("--output", type=Path, required=True)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command not in ("train", "train-detector") and args.output.suffix.lower() != ".json":
        cli.error("--output must be a .json file for this command")
    try:
        if args.command == "prepare":
            from .data import prepare, prepare_chexpert
            if args.dataset == "chexpert":
                if args.detector_weights:
                    raise ValueError("Detector weights are used with MIMIC-Eye")
                prepare_chexpert(args.root, args.output, args.seed)
            else:
                prepare(args.root, args.output, args.seed, args.detector_weights, args.device)
        elif args.command == "train-detector":
            from .detector import train_detector
            train_detector(args.manifest, args.output, args.device, args.epochs, args.batch)
        elif args.command == "train":
            from .train import train
            train(args)
        elif args.command == "predict":
            from .train import predict
            predict(args)
        elif args.command == "keywords":
            from .report import build_keywords
            build_keywords(args)
        elif args.command == "generate":
            from .report import generate
            if args.provider == "gemini" and args.endpoint:
                raise ValueError("--endpoint applies to the local report provider")
            generate(args)
        else:
            from .evaluate import evaluate
            evaluate(args)
    except (ValueError, OSError, ImportError, LookupError) as exc:
        cli.exit(2, f"{args.command}: {exc}\n")


if __name__ == "__main__":
    main()
