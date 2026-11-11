import csv
import re
from pathlib import Path

import numpy as np

from .data import load_manifest, manifest_digest, read_json, write_json
from .report import read_report

CHEXPERT_TERMS = {
    "No Finding": ("no finding", "normal", "clear", "no acute", "unremarkable", "within normal limits"),
    "Enlarged Cardiomediastinum": ("enlarged cardiomediastinum", "wide mediastinum"),
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac enlargement"),
    "Lung Opacity": ("opacity", "opacities", "opacification"),
    "Lung Lesion": ("lesion", "nodule", "mass"),
    "Edema": ("edema", "pulmonary edema", "congestion"),
    "Consolidation": ("consolidation", "infiltrate"),
    "Pneumonia": ("pneumonia", "infection"),
    "Atelectasis": ("atelectasis", "collapse"),
    "Pneumothorax": ("pneumothorax",),
    "Pleural Effusion": ("effusion", "pleural fluid"),
    "Pleural Other": ("pleural thickening", "pleural disease"),
    "Fracture": ("fracture",),
    "Support Devices": ("device", "tube", "line", "catheter", "pacemaker", "band", "wire", "lead", "clip", "stent", "implant"),
}
MEDICAL_VOCAB = (
    """lung lungs pulmonary respiratory bronchi bronchus alveoli pleura pleural diaphragm mediastinum heart
    cardiac cardio aorta aortic ventricle atrium ribs sternum clavicle spine vertebra thorax thoracic apex base
    hilum hila bilateral unilateral upper_lobe middle_lobe lower_lobe lingula right_lung left_lung costophrenic
    cardiophrenic""",
    """pneumonia infection infiltrate consolidation opacity effusion fluid edema congestion atelectasis
    pneumothorax collapse mass nodule lesion cardiomegaly enlargement hypertrophy dilation copd emphysema
    fibrosis scarring granuloma tuberculosis cancer malignancy metastasis fracture break displacement deformity""",
    """normal unremarkable clear patent intact within_normal_limits wnl negative no_abnormalities stable
    unchanged appropriate adequate""",
)
ABBREVIATIONS = {
    "wnl": "within normal limits", "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure", "pe": "pulmonary embolism", "ptx": "pneumothorax",
    "cxr": "chest x-ray", "ct": "computed tomography", "mri": "magnetic resonance imaging",
    "rul": "right upper lobe", "rml": "right middle lobe", "rll": "right lower lobe",
    "lul": "left upper lobe", "lll": "left lower lobe", "bil": "bilateral",
    "r/o": "rule out", "h/o": "history of", "s/p": "status post",
}
NEGATION = re.compile(
    r"\b(no|not|negative|absent|without|denies|denied|unremarkable|clear of|free of|rule out|cannot exclude)\b",
)
SEVERITY = ("mild minimal slight small", "moderate medium", "severe marked significant large", "massive extensive complete")


def f1_overlap(a, b, empty=0.0):
    size = len(a) + len(b)
    return 2 * len(a & b) / size if size else empty


def chexpert_labels(text):
    """Identify positive CheXpert categories using the repository's term-based scorer."""
    labels = set()
    for clause in re.split(r"[.;\n]|\b(?:but|however)\b", text.lower()):
        for label, terms in CHEXPERT_TERMS.items():
            for term in terms:
                for match in re.finditer(rf"\b{re.escape(term)}s?\b", clause):
                    before = " ".join(clause[:match.start()].split()[-6:])
                    after = clause[match.end():]
                    if label == "No Finding" or (
                        not NEGATION.search(before)
                        and not re.match(r"\s+(?:is\s+)?(?:absent|not seen|cannot be excluded)\b", after)
                    ):
                        labels.add(label)
    if len(labels) > 1:
        labels.discard("No Finding")
    return labels


def concepts(text):
    text = text.lower()
    for word, expansion in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(word)}\b", expansion, text)
    tokens = set(re.findall(r"\b\w+\b", text))
    groups = []
    for vocabulary in MEDICAL_VOCAB:
        terms = set(vocabulary.split())
        phrases = {term for term in terms if "_" in term and re.search(r"\b" + term.replace("_", r"\s+") + r"\b", text)}
        groups.append((tokens & terms) | phrases)
    return groups, tokens & set(" ".join(MEDICAL_VOCAB).split())


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def severity(text):
    for index, words in enumerate(SEVERITY, start=1):
        if re.search(r"\b(" + "|".join(words.split()) + r")\b", text.lower()):
            return index
    return 0


def medical_score(reference, candidate):
    ref, ref_terms = concepts(reference)
    cand, cand_terms = concepts(candidate)
    a, b = severity(reference), severity(candidate)
    severity_score = 0.5 if bool(a) != bool(b) else (1, 0.7, 0.4, 0.1)[min(abs(a - b), 3)]
    pattern = r"\b(?:no|not|negative|absent|without|denies|denied|unremarkable|normal)\b"
    phrases = r"\bno (?:evidence of|signs of|indication of)\b"
    negations = [sum(len(re.findall(p, text.lower())) for p in (pattern, phrases)) for text in (reference, candidate)]
    negation = (1, 0.5, 0)[min(abs(negations[0] - negations[1]), 2)]
    if (ref[2] and cand[1]) or (ref[1] and cand[2]):
        pathology = 0.0
    elif ref[1] and cand[1]:
        pathology = jaccard(ref[1], cand[1])
    else:
        pathology = 1.0 if ref[2] and cand[2] and not (ref[1] or cand[1]) else 0.5
    return (
        0.25 * f1_overlap(set.union(*ref), set.union(*cand), empty=1.0)
        + 0.20 * jaccard(ref[0], cand[0]) + 0.25 * pathology
        + 0.15 * negation + 0.10 * severity_score + 0.05 * jaccard(ref_terms, cand_terms)
    )


def graph_relations(graph):
    entities = graph["entities"]
    nodes = {key: (value["label"], value["tokens"].lower()) for key, value in entities.items()}
    return {
        (nodes[source], predicate, nodes[target])
        for source, entity in entities.items() for predicate, target in entity["relations"] if target in nodes
    }


def score_reports(pairs, metrics, radgraph_model):
    reference, candidate = zip(*pairs)
    results = {}
    if "bleu" in metrics:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        smooth = SmoothingFunction().method1
        results["bleu"] = [sentence_bleu([ref.lower().split()], gen.lower().split(), smoothing_function=smooth)
                           for ref, gen in pairs]
    if "rouge_l" in metrics:
        from rouge_score.rouge_scorer import RougeScorer
        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        results["rouge_l"] = [scorer.score(ref, gen)["rougeL"].fmeasure for ref, gen in pairs]
    if "meteor" in metrics:
        from nltk.translate.meteor_score import meteor_score
        results["meteor"] = [meteor_score([ref.lower().split()], gen.lower().split()) for ref, gen in pairs]
    if "bertscore" in metrics:
        from bert_score import score
        _, _, f1 = score(list(candidate), list(reference), model_type="bert-base-uncased", num_layers=9)
        results["bertscore"] = f1.cpu().tolist()
    if "chexpert_f1" in metrics:
        results["chexpert_f1"] = [f1_overlap(chexpert_labels(ref), chexpert_labels(gen)) for ref, gen in pairs]
    if "radgraph_f1" in metrics:
        from radgraph import RadGraph
        parser = RadGraph(model_type=radgraph_model)
        results["radgraph_f1"] = []
        for start in range(0, len(pairs), 8):
            batch = pairs[start:start + 8]
            graphs = parser([text for pair in batch for text in pair])
            results["radgraph_f1"].extend(
                f1_overlap(graph_relations(graphs[str(i)]), graph_relations(graphs[str(i + 1)]))
                for i in range(0, 2 * len(batch), 2)
            )
    if "medical" in metrics:
        results["medical"] = [medical_score(ref, gen) for ref, gen in pairs]
    return results


def evaluate(args):
    manifest = load_manifest(args.manifest)
    expected = {case["id"]: case for case in manifest["cases"] if case["split"] == args.split}
    generated = read_json(args.reports)
    if generated.get("manifest_sha256") and generated["manifest_sha256"] != manifest_digest(args.manifest):
        raise ValueError("Report provenance does not match the evaluation manifest")
    items = generated["reports"]
    reports = {report["id"]: report for report in items}
    if len(reports) != len(items) or reports.keys() != expected.keys():
        raise ValueError(f"Reports must contain exactly one entry for each of the {len(expected)} {args.split} images")
    ids, pairs, fallbacks = [], [], 0
    for case_id, case in expected.items():
        report = reports[case_id]
        if report.get("status") == "fallback":
            fallbacks += 1
            continue
        if not case["report"]:
            raise ValueError(f"Missing reference for {case_id}")
        ref, candidate = read_report(case["report"]), report["text"].strip()
        if not ref or not candidate:
            raise ValueError(f"Empty report for {case_id}")
        ids.append(case_id)
        pairs.append((ref, candidate))
    if not pairs:
        raise ValueError("No generated reports are available to score")
    scores = score_reports(pairs, args.metrics, args.radgraph_model)
    rows = [{"id": case_id, **{key: float(values[i]) for key, values in scores.items()}}
            for i, case_id in enumerate(ids)]
    summary = {key: {"mean": float(np.mean(values)), "std": float(np.std(values))} for key, values in scores.items()}
    write_json(args.output, {
        "split": args.split, "expected": len(expected), "scored": len(rows), "fallbacks": fallbacks,
        "metrics": summary, "per_image": rows,
    })
    csv_path = Path(args.output).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", *scores])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(rows)}/{len(expected)} {args.split} reports. Excluded fallbacks: {fallbacks}")
