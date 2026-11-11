import json
import math
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
from dotenv import dotenv_values

from . import LABELS
from .data import REGIONS, load_manifest, manifest_digest, read_json, write_json

BUCKETS = ("high", "medium", "low")
ALIASES = {
    "cardiac silhouette": ("heart shadow", "cardiac border", "heart size", "heart silhouette"),
    "trachea": ("windpipe", "airway", "tracheal tube", "endotracheal tube"),
    "upper mediastinum": ("superior mediastinum", "mediastinal contour"),
    **{
        f"{side} {region}": tuple(f"{side} {alias}" for alias in aliases)
        for side in ("left", "right")
        for region, aliases in {
            "clavicle": ("collarbone", "clavicular bone"),
            "costophrenic angle": ("costophrenic sulcus", "costodiaphragmatic recess"),
            "hilar structures": ("hilum", "hilar region", "lung root"),
            "lower lung zone": ("lower lung field", "lower lobe", "lung base", "basal lung"),
            "lung": ("lung field", "hemithorax", "pulmonary field"),
            "mid lung zone": ("middle lung field", "midlung", "mid lung"),
            "upper lung zone": ("upper lung field", "upper lobe", "lung apex", "apical lung"),
        }.items()
    },
}


def read_report(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sections = re.findall(
        r"(?ims)^\s*(FINDINGS|IMPRESSION)\s*:\s*(.*?)(?=^\s*[A-Z][A-Z ]+:\s*|\Z)", text,
    )
    result = "\n".join(f"{title.upper()}:\n{' '.join(body.split())}" for title, body in sections if body.strip())
    return result or text.strip()


class ReportModel:
    def __init__(self, provider, model, endpoint=None, local_only=False):
        self.provider = provider
        self.model = model
        self.session = requests.Session()
        self.session.trust_env = False
        self.headers = {}
        if provider == "local":
            base = (endpoint or "http://127.0.0.1:1234/v1").rstrip("/")
            parsed = urlsplit(base)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("The local endpoint needs an HTTP or HTTPS URL")
            if local_only and parsed.hostname not in ("127.0.0.1", "::1", "localhost"):
                raise ValueError("Building vocabulary from reports requires a loopback endpoint")
            self.url = base + "/chat/completions"
        else:
            key = os.environ.get("GEMINI_API_KEY") or dotenv_values(".env", interpolate=False).get("GEMINI_API_KEY")
            if not key:
                raise ValueError("Set GEMINI_API_KEY in the environment or .env to use the Gemini report provider")
            self.headers["x-goog-api-key"] = key
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent"

    def json_response(self, prompt, temperature):
        if self.provider == "local":
            payload = {"model": self.model, "temperature": temperature, "max_tokens": 4096,
                       "messages": [{"role": "user", "content": prompt}]}
        else:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "topK": 1, "maxOutputTokens": 4096,
                                     "responseMimeType": "application/json"},
            }
        for attempt in range(5):
            try:
                response = self.session.post(
                    self.url, json=payload, headers=self.headers, timeout=120, allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt == 4:
                    raise RuntimeError("Report provider request failed") from exc
            else:
                if 200 <= response.status_code < 300:
                    data = response.json()
                    if self.provider == "local":
                        raw = data["choices"][0]["message"]["content"]
                    else:
                        raw = "".join(
                            part["text"] for part in data["candidates"][0]["content"]["parts"]
                            if not part.get("thought") and "text" in part
                        )
                    return json.loads(re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", raw.strip()))
                if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                    raise RuntimeError(f"Report provider returned HTTP {response.status_code}")
            time.sleep(3 * 2 ** attempt)


def unique_terms(values):
    terms = {}
    for value in values:
        if isinstance(value, str) and 0 < len(value.split()) <= 8:
            text = " ".join(value.strip(" .:").split())
            if text:
                terms.setdefault(text.casefold(), text)
    return list(terms.values())


def build_keywords(args):
    manifest = load_manifest(args.manifest)
    client = ReportModel("local", args.model, args.endpoint, local_only=True)
    reports = dict.fromkeys(
        case["report"] for case in manifest["cases"] if case["split"] != "test" and case["report"]
    )
    paths = list(reports)
    if not paths:
        raise ValueError("No training or validation reports are available for keyword extraction")
    keywords = {label: {bucket: [] for bucket in BUCKETS} for label in LABELS}
    example = json.dumps({label: {bucket: [] for bucket in BUCKETS} for label in LABELS})
    for start in range(0, len(paths), 30):
        text = "\n\n".join(Path(path).read_text(encoding="utf-8", errors="replace") for path in paths[start:start + 30])
        prompt = (
            "Extract short diagnostic terms, ranked by confidence, from these development chest X-ray "
            "reports. Assign each term to the condition it actually supports. Omit names, identifiers, dates, "
            "and report boilerplate. Return only JSON using these keys and lists of strings.\n"
            f"{example}\n\nReports\n{text}"
        )
        extracted = client.json_response(prompt, temperature=0.1)
        if not isinstance(extracted, dict) or not set(extracted) & set(LABELS):
            raise ValueError("Keyword extraction must return condition objects")
        for label, buckets in keywords.items():
            for bucket in BUCKETS:
                values = extracted.get(label, {}).get(bucket, [])
                if not isinstance(values, list):
                    raise ValueError("Keyword confidence buckets must be lists")
                buckets[bucket].extend(values)
        print(f"Read {min(start + 30, len(paths))}/{len(paths)} development reports", flush=True)
    candidates = [
        (label, bucket, term) for label, buckets in keywords.items()
        for bucket in BUCKETS for term in unique_terms(buckets[bucket])
    ]
    cleaned = {label: {bucket: [] for bucket in BUCKETS} for label in LABELS}
    seen = {label: set() for label in LABELS}
    for start in range(0, len(candidates), 30):
        batch = candidates[start:start + 30]
        prompt = (
            "Check each numbered term against its assigned chest X-ray condition. Keep only relevant terms. "
            "Reject lexical variants, generic boilerplate, identifiers, and terms from unrelated conditions. "
            "Return a JSON list of booleans in the same order.\n"
            + json.dumps([[label, term] for label, _, term in batch])
        )
        keep = client.json_response(prompt, temperature=0.1)
        if not isinstance(keep, list) or len(keep) != len(batch) or any(type(value) is not bool for value in keep):
            raise ValueError("Keyword cleaning must return one boolean per candidate")
        for (label, bucket, term), accepted in zip(batch, keep):
            key = term.casefold()
            if accepted and key not in seen[label]:
                cleaned[label][bucket].append(term)
                seen[label].add(key)
    if not any(seen.values()):
        raise ValueError("Keyword extraction produced an empty vocabulary")
    write_json(args.output, {
        "manifest_sha256": manifest_digest(args.manifest), "model": args.model,
        "development_reports": len(paths), "keywords": cleaned, "regions": manifest["regions"],
    })


def match_regions(terms, atlas):
    matched = {}
    for name in REGIONS:
        for alias in (name, *ALIASES[name]):
            pattern = rf"\b{re.escape(alias)}\b"
            if any(
                re.search(pattern, term.casefold()) or SequenceMatcher(None, alias, term.casefold()).ratio() >= 0.90
                for term in terms
            ):
                matched[name] = atlas.get(name)
                break
    return matched


def report_inputs(probabilities, vocabulary):
    if set(probabilities) != set(LABELS) or any(
        not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise ValueError("Predictions must have a probability in [0, 1] for each condition")
    selected = {label: p for label, p in probabilities.items() if p > 0.60}
    if len(selected) > 1:
        selected.pop("No Finding", None)
    result = []
    for label, probability in sorted(selected.items(), key=lambda row: (-row[1], row[0])):
        bucket = "high" if probability >= 0.70 else "medium"
        terms = vocabulary["keywords"][label][bucket][:12]
        result.append({
            "condition": label, "probability": probability, "keyword_bucket": bucket, "keywords": terms,
            "regions": match_regions(terms, vocabulary["regions"]),
        })
    return result


def generate(args):
    predictions = read_json(args.predictions)
    vocabulary = read_json(args.keywords)
    digest = predictions.get("manifest_sha256")
    if digest and digest != vocabulary.get("manifest_sha256"):
        raise ValueError("Predictions and development vocabulary were built from different manifests")
    client = ReportModel(args.provider, args.model, args.endpoint)
    reports = []
    for prediction in predictions["predictions"]:
        context = report_inputs(prediction["probabilities"], vocabulary)
        prompt = (
            "Write a concise chest X-ray report using the supplied classifier findings and anatomical regions. "
            "Use uncertainty that reflects the probabilities. Describe only supported findings. "
            "Do not invent location, laterality, comparison, severity, devices, or a normal exam. "
            "Region bounds are normalized image coordinates. If evidence is insufficient, say so. "
            "Return JSON with nonempty string fields findings and impression.\n"
            + json.dumps(context)
        )
        try:
            report = client.json_response(prompt, temperature=0.3)
            findings, impression = (report[key].strip() for key in ("findings", "impression"))
            if not findings or not impression:
                raise ValueError("Incomplete report")
            status = "generated"
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            findings = "Automated report generation was unavailable."
            impression = "Review the classifier outputs and the radiograph."
            status = "fallback"
        reports.append({
            "id": prediction["id"], "text": f"FINDINGS:\n{findings}\nIMPRESSION:\n{impression}", "status": status,
            "probabilities": prediction["probabilities"], "prompt_inputs": context,
        })
    write_json(args.output, {"provider": args.provider, "model": args.model, "split": predictions.get("split"),
                             "manifest_sha256": digest, "reports": reports})
    count = sum(report["status"] == "generated" for report in reports)
    print(f"Generated {count}/{len(reports)} reports. Fallbacks: {len(reports) - count}")
