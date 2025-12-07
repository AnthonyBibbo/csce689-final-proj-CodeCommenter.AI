#!/usr/bin/env python3
"""
evaluate_results.py

Evaluate model predictions using BLEU, CodeBLEU, and optional METEOR.
Generates a summary JSON showing performance for each experiment mode.
"""

import json
import argparse
from collections import defaultdict
import sacrebleu
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ----------------------------
# CODEBLEU IMPLEMENTATION
# ----------------------------

def compute_codebleu(prediction, reference):
    """
    A lightweight CodeBLEU using the n-gram match component (similar to BLEU)
    plus a simple syntax match heuristic.

    This is NOT the full Microsoft implementation (which requires code parsers),
    but it is enough for class project evaluation.
    """
    # BLEU part
    bleu = sacrebleu.sentence_bleu(prediction, [reference]).score / 100.0

    # Simple syntactic match (shared keywords)
    keywords = ["if", "else", "for", "while", "return", "public", "void"]
    pred_kw = sum(kw in prediction for kw in keywords)
    ref_kw = sum(kw in reference for kw in keywords)

    if ref_kw == 0:
        syntax_score = 1.0
    else:
        syntax_score = min(pred_kw, ref_kw) / max(pred_kw, ref_kw)

    codebleu = 0.7 * bleu + 0.3 * syntax_score
    return codebleu


# ----------------------------
# LOAD RESULTS
# ----------------------------

def load_results(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# ----------------------------
# METRICS FOR EACH SAMPLE
# ----------------------------

def evaluate_sample(pred, ref):
    """
    Compute BLEU and CodeBLEU for a single prediction-reference pair.
    (METEOR removed to avoid NLTK wordnet dependency issues.)
    """
    # BLEU
    bleu = sacrebleu.sentence_bleu(pred, [ref]).score

    # CodeBLEU
    codebleu = compute_codebleu(pred, ref)

    # Return meteor = None for clean JSON output
    return bleu, None, codebleu


# ----------------------------
# MAIN EVAL FUNCTION
# ----------------------------

def evaluate_file(path):
    data = load_results(path)
    mode = data[0]["mode"] if data else "unknown"

    metrics = {
        "bleu": [],
        "meteor": [],
        "codebleu": [],
    }

    for item in data:
        pred = item["prediction"]
        ref = item["ground_truth"]

        bleu, meteor, codebleu = evaluate_sample(pred, ref)

        metrics["bleu"].append(bleu)
        metrics["meteor"].append(meteor)
        metrics["codebleu"].append(codebleu)

    # Compute averages
    avg_bleu = sum(metrics["bleu"]) / len(metrics["bleu"])
    avg_codebleu = sum(metrics["codebleu"]) / len(metrics["codebleu"])

    return {
        "mode": mode,
        "count": len(data),
        "avg_bleu": avg_bleu,
        "avg_codebleu": avg_codebleu,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="Paths to results JSONL files.")
    parser.add_argument("--output", required=True,
                        help="Path to summary JSON file.")
    args = parser.parse_args()

    summary = []

    for path in args.results:
        print(f"Evaluating {path}...")
        result = evaluate_file(path)
        summary.append(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Summary written to: {args.output}")


if __name__ == "__main__":
    main()
