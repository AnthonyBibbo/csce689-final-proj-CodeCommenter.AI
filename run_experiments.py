#!/usr/bin/env python3
"""
run_experiments.py

Run zero-shot, few-shot, and domain-aware experiments using your dataset.

Usage:

    python run_experiments.py \
        --dataset data/datasets/unity_dataset.jsonl \
        --model gpt-4o-mini \
        --mode zero-shot \
        --output results_zero_shot.jsonl \
        --max-samples 25
"""

import json
import argparse
from openai import OpenAI

from code_commenter import build_prompt   # reuse your prompt logic


# -----------------------------------------------------
# UTILITIES
# -----------------------------------------------------

def load_dataset(path):
    """
    Loads dataset jsonl. Returns a list of records.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def few_shot_prefix(samples, k=3):
    """
    Generate a few-shot prefix using k dataset examples.
    """
    prefix = ""
    for s in samples[:k]:
        prefix += (
            f"### Example Comment for function '{s['function_name']}'\n"
            f"Code:\n{s['code']}\n"
            f"Comment:\n{s['comment']}\n\n"
        )
    return prefix


def domain_hints(language="unity"):
    """
    Add domain-specific guidance for the LLM.
    """
    if language == "unity":
        return (
            "IMPORTANT UNITY CONTEXT:\n"
            "- Unity uses MonoBehaviour lifecycle methods (Start, Update, FixedUpdate).\n"
            "- Use clear comments explaining gameplay purpose.\n"
            "- Explain physics callbacks (OnCollisionEnter, OnTriggerEnter).\n"
            "- Keep comments concise (1–3 lines).\n\n"
        )
    return ""


# -----------------------------------------------------
# LLM CALL
# -----------------------------------------------------

def call_llm(model, prompt_text, temperature=0.2):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=180,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# -----------------------------------------------------
# EXPERIMENT RUNNER
# -----------------------------------------------------

def run_experiment(dataset, model, mode, output_path, max_samples=20):
    """
    Modes:
        zero-shot
        few-shot
        domain
        few-shot-domain
    """
    k_shot = 3
    fewshot_text = few_shot_prefix(dataset, k=k_shot)
    domain_text = domain_hints("unity")

    results = []

    for i, record in enumerate(dataset[:max_samples]):
        func_name = record["function_name"]
        code = record["code"]
        true_comment = record["comment"]

        # Build prompt depending on mode
        if mode == "zero-shot":
            prompt = f"### Code:\n{code}\n\nWrite a concise comment (1-3 lines) describing what this Unity function does."

        elif mode == "few-shot":
            prompt = (
                fewshot_text +
                f"### Now generate a comment for function '{func_name}':\n{code}"
            )

        elif mode == "domain":
            prompt = domain_text + f"### Code:\n{code}\n\nWrite a concise Unity-style comment."

        elif mode == "few-shot-domain":
            prompt = (
                fewshot_text +
                domain_text +
                f"### Generate a comment for '{func_name}':\n{code}"
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"[{i+1}/{max_samples}] Running {mode} on function '{func_name}'...")

        try:
            pred = call_llm(model, prompt)
        except Exception as e:
            print(f"[ERROR] {e}")
            pred = "ERROR_GENERATING_COMMENT"

        results.append({
            "function_name": func_name,
            "code": code,
            "ground_truth": true_comment,
            "prediction": pred,
            "mode": mode
        })

    # Write results to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Results saved to: {output_path}")


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["zero-shot", "few-shot", "domain", "few-shot-domain"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    run_experiment(dataset, args.model, args.mode, args.output, args.max_samples)


if __name__ == "__main__":
    main()
