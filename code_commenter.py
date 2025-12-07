#!/usr/bin/env python3
"""
CodeCommenter.AI - Simple pipeline prototype

Usage:
    python code_commenter.py input_file.cs --model gpt-4o-mini --max-functions 20 --dry-run

- Reads a Unity (.cs) or Unreal (.cpp/.h) file
- Finds function-level code chunks
- Asks an LLM to generate a concise comment for each function
- Writes an annotated version of the file with comments inserted above functions
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from openai import OpenAI

# ------------- Config / Client ------------- #

def get_openai_client() -> "OpenAI":
    """
    Returns an OpenAI client. Requires OPENAI_API_KEY in your env.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")
    # The OpenAI client automatically reads OPENAI_API_KEY
    return OpenAI()


# ------------- Data Structures ------------- #

@dataclass
class CodeChunk:
    """Represents a function-level chunk of code."""
    name: str
    start_line: int   # inclusive, 0-based
    end_line: int     # inclusive, 0-based
    code: str


# ------------- Language Detection ------------- #

def detect_language_from_extension(path: str) -> str:
    """
    Return 'unity' for C# (.cs) and 'unreal' for C++ (.cpp/.h).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cs":
        return "unity"
    elif ext in (".cpp", ".cc", ".cxx", ".h", ".hpp"):
        return "unreal"
    else:
        # Default to 'unity' but you can change this
        return "unity"


def get_comment_prefix(language: str) -> str:
    """
    Both C# and C++ accept // comments. We keep it simple.
    """
    return "// "


# ------------- Naive Function Parser ------------- #

CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch", "foreach", "else"}


def looks_like_function_signature(line: str) -> bool:
    """
    Heuristic to determine if a line looks like the start of a function.

    Very naive but good enough for a class project:
    - Contains '(' and ')'
    - Does NOT start with control keywords (if, for, while, etc.)
    - Does NOT end with ';' (to avoid prototypes / declarations)
    """
    stripped = line.strip()
    if "(" not in stripped or ")" not in stripped:
        return False
    if stripped.endswith(";"):
        return False

    # get first word
    first_token = re.split(r"\s+", stripped)[0]
    first_token = first_token.split("(")[0]
    if first_token in CONTROL_KEYWORDS:
        return False

    return True


def find_function_chunks(lines: List[str]) -> List[CodeChunk]:
    """
    Naively find function-level chunks using brace counting from signatures.
    Returns a list of CodeChunk objects.
    """
    chunks: List[CodeChunk] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if looks_like_function_signature(line):
            # Find where the opening brace { is
            brace_line_index = i
            if "{" not in line:
                # maybe brace is on the next line
                j = i + 1
                while j < n and "{" not in lines[j]:
                    j += 1
                if j >= n:
                    i += 1
                    continue
                brace_line_index = j

            # Now do brace counting from brace_line_index
            brace_count = 0
            end_index = brace_line_index
            j = brace_line_index
            while j < n:
                brace_count += lines[j].count("{")
                brace_count -= lines[j].count("}")
                if brace_count == 0:
                    end_index = j
                    break
                j += 1

            # Extract function name (best-effort)
            func_name = extract_function_name(line)

            code_block = "".join(lines[i:end_index + 1])
            chunks.append(CodeChunk(
                name=func_name or f"func_at_line_{i+1}",
                start_line=i,
                end_line=end_index,
                code=code_block
            ))

            i = end_index + 1
        else:
            i += 1

    return chunks


def extract_function_name(signature_line: str) -> str:
    """
    Very naive function name extractor:
    - Looks for pattern 'name(' and returns name.
    """
    # Remove generics to simplify (e.g., List<int> -> List)
    line = re.sub(r"<[^>]*>", "", signature_line)
    match = re.search(r"(\w+)\s*\(", line)
    if not match:
        return ""
    candidate = match.group(1)

    # Avoid catching control keywords or return types like 'int'
    if candidate in CONTROL_KEYWORDS:
        return ""
    return candidate


# ------------- LLM Comment Generation ------------- #

def build_prompt(code: str, language: str, func_name: str) -> List[dict]:
    """
    Build chat prompt for the LLM.
    """
    if language == "unity":
        domain_context = (
            "The code is from a Unity C# script. "
            "Unity uses MonoBehaviour methods like Start, Update, FixedUpdate, "
            "OnCollisionEnter, etc. "
        )
    else:
        domain_context = (
            "The code is from an Unreal Engine C++ Actor or Component class. "
            "Unreal uses macros like UCLASS, UPROPERTY, and methods like BeginPlay, Tick, etc. "
        )

    system_msg = (
        "You are a senior game developer writing concise, clear comments for game engine code. "
        "Your job is to write a short, high-quality comment that explains what a function does, "
        "its purpose, and any important side effects. "
        "The style should be professional but easy to read."
    )

    user_msg = (
        f"{domain_context}\n\n"
        f"Write a single, concise comment (1–3 lines) that should be placed directly above "
        f"the function named '{func_name}'. Do NOT rewrite the function code. "
        f"Do NOT include comment markers like // in your response.\n\n"
        f"Here is the function code:\n\n"
        f"{code}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def generate_comment_for_chunk(
    client: "OpenAI",
    model: str,
    chunk: CodeChunk,
    language: str,
) -> str:
    """
    Call the LLM to generate a comment for a specific function chunk.
    """
    messages = build_prompt(chunk.code, language, chunk.name)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=120,
    )

    comment_text = response.choices[0].message.content.strip()

    # Normalize comment: split into lines, strip whitespace
    lines = [line.strip() for line in comment_text.splitlines() if line.strip()]
    return "\n".join(lines)


# ------------- Inserting Comments ------------- #

def insert_comments_into_lines(
    lines: List[str],
    chunks: List[CodeChunk],
    comments: List[str],
    language: str,
) -> List[str]:
    """
    Given the original lines, a list of chunks, and matching comments,
    return a new list of lines with comments inserted above each chunk.
    """
    comment_prefix = get_comment_prefix(language)
    # Make sure chunks are sorted by start_line
    chunks_with_comments = sorted(
        zip(chunks, comments),
        key=lambda x: x[0].start_line
    )

    new_lines: List[str] = []
    current_index = 0

    for chunk, comment in chunks_with_comments:
        # Copy lines before this chunk
        while current_index < chunk.start_line:
            new_lines.append(lines[current_index])
            current_index += 1

        # Insert comment
        indent = get_line_indentation(lines[chunk.start_line])
        for comment_line in comment.splitlines():
            new_lines.append(f"{indent}{comment_prefix}{comment_line}\n")

        # Now copy the chunk lines themselves
        while current_index <= chunk.end_line:
            new_lines.append(lines[current_index])
            current_index += 1

    # Copy any remaining lines after the last chunk
    while current_index < len(lines):
        new_lines.append(lines[current_index])
        current_index += 1

    return new_lines


def get_line_indentation(line: str) -> str:
    """
    Return leading whitespace of a line (for nicely aligned comments).
    """
    return line[:len(line) - len(line.lstrip())]


# ------------- CLI / Main ------------- #

def main():
    parser = argparse.ArgumentParser(description="CodeCommenter.AI - LLM-powered comment generator")
    parser.add_argument("input_file", help="Path to input .cs or .cpp/.h file")
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output file (default: <input>.commented.<ext>)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, gpt-4o, etc.)",
    )
    parser.add_argument(
        "--max-functions",
        type=int,
        default=20,
        help="Maximum number of functions to annotate (for cost control)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API, just show what would happen",
    )

    args = parser.parse_args()

    input_path = args.input_file
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    language = detect_language_from_extension(input_path)
    print(f"[INFO] Detected language: {language}")

    chunks = find_function_chunks(lines)
    print(f"[INFO] Found {len(chunks)} function-like chunks")

    if not chunks:
        print("[WARN] No functions detected. Exiting.")
        return

    if len(chunks) > args.max_functions:
        print(f"[INFO] Limiting to first {args.max_functions} functions for this run.")
        chunks = chunks[:args.max_functions]

    comments: List[str] = []

    if args.dry_run:
        print("[DRY RUN] Skipping API calls. Listing detected functions:")
        for chunk in chunks:
            print(f"  - {chunk.name} (lines {chunk.start_line+1}-{chunk.end_line+1})")
        return

    client = get_openai_client()

    for idx, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Generating comment for function {idx}/{len(chunks)}: {chunk.name} "
              f"(lines {chunk.start_line+1}-{chunk.end_line+1})")
        try:
            comment = generate_comment_for_chunk(client, args.model, chunk, language)
        except Exception as e:
            print(f"[ERROR] Failed to generate comment for {chunk.name}: {e}")
            comment = "TODO: Failed to generate comment."
        comments.append(comment)

    # Insert comments into file
    new_lines = insert_comments_into_lines(lines, chunks, comments, language)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}.commented{ext}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[DONE] Wrote commented file to: {output_path}")


if __name__ == "__main__":
    main()
