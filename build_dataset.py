#!/usr/bin/env python3
"""
build_dataset.py

Extract code-comment pairs from Unity/Unreal source files.

Usage example:

    python build_dataset.py \
        --input-dir data/raw_unity \
        --language unity \
        --output data/datasets/unity_dataset.jsonl
"""

import os
import json
import argparse
from typing import List, Optional

# We reuse the parser helpers from code_commenter.py
from code_commenter import find_function_chunks, detect_language_from_extension, CodeChunk


def is_line_comment(line: str) -> bool:
    """
    Check if a line is a single-line comment (// or ///).
    """
    stripped = line.strip()
    return stripped.startswith("//") or stripped.startswith("///")


def strip_line_comment_prefix(line: str) -> str:
    """
    Remove // or /// prefix from a comment line and trim whitespace.
    """
    stripped = line.lstrip()
    if stripped.startswith("///"):
        return stripped[3:].lstrip()
    if stripped.startswith("//"):
        return stripped[2:].lstrip()
    return stripped


def is_block_comment_start(line: str) -> bool:
    return "/*" in line


def is_block_comment_end(line: str) -> bool:
    return "*/" in line


def extract_preceding_comment_block(lines: List[str], start_line: int) -> str:
    """
    Look above start_line to find a contiguous block of comments.

    Supports:
        // single line comments
        /// XML-doc style comments
        /* block comments */

    Returns a cleaned multi-line comment string, or "" if none found.
    """
    if start_line <= 0:
        return ""

    comments_block: List[str] = []
    i = start_line - 1

    # First, handle line comments (// or ///) directly stacked above the function.
    while i >= 0 and (is_line_comment(lines[i]) or lines[i].strip() == ""):
        if is_line_comment(lines[i]):
            comments_block.append(strip_line_comment_prefix(lines[i]))
        i -= 1

    if comments_block:
        comments_block.reverse()  # we collected them bottom-up
        # Remove leading/trailing empty lines in the block
        while comments_block and comments_block[0].strip() == "":
            comments_block.pop(0)
        while comments_block and comments_block[-1].strip() == "":
            comments_block.pop()
        return "\n".join(comments_block)

    # If no line comments found, check for a block comment immediately above.
    # Look for /* ... */ block above start_line.
    i = start_line - 1
    if i < 0:
        return ""

    # Skip blank lines
    while i >= 0 and lines[i].strip() == "":
        i -= 1

    if i < 0:
        return ""

    # If the line doesn't contain */ and we never find /* above, bail out.
    if "*/" not in lines[i]:
        return ""

    end_idx = i
    start_idx = i

    # Search upwards for the start of the block comment
    while start_idx >= 0 and "/*" not in lines[start_idx]:
        start_idx -= 1

    if start_idx < 0:
        return ""

    # Extract the block
    block_lines = lines[start_idx:end_idx + 1]

    cleaned_block: List[str] = []
    for line in block_lines:
        text = line.strip()
        # Remove /* and */
        text = text.replace("/*", "").replace("*/", "")
        if text.startswith("*"):
            text = text[1:].lstrip()
        cleaned_block.append(text)

    # Clean leading/trailing empty lines
    while cleaned_block and cleaned_block[0].strip() == "":
        cleaned_block.pop(0)
    while cleaned_block and cleaned_block[-1].strip() == "":
        cleaned_block.pop()

    return "\n".join(cleaned_block)


def iter_source_files(root_dir: str, language_hint: Optional[str] = None):
    """
    Yield absolute file paths for source files under root_dir.
    If language_hint == 'unity', only .cs
    If language_hint == 'unreal', only .cpp/.h
    Otherwise, filter by extension.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            full_path = os.path.join(dirpath, fname)

            if language_hint == "unity":
                if ext == ".cs":
                    yield full_path
            elif language_hint == "unreal":
                if ext in (".cpp", ".cc", ".cxx", ".h", ".hpp"):
                    yield full_path
            else:
                # fallback: accept both
                if ext in (".cs", ".cpp", ".cc", ".cxx", ".h", ".hpp"):
                    yield full_path


def build_dataset(
    input_dir: str,
    language: Optional[str],
    output_path: str,
    min_comment_length: int = 1,
) -> None:
    """
    Walk the input_dir, extract (code, comment) pairs per function,
    and write them as JSONL to output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_files = 0
    total_pairs = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for file_path in iter_source_files(input_dir, language_hint=language):
            total_files += 1
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Auto-detect if language not explicitly given
            lang = language or detect_language_from_extension(file_path)
            chunks: List[CodeChunk] = find_function_chunks(lines)

            if not chunks:
                continue

            for chunk in chunks:
                comment_text = extract_preceding_comment_block(lines, chunk.start_line)
                if not comment_text:
                    continue

                # basic filter: skip tiny comments like "TODO"
                if len(comment_text.strip()) < min_comment_length:
                    continue

                record = {
                    "file": os.path.relpath(file_path, start=input_dir),
                    "language": lang,
                    "function_name": chunk.name,
                    "start_line": chunk.start_line + 1,
                    "end_line": chunk.end_line + 1,
                    "comment": comment_text,
                    "code": chunk.code,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"[DONE] Processed {total_files} files.")
    print(f"[DONE] Extracted {total_pairs} code-comment pairs.")
    print(f"[OUT] Dataset written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build code-comment dataset from Unity/Unreal repos.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing raw source files (.cs, .cpp, .h, etc.)",
    )
    parser.add_argument(
        "--language",
        choices=["unity", "unreal"],
        default=None,
        help="Optional language hint: 'unity' (C#) or 'unreal' (C++).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to JSONL output file, e.g. data/datasets/unity_dataset.jsonl",
    )
    parser.add_argument(
        "--min-comment-length",
        type=int,
        default=4,
        help="Minimum length of comment text to keep (in characters).",
    )

    args = parser.parse_args()

    build_dataset(
        input_dir=args.input_dir,
        language=args.language,
        output_path=args.output,
        min_comment_length=args.min_comment_length,
    )


if __name__ == "__main__":
    main()

