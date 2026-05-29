import hashlib
import html
import re
from typing import Any

import numpy as np


def snapshot_embedding(emb):
    if emb is None:
        return None
    try:
        if len(emb) == 0:
            return None
    except TypeError:
        return None
    return np.asarray(emb, dtype=np.float32).copy()


def materialize_embedding(stored):
    if stored is None:
        return None
    return stored.tolist()


def stable_text_key(text) -> str:
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def chunk_text(text: str, chunk_level: int = 2, is_pdf_content: bool = False) -> list[str]:
    if chunk_level <= 0:
        return [text]

    if chunk_level == 1:
        paragraphs = text.split("\n")
        chunks = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            paragraph_phrases = re.split(r"(?<=[,;:])\s+", paragraph)
            for phrase in paragraph_phrases:
                if phrase.strip():
                    chunks.append(phrase.strip())
        return chunks

    if chunk_level == 2:
        if is_pdf_content:
            chunks = []
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                if sentence.strip():
                    chunks.append(sentence.strip())
        else:
            paragraphs = text.split("\n")
            chunks = []
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if sentence.strip():
                        chunks.append(sentence.strip())
        return chunks

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if chunk_level == 3:
        return paragraphs

    chunks = []
    paragraphs_per_chunk = chunk_level - 2
    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk = "\n".join(paragraphs[i : i + paragraphs_per_chunk])
        chunks.append(chunk)
    return chunks


async def clean_text_formatting(content: str) -> str:
    lines = content.split("\n")
    cleaned_lines = []

    for line in lines:
        repeated_char_pattern = re.compile(r"((.)\2{4,})")
        matches = list(repeated_char_pattern.finditer(line))

        if matches:
            for match in reversed(matches):
                char_sequence = match.group(1)
                char = match.group(2)
                if len(char_sequence) >= 5:
                    replacement = char * 2 + "(...)" + char * 2
                    start, end = match.span()
                    line = line[:start] + replacement + line[end:]

        for pattern_length in range(2, 4):
            i = 0
            while i <= len(line) - pattern_length * 5:
                pattern = line[i : i + pattern_length]
                repetition_count = 0
                for j in range(i, len(line) - pattern_length + 1, pattern_length):
                    if line[j : j + pattern_length] == pattern:
                        repetition_count += 1
                    else:
                        break
                if repetition_count >= 5:
                    replacement = pattern * 2 + "(...)" + pattern * 2
                    total_length = pattern_length * repetition_count
                    line = line[:i] + replacement + line[i + total_length :]
                i += 1

        ellipsis_pattern = re.compile(r"(\S\S\(\.\.\.\)\S\S\s+)(\1){2,}")
        ellipsis_matches = list(ellipsis_pattern.finditer(line))
        if ellipsis_matches:
            for match in reversed(ellipsis_matches):
                single_instance = match.group(1)
                start, end = match.span()
                line = line[:start] + single_instance + line[end:]

        cleaned_lines.append(line)

    lines = cleaned_lines
    merged_lines: list[str] = []
    short_line_group: list[str] = []
    mixed_case_pattern = re.compile(r"[a-z][A-Z]")

    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        word_count = len(current_line.split())

        if word_count <= 5 and current_line:
            is_numbered_item = False
            number_patterns = [
                r"^\d+[\.\)\:]",
                r"^[A-Za-z][\.\)\:]",
                r".*\d+[\.\)\:]$",
            ]
            for pattern in number_patterns:
                if re.search(pattern, current_line):
                    is_numbered_item = True
                    break

            if is_numbered_item and short_line_group:
                prev_line = short_line_group[-1]
                prev_match = re.search(r"(\d+)[\.\)\:]", prev_line)
                curr_match = re.search(r"(\d+)[\.\)\:]", current_line)
                if prev_match and curr_match:
                    try:
                        prev_number = int(prev_match.group(1))
                        curr_number = int(curr_match.group(1))
                        is_numbered_item = curr_number == prev_number + 1
                    except ValueError:
                        pass

            if is_numbered_item:
                if short_line_group:
                    merged_lines.extend(short_line_group)
                    short_line_group = []
                merged_lines.append(current_line)
            else:
                short_line_group.append(current_line)
        else:
            if short_line_group:
                if len(short_line_group) >= 5:
                    mixed_case_count = 0
                    total_lc_to_uc = 0
                    for line in short_line_group:
                        for j in range(1, len(line)):
                            if j > 0 and line[j - 1].islower() and line[j].isupper():
                                total_lc_to_uc += 1
                        if mixed_case_pattern.search(line):
                            mixed_case_count += 1
                    has_mixed_case = (
                        mixed_case_count >= len(short_line_group) * 0.3
                    ) or (total_lc_to_uc >= 3)

                    if merged_lines:
                        for j in range(min(2, len(short_line_group))):
                            merged_lines[-1] += f". {short_line_group[j]}"
                        if has_mixed_case:
                            merged_lines.append("(Navigation menu removed)")
                        else:
                            merged_lines.append("(Headers removed)")
                        last_idx = len(short_line_group) - 2
                        if last_idx >= 2:
                            merged_lines.append(short_line_group[last_idx])
                            merged_lines.append(short_line_group[last_idx + 1])
                    else:
                        for j in range(min(2, len(short_line_group))):
                            merged_lines.append(short_line_group[j])
                        if has_mixed_case:
                            merged_lines.append("(Navigation menu removed)")
                        else:
                            merged_lines.append("(Headers removed)")
                        last_idx = len(short_line_group) - 2
                        if last_idx >= 2:
                            merged_lines.append(short_line_group[last_idx])
                            merged_lines.append(short_line_group[last_idx + 1])
                else:
                    for j, short_line in enumerate(short_line_group):
                        if j == 0 and merged_lines:
                            merged_lines[-1] += f". {short_line}"
                        else:
                            merged_lines.append(short_line)
                short_line_group = []

            if current_line:
                merged_lines.append(current_line)
        i += 1

    if short_line_group:
        if len(short_line_group) >= 5:
            mixed_case_count = 0
            total_lc_to_uc = 0
            for line in short_line_group:
                for j in range(1, len(line)):
                    if j > 0 and line[j - 1].islower() and line[j].isupper():
                        total_lc_to_uc += 1
                if mixed_case_pattern.search(line):
                    mixed_case_count += 1
            has_mixed_case = (mixed_case_count >= len(short_line_group) * 0.3) or (
                total_lc_to_uc >= 3
            )

            if merged_lines:
                for j in range(min(2, len(short_line_group))):
                    merged_lines[-1] += f". {short_line_group[j]}"
                if has_mixed_case:
                    merged_lines.append("(Navigation menu removed)")
                else:
                    merged_lines.append("(Headers removed)")
                last_idx = len(short_line_group) - 2
                if last_idx >= 2:
                    merged_lines.append(short_line_group[last_idx])
                    merged_lines.append(short_line_group[last_idx + 1])
            else:
                for j in range(min(2, len(short_line_group))):
                    merged_lines.append(short_line_group[j])
                if has_mixed_case:
                    merged_lines.append("(Navigation menu removed)")
                else:
                    merged_lines.append("(Headers removed)")
                last_idx = len(short_line_group) - 2
                if last_idx >= 2:
                    merged_lines.append(short_line_group[last_idx])
                    merged_lines.append(short_line_group[last_idx + 1])
        else:
            for j, short_line in enumerate(short_line_group):
                if j == 0 and merged_lines:
                    merged_lines[-1] += f". {short_line}"
                else:
                    merged_lines.append(short_line)

    return "\n".join(merged_lines)


def escape_html(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def response_text(response: Any) -> str:
    """Safely extract assistant content from an OWUI chat-completion response.

    Returns "" for any malformed/None/empty response instead of raising, so a
    single bad model reply never aborts the whole research run.
    """
    try:
        choices = (response or {}).get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""
    except (AttributeError, IndexError, TypeError):
        return ""
