"""Split already-verified spoken answers without rewriting their contents."""

import re


def speech_chunks(text: str, target_chars: int = 220) -> list[str]:
    sentences = []
    start = 0
    for boundary in re.finditer(r"[.!?…](?:[\"”’])?\s+", text):
        end = boundary.end()
        prefix = text[start:boundary.start() + 1]
        if re.search(r"\b(?:TP|Tp|Mr|Mrs|Dr|ThS|TS)\.$", prefix):
            continue
        sentences.append(text[start:end].strip())
        start = end
    if text[start:].strip():
        sentences.append(text[start:].strip())
    chunks = []
    for sentence in sentences:
        # Keep the first sentence small to reduce time to first audio. Later
        # sentences are grouped to limit encoder padding and synthesis overhead.
        if len(chunks) > 1 and len(chunks[-1]) + len(sentence) + 1 <= target_chars:
            chunks[-1] += " " + sentence
        else:
            chunks.append(sentence)
    return chunks
