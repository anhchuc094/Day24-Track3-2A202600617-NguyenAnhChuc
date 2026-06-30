from __future__ import annotations

"""Module 1: Advanced Chunking Strategies."""

import glob
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE, SEMANTIC_THRESHOLD


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract the PDF text layer. Scanned PDFs return an empty string."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown files and text-layer PDFs from data/."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  Skip {os.path.basename(fp)}: scanned PDF has no text layer.")

    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Baseline paragraph chunking."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _word_vector(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(t, 0) for t, v in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b + 1e-9)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group neighboring sentences when their lexical similarity stays above threshold."""
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n+", text) if s.strip()]
    if not sentences:
        return []

    vectors = [_word_vector(sentence) for sentence in sentences]
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if _cosine(vectors[i - 1], vectors[i]) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": i},
        )
        for i, group in enumerate(groups)
        if " ".join(group).strip()
    ]


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Build parent chunks for context and smaller child chunks for retrieval."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    parents: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush_parent() -> None:
        if not current:
            return
        pid = f"{metadata.get('source', 'doc')}_parent_{len(parents)}"
        parents.append(
            Chunk(
                text="\n\n".join(current).strip(),
                metadata={
                    **metadata,
                    "chunk_type": "parent",
                    "parent_id": pid,
                    "chunk_index": len(parents),
                },
            )
        )

    for paragraph in paragraphs:
        projected = current_len + len(paragraph) + (2 if current else 0)
        if current and projected > parent_size:
            flush_parent()
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)
    flush_parent()

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parts = [p.strip() for p in re.split(r"\n\s*\n+", parent.text) if p.strip()]
        buffer: list[str] = []
        buffer_len = 0

        def flush_child() -> None:
            if not buffer:
                return
            children.append(
                Chunk(
                    text="\n\n".join(buffer).strip(),
                    metadata={**metadata, "chunk_type": "child", "chunk_index": len(children)},
                    parent_id=pid,
                )
            )

        for part in parts:
            subparts = [part]
            if len(part) > child_size:
                subparts = [part[i : i + child_size].strip() for i in range(0, len(part), child_size)]
            for subpart in subparts:
                projected = buffer_len + len(subpart) + (2 if buffer else 0)
                if buffer and projected > child_size:
                    flush_child()
                    buffer = []
                    buffer_len = 0
                buffer.append(subpart)
                buffer_len += len(subpart) + (2 if buffer_len else 0)
        flush_child()

    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Chunk markdown by H1-H3 sections while preserving headers."""
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = ""
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body and not current_header:
            return
        section = current_header.lstrip("#").strip() if current_header else "Document"
        chunk_text = f"{current_header}\n\n{body}".strip() if current_header else body
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
            )
        )

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+.+$", line):
            flush()
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not chunks and text.strip():
        chunks.append(
            Chunk(
                text=text.strip(),
                metadata={**metadata, "section": "Document", "strategy": "structure", "chunk_index": 0},
            )
        )
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies on documents and compare simple length stats."""
    def stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}
    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": stats(basic),
        "semantic": stats(semantic),
        "hierarchical": {**stats(children), "parents": len(parents)},
        "structure": stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    for name, strategy_stats in compare_strategies(docs).items():
        print(f"  {name}: {strategy_stats}")
