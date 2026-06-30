from __future__ import annotations

"""Module 5: Enrichment pipeline for chunks before embedding."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_client import chat_completion, llm_available

LLM_AVAILABLE = llm_available()


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _local_summary(text: str) -> str:
    sentences = _sentences(text)
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


def _openai_json(messages: list[dict], max_tokens: int = 400) -> dict:
    if not LLM_AVAILABLE:
        return {}
    try:
        content = chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=0.1,
            response_format={"type": "json_object"},
        ).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
        return json.loads(content)
    except Exception as exc:
        print(f"  Enrichment API failed: {exc}")
        return {}


def summarize_chunk(text: str) -> str:
    """Create a short summary for a chunk."""
    if LLM_AVAILABLE:
        try:
            return chat_completion(
                [
                    {"role": "system", "content": "Tom tat doan van sau bang tieng Viet trong 2 cau ngan."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.1,
            ).strip()
        except Exception as exc:
            print(f"  LLM summarize failed: {exc}")

    return _local_summary(text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that the chunk can answer."""
    if LLM_AVAILABLE:
        try:
            content = chat_completion(
                [
                    {
                        "role": "system",
                        "content": f"Tao {n_questions} cau hoi ma doan van co the tra loi. Moi cau mot dong.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            lines = content.strip().splitlines()
            return [line.strip().lstrip("0123456789.-) ") for line in lines if line.strip()][:n_questions]
        except Exception as exc:
            print(f"  LLM HyQA failed: {exc}")

    sentences = _sentences(text)
    questions = []
    for sentence in sentences[:n_questions]:
        clean = sentence.rstrip(".!?")
        if any(token in clean.lower() for token in ["ngay", "ngày", "bao", "may", "mấy", "so", "số"]):
            questions.append(f"{clean} bao nhieu?")
        else:
            questions.append(f"{clean}?")
    return questions


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a concise context line while preserving the original chunk."""
    if LLM_AVAILABLE:
        try:
            context = chat_completion(
                [
                    {
                        "role": "system",
                        "content": "Viet 1 cau ngan mo ta doan nay nam trong tai lieu nao va noi ve chu de gi.",
                    },
                    {"role": "user", "content": f"Tai lieu: {document_title}\n\nDoan van:\n{text}"},
                ],
                max_tokens=80,
                temperature=0.1,
            ).strip()
            return f"{context}\n\n{text}"
        except Exception as exc:
            print(f"  LLM contextual failed: {exc}")

    prefix = f"Trich tu {document_title}. " if document_title else "Ngu canh tai lieu. "
    return f"{prefix}{text}"


def _extract_metadata_local(text: str) -> dict:
    lowered = text.lower()
    if any(word in lowered for word in ["mat khau", "mật khẩu", "vpn", "mfa"]):
        category = "it"
    elif any(word in lowered for word in ["luong", "lương", "thuong", "thưởng", "tai chinh"]):
        category = "finance"
    elif any(word in lowered for word in ["nghi", "nghỉ", "nhan vien", "nhân viên", "phep", "phép"]):
        category = "hr"
    else:
        category = "policy"

    entities = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", text)))[:5]
    topic = _local_summary(text)[:120]
    return {"topic": topic, "entities": entities, "category": category, "language": "vi"}


def extract_metadata(text: str) -> dict:
    """Extract simple metadata for filtering and diagnostics."""
    if LLM_AVAILABLE:
        data = _openai_json(
            [
                {
                    "role": "system",
                    "content": (
                        'Tra ve JSON: {"topic": "...", "entities": ["..."], '
                        '"category": "policy|hr|it|finance", "language": "vi|en"}'
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
        )
        if data:
            return data

    return _extract_metadata_local(text)


def _fallback_combined(text: str, source: str) -> dict:
    sentences = _sentences(text)
    summary = _local_summary(text)
    questions = []
    for sentence in sentences[:3]:
        clean = sentence.rstrip(".!?")
        questions.append(f"{clean}?")
    return {
        "summary": summary,
        "questions": questions,
        "context": f"Trich tu {source}." if source else "Ngu canh tai lieu noi bo.",
        "metadata": _extract_metadata_local(text),
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Single-call enrichment for summary, HyQA, context, and metadata."""
    if LLM_AVAILABLE:
        data = _openai_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Phan tich doan van va tra ve JSON voi cac key: "
                        "summary, questions, context, metadata. Metadata gom topic, "
                        "entities, category, language."
                    ),
                },
                {"role": "user", "content": f"Tai lieu: {source}\n\nDoan van:\n{text}"},
            ],
            max_tokens=400,
        )
        if data:
            return data
    return _fallback_combined(text, source)


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Run enrichment over chunks."""
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print(summarize_chunk(sample))
    print(generate_hypothesis_questions(sample))
    print(contextual_prepend(sample, "policy.md"))
    print(extract_metadata(sample))
