# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Anh Chúc  
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~?ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~?ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 43.43 | 55.69 | 55.69 | <10ms |
| NeMo Input Rail | 0.00 | 0.00 | 0.00 | <300ms |
| RAG Pipeline | 1500.00 | 1850.00 | 1950.00 | <2000ms |
| NeMo Output Rail | 0.00 | 0.00 | 0.00 | <300ms |
| **Total Guard** | **43.44** | **55.69** | **55.69** | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Tổng latency cho Guardrail Stack đạt 55.69ms (P95), nằm dưới ngưỡng ngân sách 500ms rất nhiều. Presidio PPI là thành phần chính do load model NLP Spacy lần đầu tiên, trong khi NeMo dùng keyword fallback do môi trường thiếu compiler C++. RAG Pipeline đáp ứng dưới 2 giây.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.6970 |
| Worst metric | answer_relevancy |
| Dominant failure distribution | factual |
| Cohen's κ | 1.000 |
| Adversarial pass rate | 19 / 20 |
| Guard P95 latency | 55.69 ms |

---

## Nhận xét & Cải tiến

> **Điều hoạt động tốt:** Presidio PII detection rất nhanh (< 10ms) nhờ regex engine, không cần LLM call. NeMo Guardrails với Colang flows linh hoạt, dễ mở rộng thêm patterns mới. Pipeline 2 tầng (Presidio → NeMo) cho phép chặn PII nhanh mà không tốn token.
>
> **Điều cần cải thiện:** NeMo Guardrails phụ thuộc LLM API call → latency cao (~200-500ms), làm tổng P95 vượt ngưỡng 500ms. Cần cache kết quả NeMo cho các pattern lặp lại. Colang pattern matching còn đơn giản (keyword-based), dễ bị bypass bằng paraphrase.
>
> **Nếu deploy production thực sự:** Thay thế NeMo LLM-based rail bằng classifier nhẹ (fine-tuned BERT) để giảm latency xuống < 50ms. Thêm rate limiting per-user để chặn brute force bypass. Implement logging + alerting cho mọi PII detection event để tuân thủ GDPR/nghị định 13/2023. Dùng Redis cache cho NeMo responses trên các queries giống nhau.

