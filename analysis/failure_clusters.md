# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Anh Chúc  
**Ngày:** 30/06/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

*(Bảng này được điền sau khi chạy `python src/phase_a_ragas.py` — xem `reports/ragas_50q.json`)*

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | ? | ? | ? |
| answer_relevancy | ? | ? | ? |
| context_precision | ? | ? | ? |
| context_recall | ? | ? | ? |
| **avg_score** | ? | ? | ? |

> **Nhận xét kỳ vọng:** Adversarial avg_score thường thấp hơn factual do pipeline khó xử lý các câu hỏi về version conflicts (v2023 vs v2024) và negation traps trong corpus HR policy.

---

## 2. Bottom 10 Questions

*(Xem chi tiết trong `reports/ragas_50q.json` → key `bottom_10`)*

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| 9 | | | | |
| 10 | | | | |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | | | | |
| answer_relevancy | | | | |
| context_precision | | | | |
| context_recall | | | | |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** multi_hop (dự kiến)  
**Dominant metric:** context_recall (dự kiến)

**Lý do phân tích:**

> Các câu hỏi `multi_hop` yêu cầu kết hợp thông tin từ nhiều tài liệu (cross-doc reasoning), nên pipeline hay bỏ sót chunks liên quan → context_recall thấp. Với corpus HR policy tiếng Việt có nhiều policy versions (v2023/v2024), chunking hierarchical có thể không đủ để giữ ngữ cảnh toàn document. Pipeline cần cải thiện BM25 hybrid search để bắt được các policy cross-reference. Adversarial distribution dùng negation traps và version conflicts mà LLM có xu hướng xử lý sai vì context không đủ rõ ràng về phiên bản áp dụng.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating — sinh thêm thông tin không có trong context | Tighten system prompt, lower temperature, thêm citation requirement |
| context_recall | Missing relevant chunks — chunking bỏ sót tài liệu liên quan | Improve chunking overlap, add BM25 keyword search, expand hybrid search |
| context_precision | Too many irrelevant chunks — reranker không lọc tốt | Tăng CrossEncoder reranker strictness, thêm metadata filter theo policy version |
| answer_relevancy | Answer doesn't match question — LLM trả lời lạc đề | Improve prompt template, thêm query rewriting step, kiểm tra intent |

---

## 6. Nhận xét về Adversarial Distribution

> Bộ câu hỏi adversarial được thiết kế để bẫy pipeline bằng 3 loại khó khăn: (1) version conflicts giữa policy v2023 và v2024 — pipeline cần biết ưu tiên phiên bản mới nhất, (2) negation traps như "có nên tự xử lý không?" — LLM dễ bị confuse bởi phủ định trong tiếng Việt, (3) câu hỏi về VPN cá nhân và các policy ngoài phạm vi HR. Kỳ vọng adversarial avg_score < factual avg_score là hợp lý vì corpus không có index rõ ràng theo version. Nếu đạt bonus, đây là bằng chứng pipeline đã có khả năng phân biệt version conflicts tương đối tốt.
