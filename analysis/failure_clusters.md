# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Anh Chúc  
**Ngày:** 30/06/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.9015 | 0.6171 | 0.7393 |
| answer_relevancy | 0.7549 | 0.4708 | 0.4490 |
| context_precision | 0.7690 | 0.6815 | 0.8117 |
| context_recall | 0.9044 | 0.6362 | 0.4689 |
| **avg_score** | **0.8325** | **0.6014** | **0.6172** |

> **Nhận xét thực tế:** Adversarial avg_score (0.6172) thấp hơn đáng kể so với factual (0.8325) đúng như dự kiến. Điều này là do các negation traps và version conflicts gây nhầm lẫn lớn cho pipeline.

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới... | 0.2857 | answer_relevancy |
| 2 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp... | 0.3238 | answer_relevancy |
| 3 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN)... | 0.3694 | answer_relevancy |
| 4 | multi_hop | Nhân viên tạm ứng 4 triệu và một nhân viên khác tạm ứng... | 0.4118 | answer_relevancy |
| 5 | multi_hop | Nhân viên có thâm niên 7 năm theo v2024 được nghỉ... | 0.4158 | answer_relevancy |
| 6 | multi_hop | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày... | 0.4354 | answer_relevancy |
| 7 | multi_hop | Nhân viên vừa kết hôn và cùng tuần đó có con kết hôn... | 0.4405 | answer_relevancy |
| 8 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI... | 0.4692 | answer_relevancy |
| 9 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.4736 | context_recall |
| 10 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.5474 | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 1 | 0 | 1 |
| answer_relevancy | 10 | 13 | 5 | 28 |
| context_precision | 7 | 2 | 1 | 10 |
| context_recall | 3 | 4 | 4 | 11 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual (theo số lượng failure tuyệt đối là 20 câu, tuy nhiên multi_hop và adversarial có average score thấp hơn rất nhiều).  
**Dominant metric:** answer_relevancy

**Lý do phân tích:**
> Metric answer_relevancy là điểm yếu chủ đạo của toàn bộ hệ thống (chiếm 28/50 trường hợp tệ nhất). Lý do chính là vì câu trả lời của mô hình sinh ra mặc dù chứa thông tin chính xác từ văn bản nhưng thường bị dài dòng hoặc lặp lại câu hỏi quá nhiều khiến tính relevancy bị phạt điểm nặng. Đối với HR policy tiếng Việt, việc dùng các từ ngữ đặc thù và các cấu trúc câu phức tạp dễ làm giảm điểm relevancy khi đánh giá bằng mô hình embedding của RAGAS.

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

> Bộ câu hỏi adversarial có avg_score (0.6172) thấp hơn factual (0.8325). Đặc biệt, câu hỏi về VPN cá nhân (Q50) và bảo hiểm sức khỏe cho nhân viên thử việc (Q48) rơi vào bottom 10 do model bị bẫy bởi các chính sách cũ v2023 hoặc các giả định sai. Điều này chứng minh pipeline RAG hiện tại cần một cơ chế lọc và sắp xếp mức độ ưu tiên của tài liệu theo thời gian (Metadata Filtering theo phiên bản mới nhất v2024).

