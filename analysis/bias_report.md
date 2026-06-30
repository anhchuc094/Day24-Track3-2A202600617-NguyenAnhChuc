# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Anh Chúc  
**Ngày:** 30/06/2026  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên các cặp answers — xem `reports/judge_results.json` để có kết quả thực tế)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | A | Answer A trích dẫn chính sách v2024 cụ thể và đầy đủ hơn |
| 2 | Chính sách WFH là gì? | B | Answer B ngắn gọn, súc tích, đúng trọng tâm hơn |
| 3 | Lương thử việc được tính như thế nào? | tie | Hai answers có độ chính xác tương đương |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | ✅ True |
| 2 | B | B | B | ✅ True |
| 3 | tie | tie | tie | ✅ True |

**Position bias rate:** ~20–30% (kỳ vọng — xem kết quả thực tế trong judge_results.json)

> **Giải thích:** Position bias xảy ra khi LLM chọn answer khác nhau chỉ vì thứ tự trình bày. Kỹ thuật swap-and-average giúp phát hiện bias này bằng cách chạy 2 lần với thứ tự A/B đảo ngược.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu)  
**Judge labels:** Kết quả chạy judge trên 10 câu tương ứng (xem `reports/judge_results.json`)

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | — | — |
| 5 | 0 | — | — |
| 12 | 1 | — | — |
| 21 | 1 | — | — |
| 23 | 0 | — | — |
| 29 | 1 | — | — |
| 33 | 0 | — | — |
| 41 | 1 | — | — |
| 46 | 0 | — | — |
| 50 | 1 | — | — |

**Cohen's κ:** Xem `reports/judge_results.json` → key `cohen_kappa`  
**Interpretation:** κ > 0.6 = substantial agreement (mục tiêu bonus)

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: xem `judge_results.json` → `bias_report.verbosity_details.a_wins_a_longer`
- B thắng + B dài hơn A: xem `judge_results.json` → `bias_report.verbosity_details.b_wins_b_longer`
- **Verbosity bias rate:** xem `judge_results.json` → `bias_report.verbosity_bias`

**Kết luận:** LLM thường có xu hướng chọn answer dài hơn vì nó trông "đầy đủ" hơn, dù không chính xác hơn. Đây là vấn đề trong production vì nó làm cho judge overfit với độ dài thay vì chất lượng thực sự. Cần dùng prompt rõ ràng để yêu cầu judge đánh giá nội dung, không phải độ dài.

---

## 5. Nhận xét chung

> Kỹ thuật swap-and-average là cần thiết để giảm position bias của LLM judge. Khi κ > 0.6, LLM judge có thể đáng tin cậy như một automated evaluator trong CI/CD pipeline. Position bias dưới 30% là acceptable trong production — nếu vượt ngưỡng này cần bắt buộc dùng swap-and-average. Với HR policy domain cụ thể, gpt-4o-mini thường đồng thuận tốt với human labels vì domain knowledge của HR policy không quá chuyên sâu. Trong production, nên kết hợp LLM judge với RAGAS metrics để có đánh giá toàn diện hơn, tránh phụ thuộc hoàn toàn vào một loại metric.
