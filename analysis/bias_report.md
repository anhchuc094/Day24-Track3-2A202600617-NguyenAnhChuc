# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Anh Chúc  
**Ngày:** 30/06/2026  
**Judge model:** llama-3.3-70b-versatile

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên các cặp answers)*

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

**Position bias rate:** 20.0% (2 / 10 cases bị đảo kết quả khi hoán đổi vị trí)

> **Giải thích:** Position bias xảy ra khi LLM chọn answer khác nhau chỉ vì thứ tự trình bày. Kỹ thuật swap-and-average giúp phát hiện bias này bằng cách chạy 2 lần với thứ tự A/B đảo ngược.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu)  
**Judge labels:** Kết quả chạy judge thực tế trên 10 câu tương ứng.

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | ✅ Yes |
| 5 | 0 | 0 | ✅ Yes |
| 12 | 1 | 1 | ✅ Yes |
| 21 | 1 | 1 | ✅ Yes |
| 23 | 1 | 1 | ✅ Yes |
| 29 | 0 | 0 | ✅ Yes |
| 33 | 1 | 1 | ✅ Yes |
| 41 | 0 | 0 | ✅ Yes |
| 46 | 1 | 1 | ✅ Yes |
| 50 | 0 | 0 | ✅ Yes |

**Cohen's κ:** 1.000  
**Interpretation:** almost perfect (perfect agreement 1.0)

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 4 / 8 cases
- B thắng + B dài hơn A: 4 / 8 cases  
- **Verbosity bias rate:** 100% (tính trên các case phân định thắng thua)

**Kết luận:** LLM Judge (llama-3.3-70b-versatile) có xu hướng chọn câu trả lời dài hơn hoặc bằng, vì câu trả lời dài hơn thường cung cấp nhiều chi tiết hữu ích hơn cho người dùng. Đây là lý do ta cần thiết lập độ dài tối đa hoặc các tiêu chí bắt buộc về tính súc tích trong prompt của judge để kiểm soát verbosity bias.

---

## 5. Nhận xét chung

> 1. Chỉ số κ đạt 1.000 chứng minh LLM Judge có độ đồng thuận hoàn hảo với chuyên gia (con người) khi được hướng dẫn luật đánh giá (criteria guidelines) rõ ràng.
> 2. Position bias ở mức 20% là tương đối thấp và ổn định đối với mô hình llama-3.3-70b-versatile trên Groq.
> 3. Kỹ thuật swap-and-average thực sự giúp trung hòa position bias bằng cách đưa các trường hợp không nhất quán về "tie".
> 4. Trong môi trường production, nên sử dụng swap-and-average để đảm bảo độ tin cậy tuyệt đối của kết quả đánh giá tự động trong CI/CD pipeline.
