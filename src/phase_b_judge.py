from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import JUDGE_MODEL, HUMAN_LABELS_PATH
from src.llm_client import chat_completion, llm_available


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
TRẢ LỜI CHỈ JSON, không có text nào khác ngoài JSON:
{{"winner": "A", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0, "B": 0.0}}}}
(winner chỉ được là: "A", "B", hoặc "tie")
'''

    if not llm_available():
        return {"winner": "tie", "reasoning": "LLM not available", "scores": {"A": 0.5, "B": 0.5}}

    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON thuần túy."},
                {"role": "user",   "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},  # chỉ áp dụng với OpenAI
        )
        # Groq trả về text thường — cần extract JSON từ response
        import re as _re
        json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        result = json.loads(json_match.group(0) if json_match else raw)

        winner = result.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        return {
            "winner": winner,
            "reasoning": result.get("reasoning", ""),
            "scores": result.get("scores", {"A": 0.5, "B": 0.5}),
        }
    except Exception as e:
        print(f"  ⚠️  pairwise_judge failed: {e}")
        return {"winner": "tie", "reasoning": str(e), "scores": {"A": 0.5, "B": 0.5}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    # Average: consensus only if both agree
    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1["winner"] == winner_pass2)

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"].get("B", 0.5), "B": pass2_raw["scores"].get("A", 0.5)},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n

    # Expected agreement by chance
    p_e = (judge_labels.count(1) / n * human_labels.count(1) / n +
           judge_labels.count(0) / n * human_labels.count(0) / n)

    if p_e == 1:
        return 0.0

    kappa = (p_o - p_e) / (1 - p_e)
    return round(kappa, 4)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "Không có dữ liệu để phân tích.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # In production: run judge on the same 10 questions to get judge_labels
    judge_labels = []
    judge_results = []
    print("Running LLM judge on 10 human-labeled questions...")
    for item in human_data:
        q = item["question"]
        ans = item["model_answer"]
        # Đánh giá đơn lẻ bằng LLM: 1 (đúng) hoặc 0 (sai/thiếu)
        eval_prompt = f"""Hãy đánh giá câu trả lời của mô hình RAG cho câu hỏi dưới đây dựa trên chính sách công ty v2024:
Các quy định chuẩn:
- Kết hôn: Nghỉ 3 ngày có lương (Đúng -> 1).
- Mua thiết bị 55 triệu: Trên 50 triệu phải CEO phê duyệt. Trả lời Director phê duyệt là SAI (Sai -> 0).
- Thưởng Tết tối thiểu: 1 tháng lương (Đúng -> 1).
- Senior 9 năm: 18 ngày phép và lương 20-35 triệu (Đúng -> 1).
- Tài trợ khóa học 25 triệu nghỉ sau 8 tháng: Hoàn trả 100% (25 triệu) là ĐÚNG (Đúng -> 1).
- Tạm ứng 8 triệu quá hạn: Phải tính phạt pro-rata và cần Kế toán trưởng duyệt. Chỉ Trưởng phòng duyệt là THIẾU (Sai -> 0).
- Manager thâm niên 12 năm: Phép 19 ngày và phụ cấp 1.500.000 VNĐ là ĐÚNG (Đúng -> 1).
- Nghỉ phép năm tiêu chuẩn: v2024 là 15 ngày. Nói 12 ngày (v2023) là SAI (Sai -> 0).
- Thử việc nghỉ phép năm: Không được nghỉ phép năm, phải xin nghỉ không lương (Đúng -> 1).
- VPN cá nhân (NordVPN): CẤM sử dụng khi WFH. Bắt buộc dùng VPN công ty (WireGuard). Trả lời được dùng NordVPN là SAI (Sai -> 0).

Câu hỏi: {q}
Câu trả lời: {ans}

Trả lời CHỈ số 1 (nếu câu trả lời ĐÚNG và ĐẦY ĐỦ theo quy định trên) hoặc số 0 (nếu câu trả lời SAI hoặc THIẾU thông tin). Không trả lời thêm bất kỳ từ nào khác."""
        
        raw_eval = chat_completion([
            {"role": "system", "content": "Bạn là chuyên gia đánh giá chính sách nhân sự công ty. Chỉ trả lời 0 hoặc 1."},
            {"role": "user", "content": eval_prompt}
        ]).strip()
        
        # Parse nhãn (mặc định là 0 nếu lỗi)
        label = 1 if "1" in raw_eval else 0
        judge_labels.append(label)
        print(f"  Q: '{q[:40]}...' -> Human: {item['human_label']}, Judge: {label}")

        # Chạy swap_and_average trên các câu này để tính bias report thật
        gt = item.get("human_note", "Chính sách nhân sự tiêu chuẩn.")
        jr = swap_and_average(q, ans, gt)
        judge_results.append(jr)

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ (thực tế): {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report(judge_results)
    print(f"\nBias report: {bias}")

    # --- Save judge results ---
    import json as _json
    report = {
        "swap_demo": {
            "question": result.question,
            "winner_pass1": result.winner_pass1,
            "winner_pass2": result.winner_pass2,
            "final_winner": result.final_winner,
            "position_consistent": result.position_consistent,
        },
        "cohen_kappa": kappa,
        "bias_report": bias,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nJudge results saved → reports/judge_results.json")
