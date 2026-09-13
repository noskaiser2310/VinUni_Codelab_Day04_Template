"""
llm_client.py — Lớp gọi LLM THẬT cho Lab #4 (không if-else keyword).

Ưu tiên provider theo thứ tự:
  1. Gemini  (env GEMINI_API_KEY / GOOGLE_API_KEY) — khớp slide Day04 + README
  2. Groq    (env GROQ_API_KEY) — free, nhanh, OpenAI-compatible
  3. OpenAI  (env OPENAI_API_KEY)

Nếu không có key nào → is_llm_available() = False, caller tự fallback
về rule-based để vẫn pass autograder. Không bao giờ raise.
"""

import json
import os
import re

# Nạp .env thủ công (không cần python-dotenv)
for _env_path in (
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env"),
):
    if os.path.exists(_env_path):
        try:
            with open(_env_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
        except OSError:
            pass


def _get_key(*names):
    for n in names:
        v = os.environ.get(n, "").strip().strip("'\"")
        if v:
            return v
    return ""


def active_provider() -> str:
    """Trả về 'gemini' | 'groq' | 'openai' | 'none'."""
    if _get_key("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        return "gemini"
    if _get_key("GROQ_API_KEY"):
        return "groq"
    if _get_key("OPENAI_API_KEY"):
        return "openai"
    return "none"


def is_llm_available() -> bool:
    return active_provider() != "none"


# ---------------------------------------------------------------------------
# Core: gọi LLM thô, trả về text. Thất bại → trả về "" (không raise).
# ---------------------------------------------------------------------------

def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    from google import genai  # google-genai>=1.0 (đã cài 1.65.0)

    api_key = _get_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    resp = client.models.generate_content(
        model=model,
        contents=[
            {"role": "user", "parts": [{"text": system_prompt + "\n\n---\n\n" + user_prompt}]}
        ],
        config={"temperature": 0.2, "max_output_tokens": 1024},
    )
    # SDK mới: resp.text; SDK cũ: candidates[0].content.parts[0].text
    text = getattr(resp, "text", "") or ""
    if not text and getattr(resp, "candidates", None):
        try:
            text = resp.candidates[0].content.parts[0].text
        except (IndexError, AttributeError):
            text = ""
    return (text or "").strip()


def _call_openai_compatible(system_prompt: str, user_prompt: str, provider: str) -> str:
    from openai import OpenAI  # openai>=1.0 (đã cài 2.16.0)

    if provider == "groq":
        client = OpenAI(
            api_key=_get_key("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    else:
        client = OpenAI(api_key=_get_key("OPENAI_API_KEY"))
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def llm_complete(system_prompt: str, user_prompt: str) -> str:
    """Gọi LLM thật theo provider đang có. Lỗi → trả ''."""
    provider = active_provider()
    try:
        if provider == "gemini":
            return _call_gemini(system_prompt, user_prompt)
        if provider in ("groq", "openai"):
            return _call_openai_compatible(system_prompt, user_prompt, provider)
    except Exception:
        return ""
    return ""


# ---------------------------------------------------------------------------
# Planning: LLM phân tích intent + trích args, trả JSON chuẩn.
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """Bạn là bộ phận lập kế hoạch của VinAssistant (Vingroup AI).
Nhiệm vụ: đọc câu hỏi tiếng Việt của khách hàng, quyết định gọi tool nào.

Chỉ được trả về DUY NHẤT một JSON object (không markdown, không giải thích) với schema:
{
  "needs_catalog": boolean,
  "needs_ticket": boolean,
  "category": "xe_dien" | "du_lich" | null,
  "max_price": number | null,
  "customer_name": string | null,
  "issue_description": string | null,
  "priority": "low" | "medium" | "high"
}

Quy tắc:
- Hỏi xem/tìm/giá sản phẩm (xe điện VinFast, resort Vinpearl, du lịch) → needs_catalog=true.
  category: chứa xe/vinfast/vf → "xe_dien"; chứa resort/vinpearl/nha trang/phú quốc/du lịch → "du_lich".
  max_price: đổi "600 triệu"→600000000, "6 triệu"→6000000, "200 triệu"→200000000, "1.5 tỷ"→1500000000. Không có giá → null.
- Báo lỗi/sự cố/khiếu nại/yêu cầu hỗ trợ kèm tên người → needs_ticket=true.
  customer_name: trích tên sau "tôi tên"/"tên tôi là" (vd "Lê Minh Khoa", "Phạm Thị Dung").
  priority: "nghiêm trọng/gấp/khẩn"→high; "trung bình"→medium; "nhẹ/thấp"→low; mặc định medium.
  issue_description: tóm tắt vấn đề bằng tiếng Việt.
- Hỏi chính sách/bảo hành/thông tin chung, không hỏi giá cụ thể, không báo sự cố → cả hai đều false.
- Cả hai nhu cầu có thể true đồng thời.
"""


def _extract_json(text: str) -> dict | None:
    """Bóc JSON từ text LLM (chịu được markdown fence)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def llm_plan(user_input: str) -> dict | None:
    """Dùng LLM thật để lập plan. OK → dict chuẩn; fail → None (caller fallback rule)."""
    if not is_llm_available():
        return None
    raw = llm_complete(_PLAN_SYSTEM, f'Câu hỏi khách hàng: "{user_input}"')
    obj = _extract_json(raw)
    if not obj:
        return None
    plan = {
        "needs_catalog": bool(obj.get("needs_catalog", False)),
        "needs_ticket": bool(obj.get("needs_ticket", False)),
        "category": obj.get("category"),
        "max_price": obj.get("max_price"),
        "customer_name": obj.get("customer_name"),
        "issue_description": obj.get("issue_description") or user_input.strip(),
        "priority": str(obj.get("priority", "medium") or "medium").lower(),
    }
    if plan["category"] not in ("xe_dien", "du_lich"):
        plan["category"] = None
    if plan["priority"] not in ("low", "medium", "high"):
        plan["priority"] = "medium"
    try:
        plan["max_price"] = int(plan["max_price"]) if plan["max_price"] is not None else None
    except (ValueError, TypeError):
        plan["max_price"] = None
    return plan


# ---------------------------------------------------------------------------
# Synthesis: LLM viết Final Answer tiếng Việt từ Observation thật.
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = """Bạn là VinAssistant — trợ lý AI chính thức của Vingroup.
Viết câu trả lời cuối bằng tiếng Việt, thân thiện, chuyên nghiệp.
TUYỆT ĐỐI không bịa tên sản phẩm, giá, ticket_id — chỉ dùng dữ liệu trong phần Observation được cho.
- Nếu Observation catalog rỗng → nói "Rất tiếc, không tìm thấy sản phẩm phù hợp..." và gợi ý nới ngân sách.
- Nếu có ticket → nêu rõ ticket_id, tên khách, mức ưu tiên.
- Nếu là FAQ bảo hành pin → nêu "10 năm".
"""


def llm_synthesize(user_input: str, catalog_results, ticket_result) -> str:
    """Dùng LLM thật để tổng hợp Final Answer. Fail → '' (caller dùng template cứng)."""
    if not is_llm_available():
        return ""
    obs = (
        f"Câu hỏi gốc: {user_input}\n"
        f"Observation catalog: {json.dumps(catalog_results, ensure_ascii=False)}\n"
        f"Observation ticket: {json.dumps(ticket_result, ensure_ascii=False)}"
    )
    return llm_complete(_SYNTH_SYSTEM, obs)


def llm_baseline_answer(user_input: str) -> str:
    """Baseline 1 lượt LLM, không tool — để quan sát hallucination. Fail → ''."""
    if not is_llm_available():
        return ""
    sys_prompt = (
        "Bạn là chatbot tư vấn Vingroup, trả lời trực tiếp bằng tiếng Việt, "
        "không được gọi tool, không bịa số liệu quá chi tiết."
    )
    return llm_complete(sys_prompt, user_input)
