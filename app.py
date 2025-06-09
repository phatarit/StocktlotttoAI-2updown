
import streamlit as st, pandas as pd, math
from collections import Counter, defaultdict
from itertools import combinations

# ───────────────── CONFIG ─────────────────
st.set_page_config(page_title="LoasLottoAI", page_icon="🇱🇦", layout="centered")
st.title("🎯 LoasLottoAI")

MIN_DRAW = 30
WINDOW_PAIR = 60
PAIR_KEEP = 10
ALPHA_GRID = [0.90, 0.92, 0.94, 0.96, 0.98]

# ───────────────── INPUT ─────────────────
raw = st.text_area(
    "📥 วางผลหวยลาว 4 หลัก (บรรทัดละ 1 งวด)", height=220,
    placeholder="เช่น 9767\n5319\n1961 …"
)
draws = [l.strip() for l in raw.splitlines() if l.strip().isdigit() and len(l.strip())==4]
st.write(f"📊 โหลดแล้ว **{len(draws)}** งวด")
if len(draws) < MIN_DRAW:
    st.info(f"⚠️ ต้องมีข้อมูลอย่างน้อย {MIN_DRAW} งวด")
    st.stop()
st.dataframe(pd.DataFrame(draws, columns=["เลข 4 หลัก"]), use_container_width=True)

# ───────────────── ENHANCED FUNCTIONS ─────────────────
def unordered2(a, b):
    return "".join(sorted([a, b]))

def enhanced_two_combo(hist, alpha, window=WINDOW_PAIR):
    pair_freq = Counter()
    recent_pairs = hist[-window:]
    for i, num in enumerate(reversed(recent_pairs)):
        weight = alpha ** i
        for x, y in combinations(num, 2):
            pair = unordered2(x, y)
            pair_freq[pair] += weight

    # เสริมเลขพิเศษ: ซ้ำ, ข้ามงวด, หายไป
    last = hist[-1] if len(hist) >= 1 else ""
    prev = hist[-2] if len(hist) >= 2 else ""
    skip1 = hist[-3] if len(hist) >= 3 else ""
    recent_digits = "".join(hist[-5:]) if len(hist) >= 5 else ""

    specials = []
    if prev:
        specials += [unordered2(prev[i], prev[j]) for i in range(4) for j in range(i+1, 4)]
    if skip1:
        specials += [unordered2(skip1[i], skip1[j]) for i in range(4) for j in range(i+1, 4)]
    missing_digits = [d for d in '0123456789' if d not in recent_digits]
    specials += [unordered2(a, b) for a in missing_digits for b in missing_digits if a != b]

    for s in specials:
        pair_freq[s] += 1

    return [pair for pair, _ in pair_freq.most_common(PAIR_KEEP)]

# ───────────────── CALC & DISPLAY ─────────────────
a = max(ALPHA_GRID, key=lambda alpha: sum(
    1 for i in range(MIN_DRAW, len(draws))
    if any(pair[0] in draws[i] and pair[1] in draws[i] for pair in enhanced_two_combo(draws[:i], alpha))
))

combo_two = enhanced_two_combo(draws, a)

st.subheader("🔮 เจาะสองตัวบน-ล่าง ปรับสูตรเพิ่มเลขเด่น")
st.markdown("  ".join(combo_two), unsafe_allow_html=True)
st.caption("© 2025 LoasLottoAI – Enhanced + Hot Digit Weighted Prediction")
