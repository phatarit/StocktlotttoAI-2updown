
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

# ───────────────── IMPROVED FUNCTIONS ─────────────────
def unordered2(a, b):
    return "".join(sorted([a, b]))

def improved_two_combo(hist, alpha, window=WINDOW_PAIR):
    pair_freq = Counter()
    recent_pairs = hist[-window:]
    for i, num in enumerate(reversed(recent_pairs)):
        weight = alpha ** i
        for x, y in combinations(num, 2):
            pair = unordered2(x, y)
            pair_freq[pair] += weight

    recent_numbers = [num[-2:] for num in recent_pairs]
    reversed_pairs = Counter()
    for num in recent_numbers:
        reversed_pairs[num] += 1
        reversed_pairs[num[::-1]] += 0.5

    combined_scores = Counter()
    for pair, freq in pair_freq.items():
        combined_scores[pair] = freq + reversed_pairs[pair]

    return [pair for pair, _ in combined_scores.most_common(PAIR_KEEP)]

# ───────────────── CALC & DISPLAY ─────────────────
a = max(ALPHA_GRID, key=lambda alpha: sum(1 for i in range(MIN_DRAW,len(draws))
    if any(pair[0] in draws[i] and pair[1] in draws[i] for pair in improved_two_combo(draws[:i], alpha))))

combo_two = improved_two_combo(draws, a)

st.subheader("🔮 เจาะสองตัวบน-ล่าง ปรับปรุงใหม่")
st.markdown("  ".join(combo_two), unsafe_allow_html=True)

st.caption("© 2025 LoasLottoAI – Enhanced EWMA Prediction")
