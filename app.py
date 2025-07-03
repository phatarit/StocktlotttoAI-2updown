import streamlit as st
from collections import defaultdict
import numpy as np
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="StockLottoAI 5 หลัก", page_icon="🎯", layout="centered")
st.title("🎯 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก + Improved AI")

st.markdown(
    """
    วางผลหวยในรูปแบบ `สามตัวบน วรรค สองตัวล่าง` บรรทัดละ 1 ชุด เช่น
    `123 45`, `567 89`, `098 76`
    """
)

# ─────── INPUT ───────
raw = st.text_area(
    "📥 วางผลหวย 5 หลัก (3+2)", height=200,
    placeholder="123 45\n567 89\n098 76 ..."
)
draws = []
for line in raw.splitlines():
    parts = line.strip().split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        if len(parts[0]) == 3 and len(parts[1]) == 2:
            draws.append(parts[0] + parts[1])
n_draw = len(draws)
st.write(f"📊 โหลดข้อมูล **{n_draw}** งวด (รูปแบบ 3+2)")

# ─────── Weighted helpers ───────
def weighted_count(items, weights):
    cnt = defaultdict(float)
    for it, w in zip(items, weights):
        cnt[it] += w
    return cnt

# ─────── คำนวณเลขเด่น 1 ตัว ───────
def get_hot_digit(nums, window=20, decay=0.8):
    last = nums[-window:]
    weights = [decay**i for i in range(len(last)-1, -1, -1)]
    # สร้างรายการตัวเลขแต่ละหลัก พร้อมน้ำหนัก
    items, ws = [], []
    for draw, w in zip(last, weights):
        for ch in draw:
            items.append(ch)
            ws.append(w)
    cnt = weighted_count(items, ws)
    # คืนเลขที่มีน้ำหนักสูงสุด
    hot = max(cnt.items(), key=lambda x: x[1])[0]
    return hot

# ─────── ทำนายสองตัวบน & ล่าง 4 ชุด ───────
def predict_weighted_pairs(nums, window=20, decay=0.8, topk=4):
    last = nums[-window:]
    weights = [decay**i for i in range(len(last)-1, -1, -1)]
    pairs = [d[3:] for d in last]
    cnt = weighted_count(pairs, weights)
    sorted_pairs = [p for p, _ in sorted(cnt.items(), key=lambda x: -x[1])]
    return sorted_pairs[:topk]

# ─────── ทำนายสามตัวบน 2 ชุด ───────
def predict_weighted_triplets(nums, window=20, decay=0.8, topk=2):
    last = nums[-window:]
    weights = [decay**i for i in range(len(last)-1, -1, -1)]
    tris = [d[:3] for d in last]
    cnt = weighted_count(tris, weights)
    sorted_tris = [t for t, _ in sorted(cnt.items(), key=lambda x: -x[1])]
    # กรองไม่ให้ซ้ำจากประวัติมากเกินไป
    from collections import Counter
    hist = Counter([d[:3] for d in nums])
    filtered = [t for t in sorted_tris if hist.get(t, 0) < 2]
    # ถ้าไม่พอ ให้ใช้ค่าแรก ๆ
    return filtered[:topk] if len(filtered) >= topk else sorted_tris[:topk]

# ─────── แสดงผล ───────
if n_draw >= 20:
    st.subheader("🔍 Improved Prediction (20 งวดล่าสุด)")
    hot_digit = get_hot_digit(draws)
    two_sets = predict_weighted_pairs(draws)
    three_sets = predict_weighted_triplets(draws)
    st.write(f"**เลขเด่น 1 ตัว:** <span style='color:red; font-size:2em'>{hot_digit}</span>", unsafe_allow_html=True)
    st.write("**สองตัวบน & สองตัวล่าง (4 ชุด):**", ', '.join(two_sets))
    st.write("**สามตัวบน (2 ชุด):**", ', '.join(three_sets))
else:
    st.info("ต้องมีข้อมูลอย่างน้อย 20 งวด สำหรับการวิเคราะห์ Improved Prediction")

# ─────── (Optional) ML ทำนายหลักสุดท้าย ───────
#-- สามารถใช้ฟังก์ชัน ML เดิมจากหน้า ML section ได้เพื่อทำนายหลักสุดท้าย --

st.caption("© 2025 StockLottoAI - Improved AI Prediction")
