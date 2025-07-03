import streamlit as st
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="StockLottoAI 5 หลัก", page_icon="🎯", layout="centered")
st.title("🎯 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก + AI")

# คำอธิบายวิธีป้อนข้อมูล
st.markdown(
    """
    วางผลหวยในรูปแบบ `สามตัวบน วรรค สองตัวล่าง` บรรทัดละ 1 ชุด เช่น
    `123 45`
    `567 89`
    `098 76`
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
    if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit():
        if len(parts[0])==3 and len(parts[1])==2:
            draws.append(parts[0]+parts[1])
n_draw = len(draws)
st.write(f"📊 โหลดข้อมูล **{n_draw}** งวด")

# ─────── ฟังก์ชันวิเคราะห์เลขซ้ำ ───────
def analyze_repeat_digits(nums, window=5):
    last = nums[-window:]
    # นับจำนวนครั้งแต่ละตัวเลขปรากฏ
    c = Counter(''.join(last))
    # เลือกตัวเลขที่ปรากฏอย่างน้อยในทุกงวด (หรือเกือบทุกงวด)
    repeats = [d for d, cnt in c.items() if cnt >= window-1]
    # ถ้าไม่พอ ให้ใช้ตัวเลขที่ซ้ำมากสุดสองตัว
    if len(repeats) < 2:
        repeats = [d for d, _ in c.most_common(2)]
    return repeats, c

# ─────── สร้างชุดทำนายสองตัว ───────
def predict_two_digit_sets(nums, window=5, hotk=2, prev_draws=2):
    hot_digits, freq = analyze_repeat_digits(nums, window)
    # แยกเลขเด่นสองตัว
    a, b = hot_digits[0], hot_digits[1] if len(hot_digits)>1 else (hot_digits[0], hot_digits[0])
    # ชุด cross hot_digits (2 ชุด)
    cross = [a+b, b+a]
    # ชุด double hot_digits (2 ชุด)
    doubles = [a*2, b*2]
    # ชุดผสม hot_digits กับเลขในงวดก่อนหน้า prev_draws งวด
    comb_prev = set()
    for dstr in nums[-prev_draws:]:
        for d in hot_digits:
            for x in dstr:
                comb_prev.add(d+x)
                comb_prev.add(x+d)
    comb_prev = list(comb_prev)
    # รวมเป็น 6 ชุด: cross (2), doubles (2), comb_prev (2)
    result = cross + doubles + comb_prev[:2]
    return result, hot_digits

# ─────── สร้างชุดทำนายสามตัว ───────
def predict_three_digit_sets(hot_digits):
    if len(hot_digits) >= 2:
        a, b = hot_digits[0], hot_digits[1]
        # เบิ้ล-หาม เลือกสองรูปแบบ
        return [a+a+b, b+b+a]
    elif hot_digits:
        return [hot_digits[0]*3, hot_digits[0]*3]
    else:
        return ["", ""]

# ─────── แสดงผล Logic ปกติ (5 งวด) ───────
if n_draw >= 5:
    st.subheader("📈 วิเคราะห์เชิงสถิติ ใหม่ (Cross-Draw 5 งวด)")
    two_sets, hot_digits = predict_two_digit_sets(draws, window=5)
    three_sets = predict_three_digit_sets(hot_digits)
    # แสดงเลขซ้ำบ่อยสุด
    repeats, freq = analyze_repeat_digits(draws, window=5)
    st.write("**เลขเด่น 2 ตัว:**", ', '.join(hot_digits))
    st.write("**ทำนายงวดถัดไป (สองตัวบน & ล่าง 6 ชุด):**", ', '.join(two_sets))
    st.write("**ทำนายงวดถัดไป (สามตัวบน 2 ชุด):**", ', '.join(three_sets))
else:
    st.info("ต้องมีข้อมูลอย่างน้อย 5 งวด ในรูปแบบ 3+2 สำหรับวิเคราะห์")

# ─────── ML Part (เดิม) ───────
def predict_next_digit_ml(nums, window=4):
    nums = [list(map(int, list(x))) for x in nums]
    X, y = [], []
    for i in range(len(nums)-window):
        X.append(np.array(nums[i:i+window]).flatten())
        y.append(nums[i+window][-1])
    if len(X) < 10:
        return None
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    model.fit(np.array(X), np.array(y))
    last_feat = np.array(nums[-window:]).flatten().reshape(1, -1)
    return model.predict(last_feat)[0]

if n_draw >= 25:
    st.subheader("🤖 AI (ML) ทำนายเลขหลักสุดท้ายงวดถัดไป")
    pred_digit = predict_next_digit_ml(draws, window=4)
    if pred_digit is not None:
        st.write(f"**AI ทำนายเลขหลักสุดท้าย:** <span style='font-size:2em;color:green'>{pred_digit}</span>", unsafe_allow_html=True)
    else:
        st.warning("ข้อมูลยังน้อยเกินไปสำหรับ ML ทำนาย (ต้องอย่างน้อย 25 งวด)")
else:
    st.info("ใส่ข้อมูลอย่างน้อย 25 งวด เพื่อให้ AI (ML) เรียนรู้และทำนายได้")

st.caption("© 2025 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก พร้อม AI/ML")
