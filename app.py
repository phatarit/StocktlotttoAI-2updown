import streamlit as st
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="StockLottoAI 5 หลัก", page_icon="🎯", layout="centered")
st.title("🎯 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก + AI")
st.markdown("วางเลข 5 หลักแต่ละงวด (บรรทัดละ 1 ชุด) เช่น 56789\n12345\n09876")

# ─────── INPUT ───────
raw = st.text_area("📥 วางผลหวย 5 หลัก", height=220, placeholder="56789\n12345\n09876 ...")
draws = [line.strip() for line in raw.splitlines() if line.strip().isdigit() and len(line.strip())==5]
n_draw = len(draws)
st.write(f"📊 โหลดข้อมูล **{n_draw}** งวด")

# ─────── สถิติใหม่ (5 งวด) ───────
def analyze_repeat_digits(nums, window=5):
    last = nums[-window:]
    all_digits = ''.join(last)
    c = Counter(all_digits)
    repeats = [d for d, cnt in c.items() if cnt > 1]
    return repeats, c

# ทำนายข้ามงวด
def predict_cross_patterns(nums, window=5, topk=2):
    repeats, freq = analyze_repeat_digits(nums, window)
    # เลือกตัวเลขที่ซ้ำบ่อยสุด
    hot_digits = sorted(repeats, key=lambda d: freq[d], reverse=True)[:topk]
    # สร้างชุดสองตัวบน-ล่าง 4 ชุด
    two_digit = []
    for a in hot_digits:
        for b in hot_digits:
            if len(two_digit) >= 4: break
            two_digit.append(a + b)
        if len(two_digit) >= 4: break
    # สร้างชุดสามตัวบน 1 ชุด (เบิ้ล-หาม)
    if len(hot_digits) >= 2:
        a, b = hot_digits[0], hot_digits[1]
        triple = [a + a + b, a + b + a, b + a + a]
        three_digit = triple[0]
    else:
        three_digit = hot_digits[0] * 3 if hot_digits else ''
    return two_digit, three_digit

# ─────── แสดงผล Logic ปกติ (5 งวด) ───────
if n_draw >= 5:
    st.subheader("📈 วิเคราะห์เชิงสถิติ (Cross-Draw 5 งวด)")
    two_sets, three_set = predict_cross_patterns(draws, window=5)
    st.write("**เลขที่มักออกซ้ำ (Digits Repeated):**", ', '.join(sorted(set(''.join(draws[-5:]))))
    st.write("**ทำนายงวดถัดไป - สองตัวบน & ล่าง 4 ชุด:**", ', '.join(two_sets))
    st.write("**ทำนายงวดถัดไป - สามตัวบน 1 ชุด:**", three_set)
else:
    st.info("ต้องมีข้อมูลอย่างน้อย 5 งวด สำหรับวิเคราะห์แบบ Cross-Draw")

# ─────── ML Part (เดิม) ───────
def predict_next_digit_ml(nums, window=4):
    nums = [list(map(int, list(x))) for x in nums]
    X, y = [], []
    for i in range(len(nums)-window):
        features = np.array(nums[i:i+window]).flatten()
        target = nums[i+window][-1]
        X.append(features)
        y.append(target)
    if len(X) < 10: return None
    X, y = np.array(X), np.array(y)
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    model.fit(X, y)
    last_features = np.array(nums[-window:]).flatten().reshape(1, -1)
    pred_digit = model.predict(last_features)[0]
    return pred_digit

if n_draw >= 25:
    st.subheader("🤖 AI (ML) ทำนายเลขหลักสุดท้ายงวดถัดไป")
    pred_digit = predict_next_digit_ml(draws, window=4)
    if pred_digit is not None:
        st.write(f"**AI ทำนายเลขหลักสุดท้าย:** <span style='font-size:2em;color:green'>{pred_digit}</span>", unsafe_allow_html=True)
    else:
        st.warning("ข้อมูลยังน้อยเกินไปสำหรับ ML ทำนาย (ต้องอย่างน้อย 25 งวด)")
else:
    st.info("ใส่ข้อมูลอย่างน้อย 25 งวด เพื่อให้ AI (ML) เรียนรู้และทำนายเลขหลักสุดท้ายได้")

st.caption("© 2025 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก พร้อม AI/ML")
