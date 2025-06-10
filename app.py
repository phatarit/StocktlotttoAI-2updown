
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

def get_hot_digit(nums, window):
    last_nums = nums[-window:]
    all_digits = "".join(last_nums)
    c = Counter(all_digits)
    most = c.most_common(1)[0][0] if c else ""
    return most, c

def get_hot_pairs(nums, window, topk=5):
    last_pairs = [num[-2:] for num in nums[-window:]]
    c = Counter(last_pairs)
    return [pair for pair, _ in c.most_common(topk)]

def get_hot_triplets(nums, window, hot_digit, pairs, topk=5):
    triplets = set()
    for p in pairs:
        triplets.add(hot_digit + p)
        triplets.add(p + hot_digit)
        triplets.add(p[0] + hot_digit + p[1])
    last_triples = [num[-3:] for num in nums[-window:]]
    c = Counter(last_triples)
    sorted_tris = [tri for tri, _ in c.most_common(topk)]
    output = list(triplets)[:max(0, topk-len(sorted_tris))] + [tri for tri in sorted_tris if tri not in triplets]
    return output[:topk]

def stat_prediction(nums, window=10, topk=5):
    hot_digit, digit_freq = get_hot_digit(nums, window)
    pairs = get_hot_pairs(nums, window, topk)
    triplets = get_hot_triplets(nums, window, hot_digit, pairs, topk)
    return hot_digit, pairs, triplets, digit_freq

# ----------- ML PART (Neural Network สำหรับเลขหลักสุดท้าย) -----------
def predict_next_digit_ml(nums, window=4, epochs=40):
    nums = [list(map(int, list(x))) for x in nums]
    X, y = [], []
    for i in range(len(nums)-window):
        features = np.array(nums[i:i+window]).flatten()
        target = nums[i+window][-1]
        X.append(features)
        y.append(target)
    if len(X) < 10: return None
    X, y = np.array(X), np.array(y)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(window*5,)),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X, y, epochs=epochs, verbose=0)
    last_features = np.array(nums[-window:]).flatten().reshape(1, -1)
    pred_prob = model.predict(last_features, verbose=0)
    pred_digit = np.argmax(pred_prob)
    return pred_digit

# ------- แสดงผล (10 และ 20 งวด)
for win in [10, 20]:
    if n_draw >= win:
        st.subheader(f"🎲 วิเคราะห์ {win} งวดล่าสุด (Logic ปกติ)")
        hot_digit, pairs, triplets, freq = stat_prediction(draws, window=win, topk=5)
        st.write(f"**เลขเด่น:** <span style='color:red;font-size:2em'>{hot_digit}</span>", unsafe_allow_html=True)
        st.write("**เลขสองตัว (Top 5):**", ", ".join(pairs))
        st.write("**เลขสามตัว (Top 5):**", ", ".join(triplets))
        st.caption(f"ค่าความถี่เลขเด่น: {dict(freq)}")
    else:
        st.info(f"ต้องมีข้อมูลอย่างน้อย {win} งวด สำหรับวิเคราะห์ {win} งวดล่าสุด")

# ---------- ML Predict
if n_draw >= 25:
    st.subheader("🤖 AI (ML) ทำนายเลขเด่นหลักสุดท้ายงวดถัดไป")
    pred_digit = predict_next_digit_ml(draws, window=4, epochs=60)
    if pred_digit is not None:
        st.write(f"**AI ทำนายเลขหลักสุดท้าย:** <span style='font-size:2em;color:green'>{pred_digit}</span>", unsafe_allow_html=True)
    else:
        st.warning("ข้อมูลยังน้อยเกินไปสำหรับ ML ทำนาย (ต้องอย่างน้อย 25 งวด)")
else:
    st.info("ใส่ข้อมูลอย่างน้อย 25 งวด เพื่อให้ AI (ML) เรียนรู้และทำนายเลขหลักสุดท้ายได้")

st.caption("© 2025 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก พร้อม AI/ML")
