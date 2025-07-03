import streamlit as st
from collections import defaultdict
import numpy as np
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="StockLottoAI 5 หลัก", page_icon="🎯", layout="centered")
st.title("🎯 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก + AI")

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
    p = line.strip().split()
    if len(p)==2 and p[0].isdigit() and p[1].isdigit() and len(p[0])==3 and len(p[1])==2:
        draws.append(p[0]+p[1])
n_draw = len(draws)
st.write(f"📊 โหลดข้อมูล **{n_draw}** งวด")

# ─────── ฟังก์ชันช่วยนับแบบถ่วงน้ำหนัก ───────
def weighted_count(items, weights):
    cnt = defaultdict(float)
    for it, w in zip(items, weights):
        cnt[it] += w
    return cnt

# ─────── ทำนายสองตัวบน/ล่าง ด้วย Weighted Frequency ───────
def predict_weighted_pairs(nums, window=20, decay=0.9, topk=4):
    last = nums[-window:]
    # สร้างเวกเตอร์น้ำหนัก: งวดล่าสุด w=1, งวดก่อนหน้า w*=decay
    weights = [decay**i for i in range(len(last)-1, -1, -1)]
    # ดึงคู่สองหลักท้าย (สองตัวบน/ล่าง)
    pairs = [draw[3:] for draw in last]
    cnt = weighted_count(pairs, weights)
    # เรียงตามค่าน้ำหนักสูงสุด
    sorted_pairs = sorted(cnt.items(), key=lambda x: -x[1])
    # เลือก topk
    return [p for p, _ in sorted_pairs][:topk]

# ─────── ทำนายสามตัวบน ด้วย Weighted Frequency ───────
def predict_weighted_triplets(nums, window=20, decay=0.9, topk=2):
    last = nums[-window:]
    weights = [decay**i for i in range(len(last)-1, -1, -1)]
    triplets = [draw[:3] for draw in last]
    cnt = weighted_count(triplets, weights)
    sorted_tris = sorted(cnt.items(), key=lambda x: -x[1])
    #คืน topk ชุด
    return [t for t, _ in sorted_tris][:topk]

# ─────── แสดงผลสูตรใหม่ ───────
if n_draw >= 20:
    st.subheader("📈 สูตรใหม่: Weighted Frequency 20 งวดล่าสุด")
    two_sets = predict_weighted_pairs(draws, window=20)
    three_sets = predict_weighted_triplets(draws, window=20)
    st.write("**ทำนายงวดถัดไป (สองตัวบน & ล่าง 4 ชุด):**", ', '.join(two_sets))
    st.write("**ทำนายงวดถัดไป (สามตัวบน 2 ชุด):**", ', '.join(three_sets))
else:
    st.info("ต้องมีข้อมูลอย่างน้อย 20 งวด สำหรับสูตร Weighted Frequency")

# ─────── ML Part (Neural Network) ───────
def predict_next_digit_ml(nums, window=4):
    nums_i = [list(map(int, list(x))) for x in nums]
    X, y = [], []
    for i in range(len(nums_i)-window):
        X.append(np.array(nums_i[i:i+window]).flatten())
        y.append(nums_i[i+window][-1])
    if len(X) < 10:
        return None
    model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=2000, random_state=42)
    model.fit(np.array(X), np.array(y))
    feat = np.array(nums_i[-window:]).flatten().reshape(1,-1)
    return model.predict(feat)[0]

if n_draw >= 25:
    st.subheader("🤖 AI (ML) ทำนายเลขหลักสุดท้ายงวดถัดไป")
    pred = predict_next_digit_ml(draws, window=4)
    if pred is not None:
        st.write(f"**AI ทำนายเลขหลักสุดท้าย:** <span style='font-size:2em;color:green'>{pred}</span>", unsafe_allow_html=True)
    else:
        st.warning("ข้อมูลยังน้อยเกินไปสำหรับ ML ทำนาย (ต้อง 25 งวด)")
else:
    st.info("ใส่ข้อมูลอย่างน้อย 25 งวด เพื่อให้ AI เรียนรู้และทำนาย")

st.caption("© 2025 StockLottoAI - สูตรใหม่ Weighted Frequency + AI/ML")
