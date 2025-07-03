import streamlit as st
from collections import defaultdict, Counter
import numpy as np
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="StockLottoAI 5 หลัก", page_icon="🎯", layout="centered")
st.title("🎯 StockLottoAI - วิเคราะห์หวยหุ้น 5 หลัก + Hybrid AI")

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
    if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit() and len(parts[0])==3 and len(parts[1])==2:
        draws.append(parts[0]+parts[1])
n_draw = len(draws)
st.write(f"📊 โหลดข้อมูล **{n_draw}** งวด")

# ─────── Weighted Frequency ───────
def weighted_count(items, weights):
    cnt = defaultdict(float)
    for it, w in zip(items, weights): cnt[it] += w
    return cnt

def predict_weighted_pairs(nums, window=20, decay=0.8, topk=4):
    last = nums[-window:]
    weights = [decay**i for i in range(len(last)-1, -1, -1)]
    pairs = [d[3:] for d in last]
    cnt = weighted_count(pairs, weights)
    sorted_p = [p for p,_ in sorted(cnt.items(), key=lambda x:-x[1])]
    return sorted_p[:topk]

def predict_weighted_triplets(nums, window=20, decay=0.8, topk=2):
    last = nums[-window:]
    weights = [decay**i for i in range(len(last)-1, -1, -1)]
    tris = [d[:3] for d in last]
    cnt = weighted_count(tris, weights)
    sorted_t = [t for t,_ in sorted(cnt.items(), key=lambda x:-x[1])]
    return sorted_t[:topk]

# ─────── ML-based prediction ───────
def predict_ml_pairs(nums, window=10, topk=2):
    pairs = [int(d[3:]) for d in nums]
    X, y = [], []
    for i in range(len(pairs)-window):
        X.append(pairs[i:i+window])
        y.append(pairs[i+window])
    if len(X) < window*2: return []
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=0)
    model.fit(np.array(X), np.array(y))
    probs = model.predict_proba([pairs[-window:]])[0]
    classes = model.classes_
    idx = np.argsort(probs)[-topk:][::-1]
    return [f"{classes[i]:02d}" for i in idx]

def predict_ml_triplets(nums, window=10, topk=1):
    tris = [int(d[:3]) for d in nums]
    X, y = [], []
    for i in range(len(tris)-window):
        X.append(tris[i:i+window])
        y.append(tris[i+window])
    if len(X) < window*2: return []
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=0)
    model.fit(np.array(X), np.array(y))
    probs = model.predict_proba([tris[-window:]])[0]
    classes = model.classes_
    idx = np.argmax(probs)
    return [f"{classes[idx]:03d}"]

# ─────── DISPLAY RESULTS ───────
if n_draw >= 20:
    st.subheader("🔀 Hybrid Prediction: Weighted + ML (Exclude Overused Triplets)")
    # Predict pairs
    w_pairs = predict_weighted_pairs(draws)
    ml_pairs = predict_ml_pairs(draws)
    pairs = sorted(set(w_pairs[:2] + ml_pairs)) + w_pairs[2:4]
    # Predict triplets
    w_tris = predict_weighted_triplets(draws)
    ml_tris = predict_ml_triplets(draws)
    candidate_tris = w_tris + ml_tris
    # Exclude any triplet appearing >=2 times in history
    hist = Counter([d[:3] for d in draws])
    tris_filtered = [t for t in candidate_tris if hist.get(t,0) < 2]
    tris = tris_filtered[:2]
    st.write("**ทำนายสองตัวบน & สองตัวล่าง (4 ชุด):**", ', '.join(pairs[:4]))
    st.write("**ทำนายสามตัวบน (2 ชุด):**", ', '.join(tris))
else:
    st.info("ต้องมีข้อมูลอย่างน้อย 20 งวด สำหรับ Hybrid Prediction")

# ─────── Original ML for last digit ───────
st.caption("Note: For granular digit prediction, use AI หลักสุดท้าย ในหน้า ML section")
