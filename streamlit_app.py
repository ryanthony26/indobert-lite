import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./saved_fine_tuned"   
TOKENIZER_DIR = "./tokenizers"      
MAX_LENGTH = 256

ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir: str = MODEL_DIR, tokenizer_dir: str = TOKENIZER_DIR):
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

def predict_single(text: str):
    if not text or not text.strip():
        return None
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        padding=False
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0].cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL.get(pred_id, str(pred_id))
    prob_dict = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}
    return {"label": pred_label, "probs": prob_dict, "scores": probs.tolist()}


def predict_batch(series: pd.Series, text_col: str = None, batch_size: int = 16):
    texts = series if text_col is None else series[text_col]
    results = []
    model_device = device
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        enc = tokenizer(batch_texts, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", padding=True)
        enc = {k: v.to(model_device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            preds = probs.argmax(axis=1)
        for p, pr in zip(preds, probs):
            results.append({
                "label": ID2LABEL.get(int(p), str(p)),
                "probs": {ID2LABEL[i]: float(pr[i]) for i in range(len(pr))},
                "scores": pr.tolist()
            })
    return results


st.set_page_config(page_title="Sentiment App", layout="wide")
st.title("Klasifikasi Ulasan Game Roblox — IndoBERT Lite (Fine-tuned)")

st.subheader("Prediksi 1 Teks")
user_text = st.text_area("Masukkan teks (bahasa Indonesia)", height=150, placeholder="Contoh: Gamenya keren")
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Isi teks dulu.")
        else:
            with st.spinner("Memprediksi..."):
                res = predict_single(user_text)
            if res is None:
                st.error("Gagal memprediksi.")
            else:
                st.markdown("### Hasil")
                st.write(f"**Label:** {res['label']}")
                st.write("**Probabilitas per kelas:**")
                st.table(pd.DataFrame.from_dict(res["probs"], orient="index", columns=["probability"]))
                # Bar chart
                st.bar_chart(pd.Series(res["probs"]))
