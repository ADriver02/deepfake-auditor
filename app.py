import streamlit as st
from deepfake_model import extract_score
from grok_api import ask_grok
import os

st.set_page_config(page_title="Deepfake Auditor", layout="centered")
st.title("Deepfake Auditor")
st.caption("Détecte les faux candidats en <5s")

uploaded_file = st.file_uploader("Upload vidéo (max 2 min)", type=["mp4", "mov"])

if uploaded_file:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyse IA..."):
        score = extract_score(video_path)
        prompt = f"Score deepfake : {score:.2f}. Vrai ou faux candidat ? Réponds en 1 phrase."
        verdict = ask_grok(prompt)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score IA", f"{score:.2f}")
    with col2:
        st.metric("Confiance Grok", f"{int((1-score)*100)}%")

    st.write(f"**Verdict Grok** : {verdict}")

    if score > 0.6:
        st.error("DEEPFAKE DÉTECTÉ")
    elif score > 0.4:
        st.warning("Suspicion – Vérification manuelle")
    else:
        st.success("Semble authentique")

    os.remove(video_path)
