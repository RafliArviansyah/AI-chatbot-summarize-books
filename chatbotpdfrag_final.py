import streamlit as st
import requests
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import speech_recognition as sr
from gtts import gTTS
import os

# API Key
OPENROUTER_API_KEY = st.secrets["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"]

# Konfigurasi Halaman
st.set_page_config(page_title="Chatbot Self Improvement", layout="wide")

# === CSS UI ===
st.markdown("""
    <style>
        body {
            background-color: #121212;
        }
        .stButton > button {
            width: 80px;
            background-color: #007bff;
            color: white;
            font-size: 14px;
            padding: 5px 10px;
            border-radius: 5px;
        }
        .stTextArea textarea {
            background-color: #fcfcfc;
            color: black;
            border-radius: 8px;
        }
        .chat-container {
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin-bottom: 10px;
        }
        .bot-message {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            color: black;
            text-align: left;
        }
        .user-message {
            background-color: #007AFF;
            padding: 10px;
            border-radius: 10px;
            color: white;
            text-align: right;
        }
        .chat-input-container {
            position: fixed;
            bottom: 10px;
            left: 0;
            width: 100%;
            background: #121212;
            padding: 10px;
            box-shadow: 0px -2px 10px rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# === Load Model Embedding ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Fungsi Ekstrak Teks dari PDF ===
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    doc.close()
    return text

# === Fungsi Membuat FAISS Index ===
def create_faiss_index(text_chunks):
    embeddings = embedding_model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    return index, text_chunks

# === Kontekstual Memory ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_contextual_memory():
    """Mengambil 5 interaksi terakhir untuk mempertahankan konteks."""
    return "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history[-5:]])

# === STT ===
def recognize_speech_and_respond():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="en")

            st.session_state.chat_history.append(("user", text))

            # Langsung mencari jawaban setelah input suara
            return process_user_input(text)

        except sr.UnknownValueError:
            return "⚠️ Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "⚠️ Error: Speech Recognition service is unavailable."

# === TTS ===
def text_to_speech(text):
    audio_folder = "audio_responses"
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    file_path = os.path.join(audio_folder, f"response_{int(time.time())}.mp3")
    
    tts = gTTS(text, lang="en")
    tts.save(file_path)

    return file_path

# === Fungsi Pemrosesan Input User ===
def process_user_input(user_input):
    if not uploaded_file:
        return "⚠ Please upload the PDF first."
    elif not user_input.strip():
        return "⚠ Please enter a question!"

    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("🤔 Looking for answers..."):
        user_embedding = embedding_model.encode([user_input])
        D, I = index.search(np.array(user_embedding, dtype=np.float32), k=3)
        retrieved_text = " ".join([chunk_texts[i] for i in I[0]])

        context = get_contextual_memory()
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            data=json.dumps({
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [
                    {"role": "system", "content": f"Document:\n{retrieved_text}"},
                    {"role": "user", "content": context},
                    {"role": "user", "content": user_input}
                ],
            })
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            st.session_state.chat_history.append(("bot", answer))
            st.markdown(f"<div class='chat-container bot-message'>{answer}</div>", unsafe_allow_html=True)

            # Play otomatis TTS
            audio_file = text_to_speech(answer)
            st.audio(audio_file, format="audio/mp3")
            return answer
        else:
            return f"❌ Error: {response.status_code} - {response.text}"

# === Sidebar (Upload PDF) ===
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF document here", type="pdf")

    if uploaded_file:
        with st.spinner("📖 Processing PDF..."):
            time.sleep(2)
            st.success("✅ PDF processed successfully!")

# === Proses PDF jika diunggah ===
if uploaded_file is not None:
    with st.spinner("🤓 I'm Reading..."):
        text = extract_text_from_pdf(uploaded_file)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        text_chunks = text_splitter.split_text(text)

        index, chunk_texts = create_faiss_index(text_chunks)
        
        st.success("✅ PDF successfully processed and ready to use!")

# === Chat Area ===
st.markdown("<h1 style='text-align: center; color: white;'>🤖 Chatbot - Self Improvement</h1>", unsafe_allow_html=True)

# === Menampilkan Chat History di UI ===
st.subheader("⚡Ask Anything, Get Smarter!")
for role, message in st.session_state.chat_history[-5:]:
    st.markdown(f"<div class='chat-container {'user-message' if role == 'user' else 'bot-message'}'>{message}</div>", unsafe_allow_html=True)

# === Input dengan Voice Recognition Otomatis ===
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area("💬 Type your question...", height=100)

with col2:
    if st.button("🎤 Speak"):
        recognize_speech_and_respond()

# === Tombol Kirim ===
if st.button("➤ Kirim"):
    if user_input.strip():
        process_user_input(user_input)

# === Footer ===
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; background-color: #262730; color: white; text-align: center; padding: 10px; font-size: 14px;">
        Developed by <b>Rafli Altair</b> 🚀
    </div>
""", unsafe_allow_html=True)
