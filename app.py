import os
import faiss
import numpy as np
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---- Gemini Setup ----
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ---- Load Saved Files ----
def load_saved_data():
    index = faiss.read_index("index.faiss")
    embeddings = np.load("embeddings.npy")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # reload model fresh
    return embedder, index, embeddings, chunks

def answer_question(question, chunks, embedder, index, embeddings, top_k=5):
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed), top_k)
    context = "\n".join([chunks[i] for i in I[0]])

    # Improved prompt: allows fallback if context is weak
    prompt = f"""You are a helpful assistant.
Use the context below if it is relevant. 
If the context is not sufficient, answer using your own knowledge.

Context:
{context}

Question:
{question}
"""
    response = gemini.generate_content(prompt)
    return response.text

# ---- Streamlit UI ----
st.title("📄 RAG App with Gemini + FAISS")

try:
    embedder, index, embeddings, chunks = load_saved_data()
    st.success("Index, embeddings, and chunks loaded successfully!")

    question = st.text_input("Ask a question about the PDF:")
    if question:
        answer = answer_question(question, chunks, embedder, index, embeddings)
        st.markdown("### Gemini says:")
        st.write(answer)

except Exception as e:
    st.error(f"Could not load saved data: {e}")
    st.info("Make sure index.faiss, embeddings.npy, and chunks.pkl are in your repo.")
