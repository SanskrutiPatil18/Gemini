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
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # reload fresh
    return embedder, index, embeddings, chunks

def answer_question(question, chunks, embedder, index, embeddings, top_k=5):
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed), top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant.
Here is the retrieved context from the PDF:

{context}

Now answer the question below. 
If the context is relevant, use it. 
If not, answer using your own knowledge.

Question:
{question}
"""

    response = gemini.generate_content(prompt)

    if hasattr(response, "text") and response.text:
        return response.text, retrieved_chunks
    else:
        return "Sorry, I couldn’t generate an answer. Try rephrasing your question.", retrieved_chunks

# ---- Streamlit UI ----
st.title("📄 RAG App with Gemini + FAISS")

try:
    embedder, index, embeddings, chunks = load_saved_data()
    st.success("Index, embeddings, and chunks loaded successfully!")

    question = st.text_input("Ask a question about the PDF:")
    if question:
        answer, retrieved_chunks = answer_question(question, chunks, embedder, index, embeddings)
        st.markdown("### Gemini says:")
        st.write(answer)

        st.markdown("### Retrieved Context:")
        st.write("\n\n".join(retrieved_chunks))

except Exception as e:
    st.error(f"Could not load saved data: {e}")
    st.info("Make sure index.faiss, embeddings.npy, and chunks.pkl are in your repo.")
