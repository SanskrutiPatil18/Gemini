import os
import faiss
import numpy as np
import pickle
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---- Gemini Setup ----
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ---- Helper Functions ----
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, max_length=500):
    paragraphs = text.split('\n')
    chunks, chunk = [], ""
    for para in paragraphs:
        if len(chunk) + len(para) < max_length:
            chunk += " " + para
        else:
            chunks.append(chunk.strip())
            chunk = para
    chunks.append(chunk.strip())
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index, embeddings

def answer_question(question, chunks, embedder, index, embeddings, top_k=5):
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed), top_k)
    context = "\n".join([chunks[i] for i in I[0]])
    prompt = f"""Answer the question based on the below context:

Context:
{context}

Question:
{question}
"""
    response = gemini.generate_content(prompt)
    return response.text

# ---- Streamlit UI ----
st.title("📄 RAG App with Gemini + FAISS")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

    # Extract and chunk
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Embed and index
    embedder, index, embeddings = embed_chunks(chunks)

    # Save FAISS + embeddings + chunks (small files, not the big model)
    faiss.write_index(index, "index.faiss")
    np.save("embeddings.npy", embeddings)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    st.info("Embeddings created and FAISS index built.")

    # Question input
    question = st.text_input("Ask a question about the PDF:")
    if question:
        answer = answer_question(question, chunks, embedder, index, embeddings)
        st.markdown("### Gemini says:")
        st.write(answer)
