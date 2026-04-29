import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    import faiss
except ModuleNotFoundError:
    faiss = None


DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


@st.cache_resource(show_spinner=False)
def load_saved_data():
    embeddings = np.load("embeddings.npy")
    with open("chunks.pkl", "rb") as source:
        chunks = pickle.load(source)

    if len(chunks) != len(embeddings):
        raise ValueError(
            "embeddings.npy and chunks.pkl do not contain the same number of items"
        )
    if len(chunks) == 0:
        raise ValueError("No chunks were found in chunks.pkl")

    index = None
    retrieval_message = "Retrieval backend: NumPy fallback"
    if faiss is not None and os.path.exists("index.faiss"):
        index = faiss.read_index("index.faiss")
        retrieval_message = "Retrieval backend: FAISS"
    elif faiss is None:
        retrieval_message = (
            "FAISS is not installed, so retrieval is using a NumPy fallback."
        )
    else:
        retrieval_message = (
            "index.faiss was not found, so retrieval is using a NumPy fallback."
        )

    return index, embeddings, chunks, retrieval_message


@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_embedder():
    if SentenceTransformer is None:
        return None, "sentence-transformers is not installed."
    return get_embedder(), "Query embedder: all-MiniLM-L6-v2"


def search_chunks(query_embedding, index, embeddings, top_k):
    top_k = min(top_k, len(embeddings))
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    if query_embedding.ndim == 1:
        query_embedding = query_embedding[None, :]

    if index is not None:
        _, indices = index.search(query_embedding, top_k)
        return indices[0].tolist()

    doc_embeddings = np.asarray(embeddings, dtype=np.float32)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1.0

    query_vector = query_embedding[0]
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        query_norm = 1.0

    similarity_scores = (doc_embeddings / doc_norms) @ (query_vector / query_norm)
    return np.argsort(similarity_scores)[::-1][:top_k].tolist()


def build_prompt(question, context):
    return f"""You are a helpful assistant.
Use the retrieved PDF context to answer the question.
If the context is incomplete, say so briefly.

Context:
{context}

Question:
{question}

Answer:
"""


def load_hf_pretrained(loader, model_name):
    try:
        return loader.from_pretrained(model_name)
    except Exception as exc:
        model_reference = str(model_name)
        if Path(model_reference).expanduser().exists():
            raise RuntimeError(
                f"Could not load the local model at '{model_reference}'."
            ) from exc
        raise RuntimeError(
            "Could not load the Hugging Face model. First run needs internet access, "
            f"or HF model must point to a local model folder. Model: '{model_reference}'."
        ) from exc


def get_max_input_tokens(tokenizer):
    model_max_length = getattr(tokenizer, "model_max_length", 2048)
    if (
        not isinstance(model_max_length, int)
        or model_max_length <= 0
        or model_max_length > 100000
    ):
        return 2048
    return min(model_max_length, 2048)


@st.cache_resource(show_spinner=False)
def get_hf_generator(model_name):
    tokenizer = load_hf_pretrained(AutoTokenizer, model_name)
    model = load_hf_pretrained(AutoModelForCausalLM, model_name)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cpu"
    if torch is not None:
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")
            device = "mps"

    model.eval()
    return tokenizer, model, device


def generate_answer_with_hf(prompt, model_name, max_new_tokens):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "transformers and torch are required for Hugging Face generation."
        )

    tokenizer, model, device = get_hf_generator(model_name)
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered_prompt = prompt

    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=get_max_input_tokens(tokenizer),
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if text:
        return text
    raise RuntimeError("The Hugging Face model returned no text.")


def answer_question(
    question, chunks, embedder, index, embeddings, model_name, top_k, max_new_tokens
):
    query_embedding = embedder.encode([question])
    retrieved_indices = search_chunks(query_embedding, index, embeddings, top_k)
    retrieved_chunks = [chunks[i] for i in retrieved_indices]
    context = "\n".join(retrieved_chunks)
    prompt = build_prompt(question, context)
    return generate_answer_with_hf(prompt, model_name, max_new_tokens)


st.title("RAG App with Hugging Face")

try:
    index, embeddings, chunks, _ = load_saved_data()
except Exception as exc:
    st.error(f"Could not load saved data: {exc}")
    st.stop()

embedder, _ = load_embedder()
if embedder is None:
    st.error("sentence-transformers is not installed.")
    st.stop()

st.sidebar.header("Hugging Face")
model_name = st.sidebar.text_input(
    "HF model", value=os.getenv("HF_MODEL_ID", DEFAULT_HF_MODEL)
).strip()
top_k = st.sidebar.slider(
    "Retrieved chunks",
    min_value=1,
    max_value=min(10, len(chunks)),
    value=min(5, len(chunks)),
)
max_new_tokens = st.sidebar.slider(
    "Max new tokens", min_value=32, max_value=512, value=192, step=32
)
st.sidebar.caption(
    "The model downloads on first use unless the HF model points to a local folder."
)

question = st.text_input("Ask a question about the PDF:")
if question:
    try:
        with st.spinner("Searching the PDF and generating an answer..."):
            answer = answer_question(
                question,
                chunks,
                embedder,
                index,
                embeddings,
                model_name,
                top_k,
                max_new_tokens,
            )
        st.write(answer)
    except Exception as exc:
        st.error(str(exc))
