import os
import pickle

import gradio as gr
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Retrieval artifacts
INDEX_PATH = os.path.join(BASE_DIR, "veritas_faiss.index")
# Updated to point to the original metadata with chunk text
META_PATH  = os.path.abspath(
    os.path.join(BASE_DIR, "../mistral-7b/veritas_metadata.pkl")
)

# Model artifacts
MODEL_PATH = BASE_DIR  # assumes tokenizer & model live here

# Retrieval settings
TOP_K = 5
EMBED_PROMPT = "Represent the scientific passage for retrieval: {}"

# Generation settings
MAX_NEW_TOKENS = 300
TEMPERATURE    = 0.7

# ─── Load Retrieval Components ─────────────────────────────────────────────────

index = faiss.read_index(INDEX_PATH)
print(f"[RAG] Index dimensionality: {index.d}")

with open(META_PATH, "rb") as f:
    metas = pickle.load(f)

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ─── Load Mistral Model ────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float32,
)
model.eval()

# ─── Inference Function with RAG ───────────────────────────────────────────────

def generate_response(user_query: str) -> str:
    # 1. Encode query
    emb_input = EMBED_PROMPT.format(user_query)
    q_emb = encoder.encode(emb_input, normalize_embeddings=True).astype("float32").reshape(1, -1)
    print(f"[RAG] Query embedding shape: {q_emb.shape}")

    # 2. Search index
    distances, indices = index.search(q_emb, TOP_K)
    print("[DEBUG] Indices returned:", indices)
    print("[DEBUG] Distances returned:", distances)

    # 2.a. Inspect the first metadata entry returned
    if indices.shape[1] > 0:
        first = int(indices[0][0])
        print(f"[DEBUG] metas[{first}] keys:", metas[first].keys())
        print(f"[DEBUG] metas[{first}] entry:\n", metas[first])

    # 3. Gather retrieved texts
    retrieved_texts = []
    for idx in indices[0]:
        chunk = metas[idx].get("text") or metas[idx].get("chunk") or ""
        retrieved_texts.append(chunk)

    # 4. DEBUG: show what was retrieved
    context = "\n\n".join(retrieved_texts)
    print("=== Retrieved Context ===\n", context)
    print("=== User Query =======\n", user_query)

    # 5. Build RAG prompt
    combined_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )

    # 6. Generate answer
    inputs = tokenizer(combined_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─── Launch Gradio ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gr.Interface(
        fn=generate_response,
        inputs=gr.Textbox(lines=2, placeholder="Ask a question…"),
        outputs="text",
        title="🔮 Mistral 7B + RAG Chat",
        description="Locally powered Mistral with FAISS-based retrieval"
    ).launch(share=True)
