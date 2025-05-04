import re
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS and ID mapping
index = faiss.read_index("hcup_faiss_index.idx")
with open("hcup_id_mapping.pkl", "rb") as f:
    id_mapping = pickle.load(f)

def retrieve_similar_notes(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    return [id_mapping[i]["Generated_Note"] for i in indices[0]]

def build_prompt(patient_info, historical_notes):
    return (
        "You are a clinical diagnosis assistant.\n\n"
        "### Patient Summary:\n" + patient_info.strip() + "\n\n"
        "### Historical Notes:\n" + "\n".join(f"- {n}" for n in historical_notes) + "\n\n"
        "### Task:\nProvide the most likely diagnosis and medical explanation.\n\nDiagnosis and Explanation:"
    )

def clean_response(text):
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    lines = re.split(r"[.\n]", text)
    filtered = [
        l.strip() for l in lines
        if l and not l.endswith("?") and not re.match(r"(?i)^(what|why|how|is|can|does|when|where)\b", l)
    ]
    return ". ".join(filtered).strip() + "."
