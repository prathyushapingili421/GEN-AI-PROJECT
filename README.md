# genai-diagnosis-assistant

# 🧠 Clinical Diagnosis Assistant using Fine-Tuned BioMedLM (LoRA + FAISS + RAG)

This repository contains an end-to-end **LLM-powered clinical diagnosis generation system**. It leverages a fine-tuned version of `stanford-crfm/biomedLM` using **LoRA adapters**, combines it with **semantic retrieval (FAISS)**, and uses **prompt engineering** to generate accurate and context-aware diagnoses for structured patient summaries.

---

## 🔬 Project Highlights

- ✅ **BioMedLM fine-tuned with LoRA** for efficient domain adaptation.
- 🧠 **Retrieval-Augmented Generation (RAG)** using a FAISS index of past discharge summaries.
- 📄 Cleaned, hallucination-free diagnosis output using regex-based post-processing.
- 🧪 Ready-to-query inference via `run_inference.py`.
- 🧰 Modular code structure (tokenization, model loading, semantic search, prompt generation).
- ⚙️ Lightweight: runs on CPU without full model retraining.

---

## 📁 Directory Structure

├── biomedlm-diagnosis-finetuned/ # Fine-tuned adapter weights + tokenizer

├── hcup_faiss_index.idx # Semantic index of historical notes

├── hcup_id_mapping.pkl # Mapping index to note content

├── diagnosis_model.py # Loads model + inference function

├── utils.py # Prompt engineering, semantic search, cleaning

├── run_inference.py # CLI-style test script

├── requirements.txt # Dependencies


---

## ⚙️ Setup Instructions

### 1. Clone the repo

```
git clone https://github.com/yourusername/clinical-diagnosis-assistant.git
cd clinical-diagnosis-assistant

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python run_inference.py

Sample query:
query = "Patient presents with shortness of breath, chest tightness, and was admitted for 3 days under cardiology."
Suggested Diagnosis and Explanation:
The patient is likely experiencing acute coronary syndrome. Based on the clinical history and ECG findings, the chest pain and shortness of breath suggest myocardial ischemia. Hospitalization was appropriate for further cardiac workup.
```

## Acknowledgments

### Stanford CRFM - BioMedLM

### FAISS - Facebook AI Similarity Search

### LoRA - Low-Rank Adaptation

