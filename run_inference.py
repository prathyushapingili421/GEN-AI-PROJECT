# run_inference.py
from diagnosis_model import load_model
from utils import retrieve_similar_notes, build_prompt, clean_response
import torch

tokenizer, model = load_model()

def suggest_diagnosis(patient_info):
    context = retrieve_similar_notes(patient_info)
    prompt = build_prompt(patient_info, context)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_response(decoded.split("Diagnosis and Explanation:")[-1])

# 🔍 Sample test
query = "Patient presents with shortness of breath, chest tightness, and was admitted for 3 days under cardiology."
diagnosis = suggest_diagnosis(query)
print("\n🩺 Suggested Diagnosis and Explanation:\n", diagnosis)