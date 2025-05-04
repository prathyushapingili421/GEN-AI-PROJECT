# diagnosis_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    base_model_id = "stanford-crfm/biomedlm"
    adapter_path = "./biomedlm-diagnosis-finetuned"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    return tokenizer, model
