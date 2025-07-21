from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Setup
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "/app/fine-tuned-models"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

@app.post("/chat")
def chat(req: ChatRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}
