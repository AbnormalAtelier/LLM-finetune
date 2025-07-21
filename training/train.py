from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import bitsandbytes as bnb
import torch, os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
save_dir = os.environ.get("OUTPUT_DIR", "/checkpoints")

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,          # compute still fp16
    load_in_4bit=True,                 # **bnb INT‑4 weights on GPU**
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    device_map="auto"                  # place layers across visible GPUs
)

peft_cfg = LoraConfig(
    r=8, lora_alpha=32, bias="none",
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()      # sanity check
model.gradient_checkpointing_enable()   # **recompute activations**

ds = load_dataset("yentinglin/aime_2025")

args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=1,      # ↓ micro‑batch
    gradient_accumulation_steps=32,     # keep global batch = 32
    num_train_epochs=3,
    fp16=True,                          # compute precision
    gradient_checkpointing=True,        # redundant but explicit
    ddp_find_unused_parameters=False,   # memory + speed
    deepspeed="ds_config.json",         # see below
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,                 # keep disk in check
)

Trainer(model=model, args=args, train_dataset=ds)
model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)

