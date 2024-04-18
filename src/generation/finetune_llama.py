import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
import json
import random
import sys
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def to_prompt(passage, idx):
    prompt = f"Passage {idx+1}: " 
    prompt += passage["passages"]["passages"][idx]["title"] + " | " + passage["passages"]["passages"][idx]["text"]
    return prompt

cnt = 0
with open("../../exp/generation/filtered_data_gpt4.json", "r") as f1: q_data = json.load(f1)
random.shuffle(q_data)
with open("../../exp/generation/filtered_data_gpt4_to_llama_input.json", "w") as f1:
    for ex in q_data:
        base_prompt = "### Instruction:\nYou are given two passages about an ambiguous entity with different interpretations. Create a question about the entity, which each passage must answer differently. Find the shortest span of each passage as an answer to each passage.\n\n"
        fromp = ex["from"].strip()
        if fromp.lower()[len(fromp)-len("(disambiguation)"):] == "(disambiguation)": fromp = fromp[:len(fromp)-len("(disambiguation)")].strip()
        op_list = ex["output_text"].split("\n")
        question, ans1, ans2 = op_list[0].strip(), op_list[1][len("Answer to Passage 1: "):].strip(), op_list[2][len("Answer to Passage 2: "):].strip()

        prompt = base_prompt + "Entity: "+ fromp + "\n"+ to_prompt(ex, 0) + "\n"+to_prompt(ex, 1) + "\n\n### Response:\nQuestion: "
        prompt += question + "\nAnswer to Passage 1: " + ans1 + "\nAnswer to Passage 2: " + ans2
        json.dump({"text":prompt}, f1)
        f1.write("\n")


data_files = {
    "train": "../../exp/generation/filtered_data_gpt4_to_llama_input.json",
}
dataset = load_dataset("json", data_files=data_files)
train_dataset = dataset['train']

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_auth_token=True,
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

output_dir = "../../exp/generation/gpt4"
num_train_epochs = 5
auto_find_batch_size = True
gradient_accumulation_steps = 1
optim = "paged_adamw_32bit"
save_strategy = "epoch"
learning_rate = 2e-4
lr_scheduler_type = "linear"
warmup_ratio = 0.03
logging_strategy = "steps"
logging_steps = 250
do_eval = True
evaluation_strategy = "steps"
prediction_loss_only = True
eval_steps = 0.2
bf16 = True

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    auto_find_batch_size=auto_find_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy=save_strategy,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    logging_strategy=logging_strategy,
    logging_steps=logging_steps,
    # do_eval=do_eval,
    # evaluation_strategy=evaluation_strategy,
    prediction_loss_only=prediction_loss_only,
    eval_steps=eval_steps,
    bf16=bf16,
)

max_seq_length = 4096

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()