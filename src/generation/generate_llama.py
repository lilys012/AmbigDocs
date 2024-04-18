import json
from tqdm import tqdm
import random
import sys
from transformers import AutoTokenizer,  AutoModelForSeq2SeqLM, AutoModelForCausalLM
import json
from tqdm import tqdm
import sys
import torch
from peft import PeftModel, PeftConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration

max_source_length = 128
max_target_length = 128

tokenizer_conv = AutoTokenizer.from_pretrained('../../models/question_converter-3b', max_length=max_source_length, padding=False, truncation=True, add_special_tokens=True)
model_conv = AutoModelForSeq2SeqLM.from_pretrained('../../models/question_converter-3b')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_conv.to(device)

tokenizer_nli = T5Tokenizer.from_pretrained("../../models/t5_xxl_true_nli_mixture")
model_nli = T5ForConditionalGeneration.from_pretrained("../../models/t5_xxl_true_nli_mixture")
model_nli.to(device)

peft_model_id = sys.argv[1]
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

with open("../../exp/generation/sim_pages.json", "r") as f1: passages = json.load(f1)
instruction = "### Instruction:\nYou are given two passages about an ambiguous entity with different interpretations. Create a question about the entity, which each passage must answer differently. Find the shortest span of each passage as an answer to each passage.\n\n"

def to_prompt(passage, idx):
    prompt = f"Passage {idx+1}: " 
    prompt += passage["passages"]["passages"][idx]["title"] + " | " + passage["passages"]["passages"][idx]["text"]
    return prompt

sample_idx = random.sample(range(len(passages)), len(passages))[1500:]

ret_list = []
fail, same = 0, 0
for i, sidx in enumerate(tqdm(sample_idx)):
    passage = passages[sidx]
 
    fromp = passage["from"].strip()
    if fromp.lower()[len(fromp)-len("(disambiguation)"):] == "(disambiguation)": fromp = fromp[:len(fromp)-len("(disambiguation)")].strip()
    prompt = instruction + "Entity: "+ fromp + "\n"+ to_prompt(passage, 0) + "\n"+to_prompt(passage, 1) + "\n\n### Response:\nQuestion: "

    model_input = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model_output = model.generate(**model_input, max_new_tokens=150, return_dict_in_generate=True, do_sample=False)
    sequences = model_output.sequences

    decoded_output = tokenizer.decode(
        model_output["sequences"][0],
        skip_special_tokens=True,
    )  

    output_text= decoded_output[len(prompt) :]

    ret_list.append({
        "from" : passage["from"], "passages" : passage["passages"], "output_text" : output_text
    })

print(len(ret_list))

def entail_ans(title, text, question, ans):
    global overlen, max_target_length, maxlen

    prompt = f'{question} </s> {ans}'
    input_ids = tokenizer_conv(prompt, return_tensors='pt').input_ids.to(device)
    output_ids = model_conv.generate(input_ids, max_new_tokens=max_target_length)
    responses = tokenizer_conv.batch_decode(output_ids, skip_special_tokens=True)
    declare = responses[0]

    input_ids = tokenizer_nli("premise: "+ title + " | " + text + " hypothesis: "+ declare, return_tensors="pt").input_ids.to(device)
    cutoff = 328
    if len(input_ids[0]) > maxlen: maxlen = len(input_ids[0])
    if len(input_ids[0]) > cutoff:
        overlen += 1
        return 0
    else:
        outputs = model_nli.generate(input_ids)
        return tokenizer_nli.decode(outputs[0], skip_special_tokens=True)

with open("../../exp/generation/filtered_data_gpt4.json", "r") as f1: filter_list = json.load(f1)
statuses = [0, 0, 0, 0]
for i, ex in enumerate(tqdm(ret_list)):
    status = 0

    ### Generation failure (not in the specific format)
    try:
        question = ex["output_text"].split("Answer to Passage 1: ")[0].strip()
        ans1 = ex["output_text"].split("Answer to Passage 1: ")[1].split("Answer to Passage 2: ")[0].strip()
        ans2 = ex["output_text"].split("Answer to Passage 1: ")[1].split("Answer to Passage 2: ")[1].split("</s>")[0].strip()
        if len(ans2.split("\n")) > 1: ans2 = ans2.split("\n")[0].strip()
        else: ans2 = ans2.split(". ")[0].strip()
        if len(ans1.split(".")) > 4 or len(ans2.split(".")) > 4: status = 1
    except: status = 1

    ### Does the question talk about a specific entity or just mention both entities? Are passages parallel or hierarchical? 
    if status == 0:
        fromp = ex["from"].strip()
        if fromp[len(fromp)-len("(disambiguation)"):] == "(disambiguation)": fromp = fromp[:len(fromp)-len("(disambiguation)")].strip()
        idx = question.lower().find(fromp.lower())
        if idx != -1: 
            if ex["passages"]["passages"][0]["title"].lower() == fromp.lower() or ex["passages"]["passages"][1]["title"].lower() == fromp.lower(): pass
            elif ex["passages"]["passages"][0]["title"].lower() in question.lower() or ex["passages"]["passages"][1]["title"].lower() in question.lower(): status = 2
        else: 
            if ex["passages"]["passages"][0]["title"].lower() in question.lower() and ex["passages"]["passages"][1]["title"].lower() in question.lower(): status = 2
        if 'mention' in question.lower() or 'entity' in question.lower() or 'context' in question.lower() or 'passage' in question.lower(): status = 2
        if "passage" in ans1.lower() or "mention" in ans1.lower() or "passage" in ans2.lower() or "mention" in ans2.lower(): status = 2
        ans1_words, ans2_words = set(ans1.lower().split(" ")), set(ans2.lower().split(" "))
        if (ans1.lower() in ans2.lower() or ans2.lower() in ans1.lower()) or (len(ans1_words & ans2_words) / len(ans1_words | ans2_words) > 0.75): status = 2

    ### Are both answers independently correct? 
    if status == 0:
        question1, question2 = question, question
        if idx != -1:
            question1 = question[:idx] + ex["passages"]["passages"][0]["title"] + question[idx+len(fromp):]
            question2 = question[:idx] + ex["passages"]["passages"][1]["title"] + question[idx+len(fromp):]
        gen_ans1 = entail_ans(ex["passages"]["passages"][0]["title"], ex["passages"]["passages"][0]["text"], question1, ans1)
        gen_ans2 = entail_ans(ex["passages"]["passages"][1]["title"], ex["passages"]["passages"][1]["text"], question2, ans2)

        try:
            if int(gen_ans1) == 1 and int(gen_ans2) == 1: pass
            else: status = 3
        except: status = 3

    ex["status"] = status
    ex["output_text"] = question + "\nAnswer to Passage 1: " + ans1 + "\nAnswer to Passage 2: " + ans2
    ex["data"] = {"question":question, "ans1":ans1, "ans2":ans2}
    if status == 0: 
        filter_list.append(ex)
    statuses[status] += 1
print(statuses)

with open("../../exp/generation/filtered_data.json", "w") as f1: json.dump(filter_list, f1, indent=4)