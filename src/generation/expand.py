import json
from tqdm import tqdm
import src.data
import sys
import time
import torch
from transformers import AutoTokenizer,  AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = sys.argv[1]
tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token='[PAD]'
model = LlamaForCausalLM.from_pretrained(model_path)
model = model.half()
device = torch.device("cuda")
model.to(device)

max_source_length = 128
max_target_length = 128
overlen = 0
maxlen = 0

tokenizer_conv = AutoTokenizer.from_pretrained('../../models/question_converter-3b', max_length=max_source_length, padding=False, truncation=True, add_special_tokens=True)
model_conv = AutoModelForSeq2SeqLM.from_pretrained('../../models/question_converter-3b')
model_conv.to(device)

tokenizer_nli = T5Tokenizer.from_pretrained("../../models/t5_xxl_true_nli_mixture")
model_nli = T5ForConditionalGeneration.from_pretrained("../../models/t5_xxl_true_nli_mixture")
model_nli.to(device)

print("Loading passages...")
passages = "../../data/psgs_w100.tsv"
start_time_indexing = time.time()
passages = src.data.load_passages(passages)

print("Indexing passages...")
ttoi = {}
for i, p in enumerate(tqdm(passages)):
    if p['title'] not in ttoi: ttoi[p['title']] = []
    ttoi[p['title']].append(i)
print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

with open(f"../../exp/generation/disamb_pages.json", "r") as f1: disamb_pages = json.load(f1)
with open(f"../../exp/generation/filtered_data.json", "r") as f1: train = json.load(f1)

base_prompt = '''[INST]Given a passage related to the question, answer to the question. Find the shortest span of the passage as an answer.[/INST]

Passage: Kiwi Travel International Airlines | Kiwi Travel International Airlines Kiwi Travel International Airlines was a New Zealand based airline which pioneered discount flights between secondary airports in Australia and New Zealand in the mid 1990s. The airline was established by Ewan Wilson and several associates. Ewan Wilson served as CEO and was later convicted on four counts of fraud. It was reported in March 2015 that Wilson was looking at restarting an airline, under the name Kiwi Regional Airlines. The airline started out as Kiwi Travel Air Charters in July 1994, operating weekly charters between Hamilton, New Zealand and Brisbane, Australia, using a leased Air
Question: Who founded Kiwi Airlines?
Answer: Ewan Wilson

Passage: Helianthus occidentalis | Helianthus occidentalis Helianthus occidentalis (fewleaf sunflower or western sunflower) is a species of sunflower native to the Eastern and Central United States. It grows mostly in the Great Lakes Region and in the Ozarks, with additional populations scattered as far as Massachusetts, Texas, and the Florida Panhandle. \"H. occidentalis\" differs from other, similar species by its sparse leaves, most of which are crowded around the lower part of the stem. This perennial plant reaches heights from 2 to 5 ft (60\u2013150 cm). It produces one to several yellow flower heads, each with 8-14 ray florets surrounding more than 50 disc
Question: Where is the Western sunflower primarily found?
Answer: Eastern and Central United States

Passage: Proletarian Unity Party (France) | Proletarian Unity Party (France) The Party of Proletarian Unity (, \"PUP\") was a French socialist political party. It was formed on December 21, 1930 by leftists expelled from the French Communist Party (PCF), together with some who had previously belonged to the left-wing of the Section fran\u00e7aise de l'Internationale ouvri\u00e8re (SFIO). Its members were known in France as \"pupistes\", and one of its notable leaders was Alexandre Bachelet. Owing to proportional representation, it at one time had ten seats in the Chamber of Deputies of the Third Republic. The PUP affiliated to the London Bureau of left-socialist parties. On January
Question: When was the Proletarian Unity Party formed?
Answer: December 21, 1930

Passage: The Kent Island Bay Times | The Kent Island Bay Times The Kent Island Bay Times or simply the Bay Times is a weekly newspaper based in Stevensville, Maryland and is owned by Chesapeake Publishing, the same company that owns The Record Observer in Centreville, Maryland and The Star Democrat in Easton, Maryland, and other newspapers on both sides of the Chesapeake Bay. It is published every Wednesday and covers news mostly in southern Queen Anne's County including all of Kent Island, Grasonville and Queenstown. The Bay Times was Founded by Christopher J. Rosendale Sr. and his wife Mary Lou in 1963. The first issue of
Question: What type of community does the Bay Times primarily serve?
Answer: Southern Queen Anne's County including all of Kent Island, Grasonville and Queenstown

'''

def to_prompt(passage, question):
    prompt = f"Passage: " + passage["title"] + " | " + passage["text"] + "\nQuestion: " + question +"\nAnswer: "
    return prompt

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
        value = tokenizer_nli.decode(outputs[0], skip_special_tokens=True)
        if value.isdigit(): return int(value)
        else: return 0

ret_list = []
for ex in tqdm(train):
    candidates = []
    for de in disamb_pages[ex["from"]]:
        if de not in ttoi: continue
        if de == ex["passages"]["passages"][0]["title"] or de == ex["passages"]["passages"][1]["title"]: continue
        candidate = {"pp":sys.float_info.max}
        for pid in ttoi[de]:
            try:
                question = ex["data"]["question"]
                prompt = base_prompt + to_prompt(passages[pid], question)

                model_input = tokenizer([prompt], return_tensors="pt").to(device)
                model_output = model.generate(**model_input, max_new_tokens=50, return_dict_in_generate=True, output_scores=True, do_sample=False)
                sequences = model_output.sequences
                transition_scores = model.compute_transition_scores(model_output.sequenences, model_output.scores, normalize_logits=True)

                decoded_output = tokenizer.decode(
                    model_output["sequences"][0],
                    skip_special_tokens=True,
                )

                pp = 0.
                if len(transition_scores[0]):
                    for sc in transition_scores[0]: pp += sc.cpu().item()
                    pp /= -len(transition_scores[0])

                output_text = decoded_output[len(prompt):]

                answer = output_text.split("</s>")[0].split("\n")[0].strip()
                if not len(answer): continue
                if answer[-1] == '.': answer = answer[:-1]
                if len(answer.split(".")) > 4: continue

                fromp = ex["from"].strip()
                if fromp[len(fromp)-len("(disambiguation)"):] == "(disambiguation)": fromp = fromp[:len(fromp)-len("(disambiguation)")].strip()
                idx = question.lower().find(fromp.lower())
                question_sub = question[:idx] + de + question[idx+len(fromp):]
                gen_ans = entail_ans(de, passages[pid]["text"], question_sub, answer)
                if gen_ans == 1 and candidate["pp"] > pp: candidate = {"passage": passages[pid], "answer": answer, "pp": pp}
            except: pass
        if "answer" in candidate: candidates.append(candidate)
    
    example = {"from":ex["from"], "passages":ex["passages"]["passages"], "overlap":ex["passages"]["overlap"], "output_text":ex["output_text"], "idx":ex["idx"], "status":ex["status"], "data":{"question":ex["data"]["question"], "answers":[ex["data"]["ans1"], ex["data"]["ans2"]]}, "pp":[0., 0.]}
    for cand in candidates:
        example["passages"].append(cand["passage"])
        example["data"]["answers"].append(cand["answer"])
        example["pp"].append(cand["pp"])
    ret_list.append(example)

with open(f"../../exp/generation/expand_data.json", "w") as f1: json.dump(ret_list, f1, indent=4)
