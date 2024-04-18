import json
from openai import OpenAI
from tqdm import tqdm
import random
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm
import sys
import torch
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

api_key = sys.argv[1]
client = OpenAI(api_key=api_key)

with open("../../exp/generation/sim_pages.json", "r") as f1: passages = json.load(f1)
instruction = "You are given two passages about an ambiguous entity with different interpretations. Create a question about the entity, which each passage must answer differently. Find the shortest span of each passage as an answer to each passage."

comp_prompt = '''
Entity: Gimnasia y Esgrima
Passage 1: Gimnasia y Esgrima de Buenos Aires | football. Because of the growth of the activity, the club built one field more. After leaving the official football leagues in the decade of 1920, rugby union was one of the predominant sports of Gimnasia y Esgrima, winning the Torneo de la URBA titles of 1911 and 1912. The club won two more titles in 1932 and 1939, its last championship to date. Gimnasia y Esgrima currently plays in the Grupo II, the second division of the Unión de Rugby de Buenos Aires league system. In 1942 president Aldao moved to the club's distinguished guest apartment, establishing it as its
Passage 2: Gimnasia y Esgrima de Jujuy | Gimnasia y Esgrima de Jujuy Club Atlético Gimnasia y Esgrima , (usually known as Gimnasia y Esgrima de Jujuy or GEJ) is an Argentine sports club based in San Salvador de Jujuy, Jujuy Province. The club was founded in 1931 and is best known for its football team, which currently plays in the Primera B Nacional, the second division of the Argentine football league system. The team is one of the most important football clubs in the North-West region of Argentina. It played most of the time in the Argentine second division, although it has also played at the highest
Question: What is the main sport associated with Gimnasia y Esgrima?
Answer to Passage 1: Rugby union
Answer to Passage 2: Football

Entity: Young Conservatives
Passage 1: Young Conservatives (Czech Republic) | The Young Conservatives () is a political youth organisation in the Czech Republic. It is the youth wing of the Civic Democratic Party (ODS), a centre-right political party, and shares that party's conservative and economically liberal ideology. Young people within the age from 15 to 35 apply for a membership in the MK. Several significant politicians from the ODS party started as members of Young Conservatives, including Jan Zahradil, Ji\u0159\u00ed Posp\u00ed\u0161il, Petr Sokol, Martin Baxa, Petr Gandalovi\u010d, Ivan Langer, Martin Novotn\u00fd, and Pavel Drobil. Former Chairman of Young Conservatives Petr Mach went on to found a
Passage 2: Young Conservatives (UK) | The Young Conservatives (YC) is the youth wing of the Conservative Party in the United Kingdom for members aged 25 and under. The organisation shares the same values and policies as its parent political party with branches being an integrated part of local associations, with the exception of college and university branches which are run independently. YC is both social and political, aiming to bring together young conservatives and encouraging young people to get involved in campaigning. The \"Junior Imperial and Constitutional League\" was formed in 1906 with objectives to encourage practical political work and organisation among
Question: What is the age range for membership in the Young Conservatives?
Answer to Article 1: 15 to 35
Answer to Article 2: 25 and under

'''

def to_prompt(passage, idx):
    prompt = f"Passage {idx+1}: " 
    prompt += passage["passages"]["passages"][idx]["title"] + " | " + passage["passages"]["passages"][idx]["text"]
    return prompt

sample_idx = random.sample(range(len(passages)), len(passages))[:1500]

ret_list = []
fail, same = 0, 0
for i, sidx in enumerate(tqdm(sample_idx)):
    passage = passages[sidx]

    if passage["from"] in ["Gimnasia y Esgrima", "Young Conservatives"]: continue
 
    fromp = passage["from"].strip()
    if fromp.lower()[len(fromp)-len("(disambiguation)"):] == "(disambiguation)": fromp = fromp[:len(fromp)-len("(disambiguation)")].strip()
    prompt = comp_prompt + f"Entity: "+ fromp + to_prompt(passage, 0) + to_prompt(passage, 1) + "\nQuestion:"

    response = client.chat.completions.create(
            model = "gpt-4",
            messages = [{"role":"system", "content":instruction}, {"role":"user", "content":prompt}],
        )
    output_text = response.choices[0].message.content.strip()

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

filter_list = []
statuses = [0, 0, 0, 0]
for i, ex in enumerate(tqdm(ret_list)):
    status = 0

    ### Generation failure (not in the specific format)
    try:
        op_list = ex["output_text"].split("\n")
        question, ans1, ans2 = op_list[0].strip(), op_list[1][len("Answer to Passage 1: "):].strip(), op_list[2][len("Answer to Passage 2: "):].strip()
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

with open("../../exp/generation/filtered_data_gpt4.json", "w") as f1: json.dump(filter_list, f1, indent=4)