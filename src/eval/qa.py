import json
from tqdm import tqdm
import sys
import random
from openai import OpenAI
import torch
import re
import string
from transformers import LlamaTokenizer, LlamaForCausalLM

data_path = sys.argv[1]
mode = int(sys.argv[2])
model = sys.argv[3]
cntans = 0

print(mode, model)

if "gpt" in model:
    api_key = sys.argv[4]
    client = OpenAI(api_key=api_key)
else:
    model_path = sys.argv[4]
    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token='[PAD]'
    infer_model = LlamaForCausalLM.from_pretrained(model_path)
    infer_model = infer_model.half()
    device = torch.device("cuda")
    infer_model.to(device)

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def doPush(docs, passage):
    idx = passage["id"]

    for i in range(len(docs)):
        if idx == docs[i]["pid"]: return False 
    return True

def to_prompt(ex, mode):
    prompt = ""
    n_docs = 5
    global cntans

    if mode == 1 or mode == 2 or mode == 4: 
        docs = random.sample(ex["documents"], min(n_docs, len(ex["documents"])))
        if mode == 2:
            i = 0
            while len(docs) < n_docs:
                if doPush(docs, ex["retrieved"][i]): docs.append(ex["retrieved"][i])
                i += 1
    elif mode == 3:
        docs = ex["retrieved"][:n_docs]
    
    docs = random.sample(docs, len(docs))    
    for i in range(len(docs)): prompt += f"Context: " + docs[i]["title"] + f" | " + docs[i]["text"] + "\n"
    return prompt, docs

with open(data_path, "r") as f1: test = json.load(f1)

few_prompt = '''Question: What is the location of the Griswold House?
Context: Griswold House (Guilford, Connecticut) | a limited number of days each week. The Griswold House is located east of Guilford Center, on the south side of Boston Street (Connecticut Route 146) at its junction with Lovers Lane. It is a 2-1/2 story wood frame structure, with a gabled roof, large central chimney, and clapboarded exterior. A leanto section to the rear gives the house a classic New England saltbox appearance. The main facade is three bays wide, with sash windows arranged symmetrically around the entrance. The entrance is flanked by pilasters and topped by a fully pedimented gable. The house was probably built around 1764
Context: John N. A. Griswold House | John N. A. Griswold House The John N. A. Griswold House is a National Historic Landmark at 76 Bellevue Avenue in Newport, Rhode Island. It is home to the Newport Art Museum and houses an art gallery. The home was built in 1864 by Richard Morris Hunt for John Griswold, an Old China Trade merchant and member of the Griswold Family. The house is one of the earliest American Stick\u2013style buildings and one of Hunt's first works in Newport. The house is an official project of Save America\u2019s Treasures. The Griswold House is a 2-1/2 story wood frame structure, set
Answer: There are two Griswold Houses mentioned in the passages. John N. A. Griswold House is located at 76 Bellevue Avenue in Newport, Rhode Island and Griswold House (Guilford, Connecticut) is located east of Guilford Center, on the south side of Boston Street (Connecticut Route 146) at its junction with Lovers Lane.

Question: Who is the playable character in Edna & Harvey?
Context: Edna & Harvey: The Breakout | the same vein as LucasArts' pre-1994 games created using the SCUMM engine. The game screen shows a two-dimensional cartoon world where Edna, the playable character, is incarcerated within a mental hospital. The bottom edge of the screen contains a list of verbs which must be clicked on before the player clicks on an item within the game world in order to manipulate it. When the player tries to combine or effect one item with another each combination produces a different response, unlike in most games of this type where an incorrect combination would result in a response of \"I don't
Context: Edna & Harvey: Harvey's New Eyes | Edna & Harvey: Harvey's New Eyes Edna & Harvey: Harvey's New Eyes, also known as Harvey's New Eyes, is a point-and-click adventure game created by Daedalic Entertainment. The game was released on October 16, 2012. \"Harvey's New Eyes\" is a sequel to \"\". The game is set in a 2D cartoon world. The player controls Lilli. Unlike most other adventure games, there are no clickable dialogues. After the player clicks on a character, a bar appears with some symbols indicating a subject. The game neither has a list with actions such as Walk, Look, Talk, Pick up, Use. If applicable,
Answer: In "Edna & Harvey: The Breakout" the playable character is Edna, and in "Edna & Harvey: Harvey's New Eyes," the playable character is Lilli.
'''

results = []
if model == "gpt4": test = test[:500]
for ex in tqdm(test):
    question = ex["question"]
    prompt, docs = to_prompt(ex, mode) 
    if mode == 4:
        gptprompt = few_prompt+"\n"+f'Question: {question}\n' + prompt + "Answer:"
        prompt = "[INST] Answer for the given question using only the provided context. [/INST]\n"+ few_prompt + f'\nQuestion: {question}\n' + prompt + "Answer:"
    else:
        gptprompt = f'Question: {question}\n' + prompt + "Answer:"
        prompt = "[INST] Answer for the given question using only the provided context." + f'\nQuestion: {question}\n' + prompt + "Answer: [/INST]"

    if "gpt" not in model:
        model_input = tokenizer([prompt], return_tensors="pt").to(device)
        model_output = infer_model.generate(**model_input, max_new_tokens=300, return_dict_in_generate=True, output_scores=True, do_sample=False)
        sequences = model_output.sequences
        transition_scores = infer_model.compute_transition_scores(model_output.sequences, model_output.scores, normalize_logits=True)

        decoded_output = tokenizer.decode(
            model_output["sequences"][0],
            skip_special_tokens=True,
        )  
        output_text = decoded_output[len(prompt):]
    else:
        instruction = "Answer for the given question using only the provided context."
        response = client.chat.completions.create(
            model = "gpt-4" if model == "gpt4" else "gpt-3.5-turbo",
            messages = [{"role":"system", "content":instruction}, {"role":"user", "content":gptprompt}]
        )
        output_text = response.choices[0].message.content.strip()
    
    ex["gen_answer"] = output_text
    ex["docs"] = docs
    results.append(ex)
with open(f"../../exp/eval/test_m{mode}_{model}.json", "w") as f1: json.dump(results, f1, indent=4)