import json
import sys
from rouge_score import rouge_scorer
import re
import string
import collections
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

mode = int(sys.argv[1])
model_name = sys.argv[2]
with open(f"../../exp/eval/test_m{mode}_{model_name}.json", "r") as f1: test = json.load(f1)

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

def compute_exact(ans, gen_ans):
    return int(ans.lower() in gen_ans.lower())

def remove_stopwords(tokens):
    return [w for w in tokens if not w in stop_words]

def get_stopwords_tokens(a):
    return remove_stopwords(get_tokens(a))

def get_removed_tokens(a, fromp):
    gold_toks = get_stopwords_tokens(a)
    from_toks = get_stopwords_tokens(fromp[0])
    from_toks = list(set(from_toks))
    for ft in from_toks: 
        try: 
            while True: gold_toks.remove(ft)
        except: pass
    return gold_toks

def compute_f1(a_gold, a_pred, fromp=None):
    if fromp: 
        gold_toks = get_removed_tokens(a_gold, fromp)
        if len(gold_toks) == 0: return [compute_exact(a_gold, a_pred)]*3
    else: gold_toks = get_stopwords_tokens(a_gold)
    pred_toks = get_stopwords_tokens(a_pred)
        
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def is_overlap(ans, gen_ans, method, all_pas=""):
    if method == "EM": return compute_exact(ans, gen_ans)
    elif method == "R": return compute_f1(ans, gen_ans)[1]
    elif method == "F1": return compute_f1(ans, gen_ans)[2]
    elif method == "rougeL":
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(ans.lower(), gen_ans.lower())
        return scores['rougeL'][1] # recall
    elif method == "KP": return compute_f1(all_pas, gen_ans)[0]

res_dict = {}
n_docs = 5

for method in ["R", "Entity", "EWR", "KP"]:
    mean = []
    cpam = [0, 0, 0, 0, 0]

    for ex in test:
        question, gen_ans = ex["question"], ex["gen_answer"]
        gen_ans = ' '.join(gen_ans.replace("\n", " ").split())
        fromp = ex["ambiguous_entity"].strip().lower()
        if fromp[len(fromp)-len("(disambiguation)"):] == "(disambiguation)": fromp = fromp[:len(fromp)-len("(disambiguation)")].strip()
        idx = question.lower().find(fromp)

        if method == "KP":
            all_pas = "Answer for the given question using only the provided context.\nQuestion: " + question + "\n"
            for doc in ex["docs"]:
                all_pas += f"Context: " + doc["title"] + f" | " + doc["text"] + "\n"
            all_pas += "Answer:"
            mean.append(is_overlap(None, gen_ans, "KP", all_pas))
            continue

        used_ids = range(min(len(ex["documents"]), n_docs))

        resls, entls, ansls = [], [], []
        tls = [0, 0]
        if method == "Entity" or method == "EWR":
            for uid in used_ids:
                pas = ex["documents"][uid]
                ent = pas["title"].replace("(", "").replace(")", "")
                entls.append(compute_f1(ent, gen_ans, [fromp])[1])
            if method == "EWR":
                for i, uid in enumerate(used_ids):
                    ansls.append(compute_f1(ex["documents"][uid]["answer"], gen_ans)[1])
                    resls.append(ansls[i] * entls[i])

                    temp1 = get_stopwords_tokens(ex["documents"][uid]["answer"])
                    thval = 1 / len(temp1) if len(temp1) else 1.

                    temp = get_removed_tokens(ex["documents"][uid]["title"].replace("(", "").replace(")", ""), [fromp])
                    ent_thval = 1 / len(temp) if len(temp) else 1.

                    if ansls[i] >= thval and entls[i] >= ent_thval: tls[0] += 1
                    elif entls[i] < ent_thval:
                        if len(temp1) <= 2 and ansls[i] >= .5: tls[1] += 1
                        elif ansls[i] > thval: tls[1] += 1
            else: resls = entls
        else:
            for uid in used_ids:
                resls.append(is_overlap(ex["documents"][uid]["answer"], gen_ans, method))
        mean.append(sum(resls)/len(resls))

        clsfy = 0
        if tls[0] == 0 and tls[1] == 0: clsfy = 4
        elif tls[0] >= min(5, len(resls)): clsfy = 0
        elif tls[0] == 0 and tls[1] == 1: clsfy = 2
        elif tls[1] == 0 and tls[0] > 0: clsfy = 1
        else: clsfy = 3
        cpam[clsfy] += 1

    res_dict[method] = {}
    res_dict[method]["score"] = round(sum(mean)/len(test), 3)
    if method == "EWR":
        res_dict[method]["complete"] = round(cpam[0]/len(test), 3)
        res_dict[method]["partial"] = round(cpam[1]/len(test), 3)
        res_dict[method]["ambiguous"] = round(cpam[2]/len(test), 3)
        res_dict[method]["merged"] = round(cpam[3]/len(test), 3)
        res_dict[method]["failure"] = round(cpam[4]/len(test), 3)
    print(method, res_dict[method])

### DF1-eval
with open(f"../../exp/eval/df1/test_m{mode}_{model_name}_qa.json", "r") as f1: train = json.load(f1)
id2ans = {}
for ex in train["data"]: id2ans[ex["id"]] = ex["answers"]["text"][0]

with open(f"../../exp/eval/df1/test_m{mode}_{model_name}/eval_predictions.json", "r") as f1: pred = json.load(f1)
scores = {}
cnt = 0
for idx, gen_ans in pred.items():
    preid, _ = idx.split("_")
    if preid not in scores: 
        cnt += 1
        if mode >= 2 and model_name == "gpt4" and cnt > 500: break
        scores[preid] = []
    scores[preid].append(compute_f1(id2ans[idx], gen_ans)[2])

mean = []
for idx, f1s in scores.items(): 
    mean.append(sum(f1s)/len(f1s))

len_eval = len(scores.keys())
method = "DF1"
res_dict[method] = {}
res_dict[method]["score"] = round(sum(mean)/len_eval, 3)
print(res_dict[method])