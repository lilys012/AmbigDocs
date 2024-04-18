import json
import sys

mode = int(sys.argv[1])
model_name = sys.argv[2]
with open(f"../../exp/eval/test_m{mode}_{model_name}.json", "r") as f1: train = json.load(f1)

res = {'data': []}

for ex in train:
    question, gen_ans = ex["question"], ex["gen_answer"]
    gen_ans = ' '.join(gen_ans.replace("\n", " ").split())
    if mode == 5: gen_ans = gen_ans.split("Question:")[0]
    fromp = ex["from"].strip().lower()
    if fromp[len(fromp)-len("(disambiguation)"):] == "(disambiguation)": fromp = fromp[:len(fromp)-len("(disambiguation)")].strip()
    idx = question.lower().find(fromp)
    used_ids = range(min(len(ex["documents"]), 5))

    for i, uid in enumerate(used_ids):
        question_sub = question
        if idx != -1: question_sub = question[:idx]+ex["documents"][uid]["title"]+question[idx+len(fromp):]

        res['data'].append({
            'context': gen_ans,
            'id': str(ex["pid"])+"_"+str(i),
            'question': question_sub,
            'answers': {'text': [ex["documents"][uid]["answer"]], 'answer_start': []}
        })

with open(f'../../exp/eval/df1/test_m{mode}_{model_name}_qa.json', 'w') as fid: json.dump(res, fid, indent=4)