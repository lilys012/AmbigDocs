import json
import src.data
from tqdm import tqdm

print("Loading passages...")
passages = "../../data/psgs_w100.tsv"
passages = src.data.load_passages(passages)

print("Indexing passages...")
temp_ttoi = {}
for i, p in enumerate(tqdm(passages)):
    if p['title'] not in temp_ttoi: temp_ttoi[p['title']] = []
    temp_ttoi[p['title']].append(i)

ttoi = {}
for key, item in temp_ttoi.items():
    if len(item) <= 20: ttoi[key] = item[:len(item)-1] # under 20 passages
del temp_ttoi
print(len(ttoi))

print("Gathering disambiguation pages...")
with open("../../exp/generation/disamb_pages.json", "r") as f1: temp_disamb = json.load(f1)

disamb = {}
for key, item in temp_disamb.items():
    if len(item) >= 2 and len(item) <= 10: disamb[key] = item # under 10 resolutions
del temp_disamb
print(len(disamb))

def ngram(s1, s2):
    s1_chunks, s2_chunks = s1.split(), s2.split()
    lcs_matrix = [[0] * (len(s2_chunks) + 1) for _ in range(len(s1_chunks) + 1)]
    max_length, end_position = 0, 0

    for i in range(1, len(s1_chunks) + 1):
        for j in range(1, len(s2_chunks) + 1):
            if s1_chunks[i - 1] == s2_chunks[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
                if lcs_matrix[i][j] > max_length:
                    max_length = lcs_matrix[i][j]
                    end_position = i
            else: lcs_matrix[i][j] = 0

    if max_length == 0: return ""
    else:
        start_position = end_position - max_length
        return ' '.join(s1_chunks[start_position:end_position])

def is_similar(pas_i, pas_j): 
    tok_i, tok_j = pas_i.lower().split(), pas_j.lower().split()
    if len(set(tok_i) & set(tok_j)) / len(set(tok_i) | set(tok_j)) > 0.4: return "" # too many word overlap

    return ngram(pas_i.lower(), pas_j.lower())

print("Finding similar pages...")
ret_list = []
total = 0
for page, titles in tqdm(disamb.items()):
    page_list = {}
    for i in range(len(titles)): # titles are from same disamb page.title()
        if titles[i] in ttoi:
            pas_is = ttoi[titles[i]] # passages of titles[i]
            for j in range(i+1, len(titles)):
                if titles[j] in ttoi:
                    pas_js = ttoi[titles[j]] # passages of titles[j]
                    for k in range(len(pas_is)):
                        for m in range(len(pas_js)):
                            overlap = is_similar(passages[pas_is[k]]["text"], passages[pas_js[m]]["text"])
                            if len(overlap.split()) > 3 and len(overlap.split()) < 10:
                                if not len(page_list) or (len(overlap.split()) > len(page_list["overlap"].split())):
                                    page_list = {
                                        "passages":[passages[pas_is[k]], passages[pas_js[m]]], "idx":[k, m], "overlap":overlap
                                    }
    if "overlap" in page_list and len(page_list["overlap"].split()): ret_list.append({"from":page, "passages":page_list})
    total += 1

with open(f"../../exp/generation/sim_pages.json", "w") as f2: json.dump(ret_list, f2, indent=4)
print(len(ret_list), total)
