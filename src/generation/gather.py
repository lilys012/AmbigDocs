import json
import pywikibot
from tqdm import tqdm
import wikipediaapi

def get_pages(site):
    disamb = list(pywikibot.Category(site, 'Disambiguation pages').articles(recurse=0))
    disamb.extend(list(pywikibot.Category(site, 'Buildings and structures disambiguation pages').articles(recurse=2)))
    disamb.extend(list(pywikibot.Category(site, 'Disambiguation pages with given-name-holder lists').articles(recurse=0)))
    disamb.extend(list(pywikibot.Category(site, 'Language and nationality disambiguation pages').articles(recurse=0)))
    disamb.extend(list(pywikibot.Category(site, 'Military units and formations disambiguation pages').articles(recurse=0)))
    disamb.extend(list(pywikibot.Category(site, 'Place name disambiguation pages').articles(recurse=1)))
    disamb.extend(list(pywikibot.Category(site, 'Science disambiguation pages').articles(recurse=3)))
    disamb.extend(list(pywikibot.Category(site, 'Disambiguation pages with surname-holder lists').articles(recurse=0)))
    disamb.extend(list(pywikibot.Category(site, 'Title and name disambiguation pages').articles(recurse=2)))
    return disamb

site = pywikibot.Site('en', 'wikipedia')
disamb = get_pages(site)
disamb_titles = set()
for e in disamb: disamb_titles.add(e.title())
print(len(disamb_titles))

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MyProjectName (merlin@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

def findcolon(text):
    global skipped
    idx = text.find("refer to")
    if idx != -1: return text[idx+len("refer to"):]
    else:
        idx = text.find(":")
        if idx != -1 and idx+1 < len(text)-1:
            if len(text[idx+1:].strip()) and text[idx+1:].strip()[0] == "\n":
                return text[idx+1:].strip()
        skipped += 1
        return text

def isvalid(text): # starts with alphabet or number + alphabet
    text = text.strip()
    if text == "Help:Disambiguation": return False
    if text[0].isnumeric():
        for i in range(len(text)):
            if text[i].lower().isalpha(): return True
    return text[0].lower().isalpha()

inv_idx = {}
pro_idx = {}
err = []
skipped = 0
for page in tqdm(disamb):
    try:
        title = page.title()
        if not isvalid(title):
            skipped += 1
            continue
        if title in pro_idx: continue

        p_page = wiki_wiki.page(title)
        p_wiki = p_page.text
        sa_idx = p_wiki.find("See also")
        if sa_idx != -1: p_wiki = p_wiki[:sa_idx]
        p_wiki = findcolon(p_wiki, title)

        for lp in p_page.links.keys():
            if not isvalid(lp):
                skipped += 1
                continue
            lp_idx = p_wiki.find(lp)
            if lp_idx <= 0: continue
            if p_wiki[lp_idx-1:lp_idx] == "\n":
                if lp not in inv_idx: inv_idx[lp] = []
                inv_idx[lp].append(title)
                if title not in pro_idx: pro_idx[title] = []
                pro_idx[title].append(lp)

        if "(disambiguation)" in title:
            rdtitle = title[:len(title)-len("(disambiguation)")].strip()
            page_py = wiki_wiki.page(rdtitle)
            if page_py.exists(): 
                if rdtitle not in inv_idx: inv_idx[rdtitle] = []
                inv_idx[rdtitle].append(title)
                if title not in pro_idx: pro_idx[title] = []
                pro_idx[title].append(lp)
    except:
        err.append(title)

print(len(inv_idx), len(pro_idx), skipped)
print(err)

with open("../../exp/generation/disamb_pages_inverse.json", "w") as f1:
    json.dump(inv_idx, f1, indent=4)
with open("../../exp/generation/disamb_pages.json", "w") as f1:
    json.dump(pro_idx, f1, indent=4)