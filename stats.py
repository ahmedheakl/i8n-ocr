from glob import glob
import json
from tqdm import tqdm
import seaborn as sns

path = "data/f9b4364d61802388478a7198/annotated/CC-MAIN-2023-06/multimodal"


files = glob(f"{path}/**/text*.json", recursive=True)
lang_dist = {}

unk_ok = False

def read(fp):
    with open(fp, "r") as f:
        data = json.load(f)
        return data

for file in tqdm(files):
    try:
        data = read(file)
    except json.JSONDecodeError:
        print(f"Error reading {file}")
        continue
    metadata = data["metadata"]
    
    languages = metadata["languages_fasttext"]
    max_lang = max(languages, key=lambda x: languages[x]).replace("__label__", "")
    if max_lang == "unknown": continue
    if max_lang not in lang_dist:
        lang_dist[max_lang] = 0
    if max_lang == "ar" and lang_dist[max_lang] > 100:
        print(file.replace("text_doc_", "doc_").replace(".json", ".jpg"))
    lang_dist[max_lang] += 1
print(f"Got {len(lang_dist)} languages with {sum(lang_dist.values()):,} files")


import matplotlib.pyplot as plt
import pandas as pd
lang_dist = {k: v for k, v in lang_dist.items()}
lang_dist = {k: v for k, v in sorted(lang_dist.items(), key=lambda item: item[1], reverse=True)}
df = pd.DataFrame.from_dict(lang_dist, orient="index").reset_index()
df.columns = ["language", "count"]
df = df.sort_values(by="count", ascending=False)
plt.figure(figsize=(20, 10))
sns.barplot(data=df, x="language", y="count")
plt.xticks(rotation=45)
plt.title("Language Distribution")
plt.xlabel("Language")
plt.ylabel("Count")
plt.savefig("lang_dist.png")
# plt.show()
