from glob import glob
import datasets
from PIL import Image
from tqdm import tqdm
import os
import dotenv
import json

dotenv.load_dotenv()   

data_dir = "output"
images_dir = "layouts"
doctag_html_dir = "doctag_html_outputs"
doctag_otsl_dir = "doctag_otsl_outputs"
markdown_dir = "markdown_outputs"
languages_dir = "languages"
images = glob(f"{data_dir}/{images_dir}/*")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

data = {
    "id": [],
    "image": [],
    "doctag_html": [],
    "doctag_otsl": [],
    "markdown": [],
    "language": [],
    "language_confidence": [],
}

collected_langs = {}
for img in tqdm(images):
    img_name = os.path.basename(img).replace(".jpg", "")
    image = Image.open(img)
    try:
        doctag_html = read(os.path.join(data_dir, doctag_html_dir, img_name + ".dt.xml"))
        doctag_otsl = read(os.path.join(data_dir, doctag_otsl_dir, img_name + ".dt.xml"))
        markdown = read(os.path.join(data_dir, markdown_dir, img_name + ".md"))
        language_json = read_json(os.path.join(data_dir, languages_dir, img_name + ".json"))
    except Exception as e:
        print(f"Error reading files for {img_name}: {e}")
        continue
    language = language_json.get("language", "unknown")
    if language not in collected_langs:
        collected_langs[language] = 0
    if collected_langs[language] > 10:
        continue
    language_conf = language_json.get("confidence", 1.0)
    data["id"].append(img_name)
    data["image"].append(image)
    data["doctag_html"].append(doctag_html)
    data["doctag_otsl"].append(doctag_otsl)
    data["markdown"].append(markdown)
    data['language'].append(language)
    data['language_confidence'].append(language_conf)
    collected_langs[language] = collected_langs.get(language, 0) + 1
dataset = datasets.Dataset.from_dict(data)
dataset = dataset.cast_column("image", datasets.Image())
dataset = dataset.cast_column("doctag_html", datasets.Value("string"))
dataset = dataset.cast_column("doctag_otsl", datasets.Value("string"))
dataset = dataset.cast_column("markdown", datasets.Value("string"))
dataset = dataset.cast_column("id", datasets.Value("string"))

dataset.push_to_hub("ahmedheakl/i8n-ocr-bench")
