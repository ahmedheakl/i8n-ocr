from glob import glob
import datasets
from PIL import Image
from tqdm import tqdm
import os
import dotenv

dotenv.load_dotenv()   

data_dir = "output"
images_dir = "layouts"
doctag_html_dir = "doctag_html_outputs"
doctag_otsl_dir = "doctag_otsl_outputs"
markdown_dir = "markdown_outputs"
images = glob(f"{data_dir}/{images_dir}/*")


def read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""
data = {
    "id": [],
    "image": [],
    "doctag_html": [],
    "doctag_otsl": [],
    "markdown": [],
}
for img in tqdm(images):
    img_name = os.path.basename(img).replace(".jpg", "")
    data["id"].append(img_name)
    data["image"].append(Image.open(img))
    doctag_html = os.path.join(data_dir, doctag_html_dir, img_name + ".dt.xml")
    doctag_otsl = os.path.join(data_dir, doctag_otsl_dir, img_name + ".dt.xml")
    markdown = os.path.join(data_dir, markdown_dir, img_name + ".md")
    data["doctag_html"].append(read(doctag_html))
    data["doctag_otsl"].append(read(doctag_otsl))
    data["markdown"].append(read(markdown))
dataset = datasets.Dataset.from_dict(data)
dataset = dataset.cast_column("image", datasets.Image())
dataset = dataset.cast_column("doctag_html", datasets.Value("string"))
dataset = dataset.cast_column("doctag_otsl", datasets.Value("string"))
dataset = dataset.cast_column("markdown", datasets.Value("string"))
dataset = dataset.cast_column("id", datasets.Value("string"))

dataset.push_to_hub("ahmedheakl/i8n-ocr")