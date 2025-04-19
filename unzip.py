from glob import glob 
import tarfile

files = glob("/Users/Dr. Abdulrahman Mahmoud/ocr/WordScape/data/f9b4364d61802388478a7198/annotated/CC-MAIN-2023-06/multimodal/*.tar.gz")

for file in files:
    out_name = file.replace(".tar.gz", "")
    try:
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(out_name)
    except Exception as e:
        print(f"Failed to extract {file}: {e}")
        continue
    print(f"Extracted {file} to {out_name}")
