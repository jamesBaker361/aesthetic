import datasets
import os
from diffusers.utils.loading_utils import load_image
from urllib.parse import urlparse
from tqdm import tqdm

FOREIGN_TLDS = (
    ".cn", ".ru", ".ir", ".kp", ".sy", ".de", ".fr", ".jp", ".uk",".gr",".au",".by",".er",".cu"
)

def is_foreign(url):
    domain = urlparse(url).netloc.lower()
    return any(domain.endswith(tld) for tld in FOREIGN_TLDS)

save_dir = "laion"
os.makedirs(save_dir,exist_ok=True)

data = datasets.load_dataset("laion/laion2B-en-aesthetic", split="train")

count=len([f for f in os.listdir(save_dir) if f.endswith("jpg")])
session_count=0

with open(os.path.join(save_dir, "info.csv"), "a") as file:
    for r, row in enumerate(tqdm(data, desc="Downloading")):
        if r< count:
            continue
        path = os.path.join(save_dir, f"{r}.jpg")
        if os.path.exists(path):
            continue
        url = row["URL"]
        if is_foreign(url):
            continue
        aesthetic_score = row["aesthetic"]
        unsafe_score = row["punsafe"]
        try:
            image = load_image(url).convert("RGB")
            (h,w)=image.size
            if h<4 or w<4:
                print("hella small ",path)
            else:
                image.save(path)
                file.write(f"{path},{aesthetic_score},{unsafe_score}\n")
                session_count+=1
                if session_count%250==0:
                    print(f"processed {session_count} images")
        except Exception as e:
            print(f"Failed {url}: {e}")
