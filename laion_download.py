import datasets
import os
from diffusers.utils.loading_utils import load_image
from urllib.parse import urlparse
from tqdm import tqdm

FOREIGN_TLDS = (
    ".cn", ".ru", ".ir", ".kp", ".sy", ".de", ".fr", ".jp", ".uk"
)

def is_foreign(url):
    domain = urlparse(url).netloc.lower()
    return any(domain.endswith(tld) for tld in FOREIGN_TLDS)

save_dir = "laion"

data = datasets.load_dataset("laion/laion2B-en-aesthetic", split="train")

with open(os.path.join(save_dir, "info.csv"), "a") as file:
    for r, row in enumerate(tqdm(data, desc="Downloading")):
        path = os.path.join(save_dir, f"{r}.jpg")
        if os.path.exists(path):
            continue
        url = row["URL"]
        if is_foreign(url):
            continue
        aesthetic_score = row["aesthetic"]
        unsafe_score = row["punsafe"]
        try:
            image = load_image(url)
            image.save(path)
            file.write(f"{path},{aesthetic_score},{unsafe_score}\n")
        except Exception as e:
            print(f"Failed {url}: {e}")
