# Uploads the UnlearnCanvas style/object classifier checkpoints
# (UnlearnCanvas/ckpts/cls_model/*.pth) to a Hugging Face model repo, so
# download_uc_classifiers.py can fetch them on the cluster.
# Needs `pip install huggingface_hub` and a write token (`huggingface-cli login`
# or HF_TOKEN).

import os
import argparse

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default="jlbaker361/unlearncanvas-classifiers")
parser.add_argument("--src_dir", type=str, default="UnlearnCanvas/ckpts/cls_model")
parser.add_argument("--files", nargs="*", default=["style50-001.pth", "style50_cls.pth", "style60.pth"])
parser.add_argument("--public", action="store_true", help="create the repo public (default: private)")

if __name__ == "__main__":
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    for name in args.files:
        path = os.path.join(args.src_dir, name)
        if not os.path.exists(path):
            print(f"skipping {path}: not found")
            continue
        print(f"uploading {path} ({os.path.getsize(path) / 1e9:.1f} GB) -> {args.repo_id}/{name}")
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=args.repo_id, repo_type="model")
    print("all done!")
