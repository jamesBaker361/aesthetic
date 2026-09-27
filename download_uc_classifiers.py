# Downloads the UnlearnCanvas style/object classifier checkpoints uploaded by
# upload_uc_classifiers.py into UnlearnCanvas/ckpts/cls_model, where
# evaluate_unlearncanvas.py looks for them by default. Files already there
# are skipped. A private repo needs a read token (`huggingface-cli login` or
# HF_TOKEN).

import os
import argparse

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default="jlbaker361/unlearncanvas-classifiers")
parser.add_argument("--dest_dir", type=str, default="UnlearnCanvas/ckpts/cls_model")
parser.add_argument("--files", nargs="*", default=["style50-001.pth", "style50_cls.pth"],
                    help="style60.pth is also uploaded but unused by evaluate_unlearncanvas.py")

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)
    for name in args.files:
        if os.path.exists(os.path.join(args.dest_dir, name)):
            print(f"{name} already in {args.dest_dir}")
            continue
        print(f"downloading {args.repo_id}/{name}")
        path = hf_hub_download(repo_id=args.repo_id, filename=name, repo_type="model", local_dir=args.dest_dir)
        print(f"  -> {path}")
    print("all done!")
