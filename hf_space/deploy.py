"""Publish the Space to Hugging Face: `hf auth login` once, then run this.

The Space repository receives the contents of hf_space/ at its root and the
pseul/ package beside them, so `from pseul import PSEUL` in app.py resolves to
the same frozen core.py as the GitHub package. Nothing else is uploaded.

    python hf_space/deploy.py            # private Space (the default)
    python hf_space/deploy.py --public   # once the paper is accepted

Running it again updates the existing Space.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
NAME = "pseul"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true",
                        help="create the Space public; the default is private")
    args = parser.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("Not logged in to Hugging Face. Run `hf auth login` with a write token first.")
        return 1

    repo_id = f"{user}/{NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    private=not args.public, exist_ok=True)
    # create_repo leaves an existing Space's visibility alone, so it is set here
    # on every run; without this, --public would not publish a private Space
    api.update_repo_settings(repo_id, repo_type="space", private=not args.public)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(ROOT / "hf_space"),
                      ignore_patterns=["deploy.py", "__pycache__/*"],
                      commit_message="Update the Space app")
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(ROOT / "pseul"),
                      path_in_repo="pseul", ignore_patterns=["__pycache__/*"],
                      commit_message="Update the frozen PSEUL package")
    visibility = "public" if args.public else "private"
    print(f"Space updated ({visibility}): https://huggingface.co/spaces/{repo_id}")
    print("The first build takes a few minutes while the dependencies install.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
