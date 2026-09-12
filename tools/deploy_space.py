"""Publish the demo to a Hugging Face Docker Space.

Stages exactly what the Space needs (engine modules, the two nets, `demo/`), puts the
Dockerfile and the Space card at the staging root where Hugging Face looks for them,
and uploads the folder. Needs a write token: `--token`, `HF_TOKEN`, or a cached login
from `hf auth login`.

    uv run python tools/deploy_space.py --repo-id <user>/aichessathon-engine
    uv run python tools/deploy_space.py --repo-id x/y --dry-run   # stage and list only
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = ("nnue.npz", "nnue-net5.npz")


def stage(destination: Path) -> list[str]:
    """Copy the Space's files under `destination`; return their relative paths."""
    written: list[str] = []
    for source in sorted(REPO.glob("*.py")):
        shutil.copy2(source, destination / source.name)
        written.append(source.name)
    (destination / "weights").mkdir()
    for name in WEIGHTS:
        source = REPO / "weights" / name
        if not source.is_file():
            raise SystemExit(f"missing {source}; export the net first")
        shutil.copy2(source, destination / "weights" / name)
        written.append(f"weights/{name}")
    for source in sorted((REPO / "demo").rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(REPO)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written.append(relative.as_posix())
    # Hugging Face reads both of these from the root of the Space repository.
    shutil.copy2(REPO / "demo" / "Dockerfile", destination / "Dockerfile")
    shutil.copy2(REPO / "demo" / "README.md", destination / "README.md")
    written += ["Dockerfile", "README.md"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the demo as a Hugging Face Space.")
    parser.add_argument("--repo-id", required=True, help="e.g. username/aichessathon-engine")
    parser.add_argument("--token", default=None, help="write token; defaults to HF_TOKEN/login")
    parser.add_argument("--dry-run", action="store_true", help="stage and list, do not upload")
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="space-") as temp:
        staging = Path(temp)
        written = stage(staging)
        total = sum((staging / name).stat().st_size for name in written)
        print(f"staged {len(written)} files, {total / 1e6:.1f} MB")
        for name in written:
            print(f"  {name}")
        if arguments.dry_run:
            return
        from huggingface_hub import HfApi  # imported late so the dry run needs no token

        token = arguments.token or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        url = api.create_repo(
            arguments.repo_id, repo_type="space", space_sdk="docker", exist_ok=True
        )
        api.upload_folder(
            repo_id=arguments.repo_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Deploy the playable engine demo",
        )
        print(f"uploaded; the Space builds at {url}")


if __name__ == "__main__":
    main()
