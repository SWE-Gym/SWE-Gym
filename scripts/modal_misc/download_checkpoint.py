"""Download a checkpoint from Hugging Face.

modal run download_checkpoint.py --source-repo /path/to/source_repo --target-dir /path/to/target_dir

Example:
modal run download_checkpoint.py --source-repo meta-llama/Llama-3.3-70B-Instruct --target-dir /llm-weights/meta-llama/Llama-3.3-70B-Instruct
"""

import modal
import os

MODEL_VOLUME_NAME = "llm-weights"

app = modal.App(f"download-hf-ckpts")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

image = modal.Image\
    .debian_slim(python_version="3.12")\
    .apt_install(["git", "git-lfs"])\
    .pip_install("huggingface_hub[cli]")


MINUTES = 60  # seconds
HOURS = 60 * MINUTES
@app.function(
    volumes={
        "/llm-weights": model_volume
    },
    image=image,
    timeout=1 * HOURS,
    secrets=[
        modal.Secret.from_name("xingyaoww-huggingface-secret")
    ],
)
def download_ckpts(source_repo: str, target_dir: str):
    # make sure target_dir exists
    os.makedirs(target_dir, exist_ok=True)

    import subprocess
    import sys
    command = f"git lfs install"
    subprocess.run(
        command.split(),
        stdout=sys.stdout, stderr=sys.stderr,
        check=True,
    )

    command = f"huggingface-cli download {source_repo} --local-dir {target_dir}"
    subprocess.run(
        command.split(),
        stdout=sys.stdout, stderr=sys.stderr,
        check=True,
    )
    model_volume.commit()


@app.local_entrypoint()
def main(source_repo: str, target_dir: str):
    download_ckpts.remote(source_repo=source_repo, target_dir=target_dir)
