import modal
import os
import shutil
from time import sleep

sglang_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("sglang[all]==0.3.6")
    .run_commands("pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/")
)

VOLUME_NAME = "llm-weights"
API_KEY = "xwang"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

try:
    volume = modal.Volume.lookup(VOLUME_NAME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model_to_volume.py")

N_GPUS = int(os.environ.get("N_GPUS", 2))
N_HOURS = float(os.environ.get("N_HOURS", 4))

app = modal.App(f"sglang-serve")
@app.function(
    image=sglang_image,
    gpu=modal.gpu.A100(count=N_GPUS, size="80GB"),
    # gpu=modal.gpu.H100(count=N_GPUS),
    container_idle_timeout=5 * MINUTES,
    timeout=int(N_HOURS * HOURS),
    allow_concurrent_inputs=1000,
    volumes={"/llm-weights": volume},
)
def run_server(model_path: str, served_model_name: str, tokenizer_path: str, context_length: int, n_gpus: int):
    import subprocess
    import sys
    from sglang.utils import (
        wait_for_server,
    )

    # first check if model_path has config.json, if not copy it from tokenizer_path
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Copying config.json from {tokenizer_path} to {model_path}")
        shutil.copy(os.path.join(tokenizer_path, "config.json"), os.path.join(model_path, "config.json"))
        # print the content of the config.json
        print("Content of the config.json:")
        with open(os.path.join(model_path, "config.json"), "r") as f:
            print(f.read())
    assert os.path.exists(os.path.join(model_path, "config.json")), f"config.json not found in {model_path}. os.listdir(model_path): {os.listdir(model_path)}"

    with modal.forward(3000, unencrypted=True) as tunnel:
        command = f"python -m sglang.launch_server --model-path {model_path} --tokenizer-path {tokenizer_path} --tp-size {n_gpus} --port 3000 --host 0.0.0.0 --served-model-name {served_model_name} --context-length {context_length} --api-key {API_KEY}"
        print("Server listening at", tunnel.url)
        subprocess.run(
            command.split(),
            stdout=sys.stdout, stderr=sys.stderr,
            check=True,
        )

@app.local_entrypoint()
def main(
    model_path: str,
    served_model_name: str,
    tokenizer_path: str = "/llm-weights/Qwen/Qwen2.5-Coder-32B-Instruct",
    context_length: int = 32768,
):
    print(f"Serving {model_path} on {served_model_name} with {N_GPUS} GPUs")
    print(f"Timeout: {N_HOURS} hours")
    run_server.remote(model_path, served_model_name, tokenizer_path, context_length, N_GPUS)
