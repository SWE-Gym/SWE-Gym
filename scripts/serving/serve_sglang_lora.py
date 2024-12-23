import modal
from time import sleep

sglang_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("sglang[all]", "IPython")
    .run_commands("pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/")
)

VOLUME_NAME = "llm-weights"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

try:
    volume = modal.Volume.lookup(VOLUME_NAME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model_to_volume.py")

app = modal.App(f"sglang-serve")
@app.function(
    image=sglang_image,
    gpu=modal.gpu.H100(count=1),
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={"/llm-weights": volume},
)
def run_server(
    lora_path: str,
    served_model_name: str,
    model_path: str,
    tokenizer_path: str,
):
    import subprocess
    import sys
    from sglang.utils import (
        wait_for_server,
    )

    with modal.forward(3000, unencrypted=True) as tunnel:
        command = f"python -m sglang.launch_server --model-path {model_path} --tokenizer-path {tokenizer_path} --port 3000 --host 0.0.0.0 --quantization fp8  --served-model-name {model_path.split('/')[-1]} --lora-paths {served_model_name}={lora_path} --disable-cuda-graph --disable-radix-cache"
        subprocess.run(
            command.split(),
            stdout=sys.stdout, stderr=sys.stderr,
            check=True,
        )
        wait_for_server("http://localhost:3000")
        print("Server is running!")
        while True:
            sleep(5)
            print("Server listening at", tunnel.url)

@app.local_entrypoint()
def main(
    served_model_name: str,
    lora_path: str,
    model_path: str = "/llm-weights/Qwen/Qwen2.5-Coder-32B-Instruct",
    tokenizer_path: str = "/llm-weights/Qwen/Qwen2.5-Coder-32B-Instruct",
):
    run_server.remote(lora_path, served_model_name, model_path, tokenizer_path)
