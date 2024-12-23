"""Train a language model using TorchTune.

```
modal run train_torchtune_full.py --config /path/to/config.yaml
```
"""
import os
import modal
import yaml

DATASET_VOLUME_NAME = "dataset"
MODEL_VOLUME_NAME = "llm-weights"

torchtune_image = modal.Image\
    .debian_slim(python_version="3.12")\
    .apt_install("git")\
    .pip_install(["torch", "torchvision", "torchao", "wandb", "torchtune@git+https://github.com/pytorch/torchtune.git@c2c6f4a5236ba69a8c87dcb1f23ad65daf6e75de"])


app = modal.App(f"torchtune-training")
trained_model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
N_GPUS = int(os.environ.get("N_GPUS", 2))
N_HOURS = int(os.environ.get("N_HOURS", 10))

@app.function(
    image=torchtune_image,
    # gpu=modal.gpu.A100(count=N_GPU, size="80GB"),
    gpu=modal.gpu.H100(count=N_GPUS),
    volumes={
        "/llm-weights": trained_model_volume,
        "/dataset": dataset_volume,
    },
    timeout=N_HOURS * HOURS,
    secrets=[
        modal.Secret.from_name("xingyaoww-wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
)
def run_train(config_name: str, config: dict, n_gpus: int):
    config_path = f"/tmp/{config_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    command = f"tune run --nnodes 1 --nproc_per_node {n_gpus} full_finetune_distributed --config {config_path}"
    import subprocess
    import sys
    subprocess.run(
        command.split(),
        stdout=sys.stdout, stderr=sys.stderr,
        check=True,
    )
    trained_model_volume.commit()


@app.local_entrypoint()
def main(config: str):
    # load yaml config
    config_name = os.path.basename(config)
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    run_train.remote(config_name=config_name, config=config, n_gpus=N_GPUS)
