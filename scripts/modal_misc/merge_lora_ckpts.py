"""Train a language model using TorchTune.

modal run merge_lora_ckpts.py --model-dir /path/to/model_dir --adapter-dir /path/to/adapter_dir
"""
import os
import shutil
import modal
from glob import glob

MODEL_VOLUME_NAME = "llm-weights"

app = modal.App(f"merge-lora-ckpts")
torchtune_image = modal.Image\
    .debian_slim(python_version="3.12")\
    .apt_install("git")\
    .pip_install(["torch", "torchvision", "torchao", "wandb", "torchtune@git+https://github.com/pytorch/torchtune.git@2b1ee6d31cf20d13acc7938c65613b6488601192"])\
    .pip_install(["transformers", "peft"])

trained_model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
@app.function(
    image=torchtune_image,
    volumes={
        "/llm-weights": trained_model_volume
    },
    timeout=1 * HOURS,
    secrets=[
        modal.Secret.from_name("huggingface-secret")
    ],
    cpu=16,
    memory=32768*4
)
def merge_ckpts(model_dir: str, adapter_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    import torch

    # Path to your LoRA adapter and model files
    adapter_path = adapter_dir  # Use your actual adapter file path here
    # Load the base model (replace 'llama' with the appropriate model name)
    base_model = AutoModelForCausalLM.from_pretrained(model_dir)
    # Load the tokenizer (replace if necessary)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the LoRA adapter configuration
    peft_config = PeftConfig.from_pretrained(adapter_path)

    # Load the adapter weights
    lora_model = PeftModel.from_pretrained(base_model, adapter_path, config=peft_config)

    # Move the model to the device (GPU/CPU)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # lora_model.to(device)

    # # Example usage
    # input_text = "Tell me a joke"
    # inputs = tokenizer(input_text, return_tensors='pt').to(device)

    # # Generate output from the model
    # output = lora_model.generate(**inputs)
    # print(tokenizer.decode(output[0], skip_special_tokens=True))

    # Save the merged model
    output_dir = os.path.join(adapter_dir, "lora_merged_model")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to {output_dir}")
    merged_model = lora_model.merge_and_unload()
    print(f"model merged and unload to cpu")
    merged_model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)



@app.local_entrypoint()
def main(model_dir: str, adapter_dir: str):
    merge_ckpts.remote(model_dir=model_dir, adapter_dir=adapter_dir)
