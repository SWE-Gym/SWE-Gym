"""
Infer RM scores using SGLang.

N_GPUS=2 modal run infer_rm_sglang.py --infer-jsonl-path /dataset/xingyaoww/rm_exp/qwencoder-32b-1116-sonnet-4o-491i-32k-8+1runs-rm_prompt.jsonl --model-path /llm-weights/xingyaoww/outputs/RM-1x-1116-sonnet-4o-491i-32k-qwen25_coder_32b_base_full-lr1e-4/epoch_4 --tokenizer-path /llm-weights/Qwen/Qwen2.5-Coder-32B-Instruct --context-length 32768
"""

import modal
import os
import shutil
import json
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

sglang_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("sglang[all]==0.3.6")
    .run_commands("pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/")
)

DATASET_VOLUME_NAME = "dataset"
LLM_WEIGHTS_VOLUME_NAME = "llm-weights"

dataset_volume = modal.Volume.lookup(DATASET_VOLUME_NAME, create_if_missing=False)
llm_weights_volume = modal.Volume.lookup(LLM_WEIGHTS_VOLUME_NAME, create_if_missing=False)

N_GPUS = int(os.environ.get("N_GPUS", 2))
N_HOURS = float(os.environ.get("N_HOURS", 4))

app = modal.App(f"sglang-serve")
@app.function(
    image=sglang_image,
    gpu=modal.gpu.H100(count=N_GPUS),
    container_idle_timeout=5 * MINUTES,
    timeout=int(N_HOURS * HOURS),
    allow_concurrent_inputs=1000,
    volumes={"/dataset": dataset_volume, "/llm-weights": llm_weights_volume},
)
def infer_sglang(infer_jsonl_path: str, model_path: str, tokenizer_path: str, context_length: int, n_gpus: int):
    import sglang as sgl
    import asyncio
    import pandas as pd

    model_name = '__'.join(model_path.rstrip('/').split('/')[-2:])
    output_jsonl_path = infer_jsonl_path.replace(".jsonl", f".{model_name}.output.jsonl")
    print(f"Loading data from {infer_jsonl_path} and writing to {output_jsonl_path}.")

    assert os.path.exists(infer_jsonl_path), f"File {infer_jsonl_path} does not exist"
    df = pd.read_json(infer_jsonl_path, lines=True, orient='records')
    print("Loaded dataframe with shape", df.shape)

    # First check if model_path has config.json, if not copy it from tokenizer_path
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Copying config.json from {tokenizer_path} to {model_path}")
        shutil.copy(os.path.join(tokenizer_path, "config.json"), os.path.join(model_path, "config.json"))
        # print the content of the config.json
        print("Content of the config.json:")
        with open(os.path.join(model_path, "config.json"), "r") as f:
            print(f.read())

    # Load the model
    runtime = sgl.Runtime(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        tp_size=n_gpus,
    )
    sgl.set_default_backend(runtime)

    @sgl.function
    def score_trajectory(s, rm_prompt, **kwargs):
        s += rm_prompt
        s += sgl.gen(
                "answer",
                choices=[" YES", " NO"],
                return_logprob=True,
                choices_method=sgl.token_length_normalized,
            )

    # load existing output jsonl
    outputs = []
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "r") as f:
            outputs = [json.loads(line) for line in f]
            print("Loaded existing dataframe with shape", len(outputs))
    
    # Filter out the outputs that have already been scored
    processed_ids = set(output['directory']+output['instance_id'] for output in outputs)
    df = df[~(df['directory']+df['instance_id']).isin(processed_ids)]
    print("Filtered out", len(processed_ids), "completed instances")
    print("Remaining instances to score:", len(df))

    df_to_score = df.to_dict(orient='records')
    states = score_trajectory.run_batch(
        df_to_score,
        progress_bar=True
    )
    output_fhandle = open(output_jsonl_path, "a")
    try:
        for idx, s in enumerate(states):
            # print("Attributes of state", dir(s))
            print("Meta info of state", s.get_meta_info("answer"))
            row = df_to_score[idx]
            row['rm_score'] = s.get_meta_info("answer")
            output_fhandle.write(json.dumps(row) + "\n")
            if idx % 100 == 0:
                output_fhandle.flush()
                dataset_volume.commit()
    except Exception as e:
        print("Error in infer_sglang", e)
        raise e
    finally:
        dataset_volume.commit()
        output_fhandle.close()
        runtime.shutdown()


@app.local_entrypoint()
def main(
    infer_jsonl_path: str,
    model_path: str,
    tokenizer_path: str = "/llm-weights/Qwen/Qwen2.5-Coder-32B-Instruct",
    context_length: int = 32768,
):
    infer_sglang.remote(infer_jsonl_path, model_path, tokenizer_path, context_length, N_GPUS)
