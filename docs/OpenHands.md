# Reproduce SWE-Gym Results with OpenHands

## Prerequisite

**For agent trajectory rollouts & evaluation**


1. `git clone https://github.com/SWE-Gym/OpenHands.git` Note that OpenHands is updated very often, and our results is obtained with this specific Fork. It's expected to observe notable performance difference if you are using the latest version.
2. Setup OpenHands on your workstation following [this](https://github.com/SWE-Gym/OpenHands/blob/main/Development.md#start-the-server-for-development)
3. Familarize yourself with how OpenHands performs SWE-Bench evaluation: [README](https://github.com/SWE-Gym/OpenHands/blob/main/evaluation/swe_bench/README.md)
4. (Recommended) Get access to OpenHands [`RemoteRuntime`](https://github.com/SWE-Gym/OpenHands/blob/main/evaluation/swe_bench/README.md#run-inference-on-remoteruntime-experimental). Check [this blog](https://www.all-hands.dev/blog/evaluation-of-llms-as-coding-agents-on-swe-bench-at-30x-speed) for a quick intro of `RemoteRuntime`.

**For model training**

1. Setup an [Modal](https://modal.com/) account
2. Understand [Modal basics](https://modal.com/docs/guide)

## Agent Trajectory Rollouts & Evaluation

You should have read the OpenHands [SWE-Bench instruction](https://github.com/SWE-Gym/OpenHands/blob/main/evaluation/swe_bench/README.md) and know how to configure `config.toml`.

### STEP 1

Setup your model in `config.toml`. For example, using a self-hosted open-source model:

```toml
[llm.my-oss-model]
model = "openai/my-oss-model"
api_key = "my-api-key"
base_url = "https://my-openai-compatible-endpoint/v1"
log_completions = true
temperature = 0.5 # or set to anything you want
```

Or proprietary model:

```toml
[llm.sonnet-20240620]
model = "anthropic/claude-3-5-sonnet-20240620"
api_key = "sk-ant-MYKEY"
temperature = 0.0
caching_prompt = true
```

### STEP 2

Export your `ALLHANDS_API_KEY` for the `RemoteRuntime`:

```bash
export ALLHANDS_API_KEY=ah-mykey
```

### STEP 3: Trajectory Rollout

You can modify values in the [script](https://github.com/SWE-Gym/OpenHands/blob/main/scripts/rollout-swe-train-full.sh) to change the max number of iteration and number of parallel workers.

```bash
cd OpenHands
# ./scripts/rollout-swe-train-full.sh <model_name> <your_exp_name> <optional:number_of_runs>
./scripts/rollout-swe-train-full.sh llm.my-oss-model my_exp_name 1

# OR you can use ./scripts/rollout-swe-train-lite.sh to rollout from SWE-Gym Lite
```

Once rollout is completed, you will find a folder that looks like `evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/my-oss-model_maxiter_50_N_v2.2-no-hint-my_exp_name-run_1`. Under it you will find an `output.jsonl` file.


### STEP 4: Evaluation

Then you can evaluate the SWE-Gym rollout you just generated with 32 workers using:

```bash
./scripts/eval-swetrain-full-rollout.sh evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/my-oss-model_maxiter_50_N_v2.2-no-hint-my_exp_name-run_1/output.jsonl 32
```

OR you can also rollout & evaluate your model on SWE-Bench (see OpenHands SWE-Bench instruction for details):

```bash
cd OpenHands
# ./scripts/eval-swebench-verified.sh <your_output_file> <num_workers>
./scripts/eval-swetrain-full-rollout.sh evaluation/evaluation_outputs/outputs/SWE-Bench__SWE-Bench_Verified-test/CodeActAgent/my-oss-model_maxiter_50_N_v2.2-no-hint-my_exp_name-run_1/output.jsonl 32
```

See disucssoins here on how to convert the trajectories to SFT format https://github.com/SWE-Gym/SWE-Gym/issues/9

# Training and Serving Models

## Training the Policy

1. Create a volume called `datasets` and `llm-weights`

2. Download your desired base model to `llm-weights`: `modal run scripts/modal_misc/download_checkpoint.py --source-repo Qwen/Qwen2.5-Coder-32B-Instruct --target-dir /llm-weights/Qwen/Qwen2.5-Coder-32B-Instruct`

3. Download [OpenHands SFT trajectory dataset from huggingface](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories), format it into JSONL format, then upload to `dataset`: `modal volume put dataset openhands-sft-trajectories.jsonl my_datasets`

4. Train the model. You should first understand the configuration used for full-parameter fine-tuning here: `scripts/training/openhands/configs/policy/1116-sonnet-4o-491i-32k-qwen25_coder_32b_full-lr1e-4.yaml` (the rest of the configs are for ablation studies, just for reference). Then you can use `N_GPUS=8 modal run scripts/training/openhands/train_torchtune_full.py --config scripts/training/openhands/configs/policy/1116-sonnet-4o-491i-32k-qwen25_coder_32b_full-lr1e-4.yaml` to start training your model.

## Training the Verifier

1. Follow step 1 & 2 above.

2. Download [OpenHands SFT trajectory dataset from huggingface](https://huggingface.co/datasets/SWE-Gym/OpenHands-Verifier-Trajectories), format it into JSONL format, then upload to `dataset`: `modal volume put dataset openhands-verifier-trajectories.jsonl my_datasets`

3. We use [unsloth](https://unsloth.ai/) to train the verifier for our main experiment: `N_GPUS=1 modal run scripts/training/openhands/train_unsloth_qwen25coder_32b_verifier.py`. Nevertheless, we include config files we used to train verifiers via torchtune in [this folder](../scripts/training/openhands/configs/verifiers).

## Model Serving

This will starts an OpenAI compatible server using 4 GPUs, running for max 4 hours.
The URL of the end point will be printed when the script starts running.

```bash
N_HOURS=4 N_GPUS=4 modal run --detach serve_sglang.py --model-path /llm-weights/my-oss-model --served-model-name my-oss-model --tokenizer-path /llm-weights/Qwen/Qwen2.5-Coder-32B-Instruct
```
