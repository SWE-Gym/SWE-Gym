# Reproduce SWE-Gym Results with MoatlessTools
All data and models are open-sourced at [huggingface.co/SWE-Gym](https://huggingface.co/SWE-Gym).

## Prerequisite

**For agent trajectory rollouts & evaluation**

1. `git clone git@github.com:SWE-Gym/Moatless-Agent-Fork.git` and setup the environment following its README file.
2. Setup OpenHands on your workstation following [this](https://github.com/SWE-Gym/OpenHands/blob/main/Development.md#start-the-server-for-development). We uses OpenHands remote runtime for fast SWE-Gym unit-test evaluation.
3. Download VectorIndex for SWE-Gym Lite and SWE-Bench Lite from [here](https://huggingface.co/datasets/SWE-Gym/Codebase-Index-Lite)

## Agent Trajectory Rollouts & Evaluation

Use `Moatless-Agent-Fork/scripts/parallel_sample.sh` and `Moatless-Agent-Fork/scripts/eval_preds.sh` to rollout and evaluate your agent.

**For model serving**
We use [SGLang](https://github.com/sgl-project/sglang) for high throughput model serving. You can use `scripts/serving/serve_sglang.py` to serve fine-tuned models efficiently.

## Verifier and Test-time Scaling

Given N rounds of sampled trajectories, we can use the trained verifier to perform Best-of-N selection, enabling test-time scaling.

To inspect the 7B policy rollouts used for the SWE-bench Lite scaling experiment (Figure 7), download the [eight-run archive at temperature 0.5](https://huggingface.co/datasets/SWE-Gym/MoatlessTools-Sampled-Trajectories/blob/51ba9a7f38fed087bcb1a11b3d51972ed72a8317/7b/release-bench-lite-qwen7b-v2-it1-top2-fixed-temp_0.5-ep2-fp8_8runs.zip). Each run directory contains a `preds.jsonl` with 300 records. Group the eight runs by `instance_id` and count the tasks with at least one `report.resolved` value of `true` to calculate Pass@8. For Best@8, rank the submitted patches using the verifier before reading the evaluation labels.

Parse the MoatlessAgent trajectory into verifier's format [parse_to_orm_data.py](../scripts/moatless-verifier/parse_to_orm_data.py)

Curate verifier's training data: [curate_dataset.ipynb](../scripts/moatless-verifier/curate_dataset.ipynb)

Run Best-of-N selection and evaluate the agent-verifier performance: [eval_verifier.ipynb](../scripts/moatless-verifier/eval_verifier.ipynb)
