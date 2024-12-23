from tqdm import tqdm
import os
from utils import (
    get_before_after_code_with_context,
    organize_identified_spans
)
from tqdm import tqdm
import json
from prompts import SYSTEM_PROMPT, DEI_PROMPT_TEMPLATE, SIMPLE_PROMPT_TEMPLATE

from utils import extract_spans_from_trajectory

from datasets import load_dataset

def to_simple_orm_dataset(exp_name, repo_base_dir, dataset, id_to_instance):
    preds = [json.loads(line) for line in open(exp_name + "/preds.jsonl")]
    evals = [json.loads(line) for line in open(exp_name + "/preds.swebench_eval.jsonl")]
    id_to_eval = {eval["instance_id"]: eval for eval in evals}
    instance = dataset[0]
    pred_messages = {}
    for pred in tqdm(preds):
        try:
            instance = id_to_instance[pred["instance_id"]]
            spans = extract_spans_from_trajectory(exp_name + "/trajs", instance, repo_base_dir)
            concat_spans = organize_identified_spans(spans)
            before, after = get_before_after_code_with_context(instance, pred["model_patch"], repo_base_dir, context_window=5)
            issue_text = instance["problem_statement"]
            formatted_user_input = SIMPLE_PROMPT_TEMPLATE.format(
                    issue_text=issue_text,
                    code_spans=concat_spans,
                    before_patch=before,
                    after_patch=after
                )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_user_input},
                {"role": "assistant", "content": str(id_to_eval[pred["instance_id"]]['test_result']['report']["resolved"])}
            ]
            pred_messages[pred["instance_id"]] = messages
        except Exception as e:
            print(f"Error processing instance {pred['instance_id']}: {e}")
    return pred_messages

def main():
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    exp_prefix = "EXP_PREFIX"
    id_to_instance = {instance["instance_id"]: instance for instance in dataset}
    import glob
    exp_names = glob.glob(exp_prefix + "*")
    repo_base_dir = "EXP_REPO_BASE_DIR"
    out_dir = "EXP_OUTPUT_DIR"
    for exp_name in exp_names:
        out_file = out_dir + "/" + exp_name.split("/")[-1] + ".json"
        if os.path.exists(out_file):
            print(f"Skipping {exp_name} because output filealready exists")
            continue
        pred_messages = to_simple_orm_dataset(exp_name, repo_base_dir, dataset, id_to_instance)
        with open(out_file, "w") as f:
            json.dump(pred_messages, f)

if __name__ == "__main__":
    main()
