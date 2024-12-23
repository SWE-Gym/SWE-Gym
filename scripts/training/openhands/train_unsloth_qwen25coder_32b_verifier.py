import os
import json
import modal

unsloth_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    )
    .run_commands(
        'pip install --no-deps packaging ninja einops flash-attn "xformers<0.0.26" trl peft accelerate bitsandbytes'
    )
    .run_commands('pip install ipykernel "numpy<2"')
    .run_commands('pip install wandb')
)

DATASET_VOLUME_NAME = "dataset"
MODEL_VOLUME_NAME = "llm-weights"

trained_model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app = modal.App(f"unsloth-sft")


@app.function(
    image=unsloth_image,
    # gpu=modal.gpu.A100(count=1, size="80GB"),
    gpu=modal.gpu.H100(count=1),
    # gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=3 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={
        "/llm-weights": trained_model_volume,
        "/dataset": dataset_volume,
    },
    secrets=[
        modal.Secret.from_name("xingyaoww-wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)

def train(
    output_dir,
    exp_name,
    model_name,
    data_path,
    max_seq_length=10240,
    load_in_4bit=False,
    batch_size=1,
    grad_accum_steps=8,
    epochs=2,
    learning_rate=2e-4,
    lora_r=64,
    lora_alpha=64,
):
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq

    # save args to json
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        args_dict = {
            "output_dir": output_dir,
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "exp_name": exp_name,
        }
        json.dump(args_dict, f)

    # Model initialization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    # Data loading
    with open(data_path) as f:
        dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} samples from {data_path}")
    dataset = [D["messages"] for D in dataset]
    dataset = Dataset.from_dict({"conversations": dataset})
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Model configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            warmup_steps=15,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=os.path.join(output_dir, exp_name),
            report_to="wandb",
            run_name=exp_name,
            save_strategy="epoch",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Training stats and execution
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save models
    model.save_pretrained(os.path.join(output_dir, exp_name, "adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, exp_name, "adapter"))
    model.save_pretrained_merged(
        os.path.join(output_dir, exp_name, f"merged"),
        tokenizer,
        save_method="merged_16bit",
    )


@app.local_entrypoint()  # defining a CLI entrypoint
def main():
    data_path = "/dataset/xingyaoww/2024-11-16-sonnet-4o-491instances/rm_conv_32k_claude+4o+onpolicy_1318i_negative_1x_1318i.jsonl"
    exp_name = "RM-claude+4o+onpolicy-1x-1116-sonnet-4o-491i-32k-qwen25_coder_32b_base_full-lr1e-4-lora-r64-a64"
    output_dir = f'/llm-weights/xingyaoww/outputs/{exp_name}'
    model_path = '/llm-weights/Qwen/Qwen2.5-Coder-32B'
    print(f"Running training with exp_name={exp_name}, output_dir={output_dir}, model_path={model_path}, data_path={data_path}")

    train.remote(
        output_dir=output_dir, exp_name=exp_name, model_name=model_path, data_path=data_path
    )
