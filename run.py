import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import logging
import random
import re
from datetime import datetime

import torch  # noqa: F401
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def randomly_save_to_file(completion, filename):
    if random.random() < 0.1:  # 10% chance
        os.makedirs("completion_samples", exist_ok=True)
        log_file = os.path.join("completion_samples", filename)
        with open(log_file, "a") as f:
            f.write("\n\n==============\n")
            f.write(completion)


def format_reward_func(completions, target, **kwargs):  # noqa: ARG001
    rewards = []
    for completion, _gt in zip(completions, target, strict=False):
        try:
            completion = "<think>" + completion
            randomly_save_to_file(completion, "completion_samples.txt")
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, **kwargs):
    nums = kwargs.get("nums")
    if nums is None:
        raise ValueError("nums is required")

    rewards = []
    for completion, gt, numbers in zip(completions, target, nums, strict=False):
        try:
            completion = "<think>" + completion
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue

            equation = match.group(1).strip()
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue

            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            result = eval(equation, {"__builtins__": None}, {})
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
                randomly_save_to_file(completion, "success_completion_samples.txt")
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def train_model(model_args: ModelConfig, training_args: GRPOConfig):
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Training parameters: {training_args}")

    if not model_args.model_name_or_path:
        raise ValueError("Tokenizer name or path is required")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=42).select(range(50000))  # type: ignore

    def generate_r1_prompt(numbers, target):
        r1_prefix = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
            },
            {
                "role": "user",
                "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.",
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>",
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),
            "target": target,
            "nums": numbers,
        }

    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    logger.info(
        f"*** Starting training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs***"
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True  # type: ignore
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # type: ignore

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card()
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Model saved ***")


def main():
    model_args = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        model_revision="main",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )

    training_args = GRPOConfig(
        # training
        output_dir="runs/qwen-2.5-3b-r1-countdown",
        seed=42,
        max_steps=10,  # 450,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5.0e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # GRPO
        beta=0.001,
        max_prompt_length=256,
        max_completion_length=1024,
        num_generations=8,
        # vllm
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,
        # model
        bf16=True,
        tf32=True,
        # logging
        logging_strategy="steps",
        logging_steps=2,
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=25,
        # hub
        push_to_hub=True,
        hub_strategy="every_save",
    )

    train_model(model_args, training_args)


if __name__ == "__main__":
    main()
