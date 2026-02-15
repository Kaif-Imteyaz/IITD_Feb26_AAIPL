#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for AAIPL Hackathon
Using Unsloth for AMD ROCm (MI300X with 192GB HBM3)
Model: Qwen/Qwen2.5-14B-Instruct
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset

torch.manual_seed(42)


@dataclass
class TrainConfig:
    """Configuration for SFT training"""
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    data_dir: str = "./data"
    output_dir: str = "./sft_checkpoints"

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # Training settings
    max_seq_length: int = 2048
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4

    # Agent type
    agent_type: str = "question"


def find_model_path(model_name: str) -> str:
    """Find valid model path"""
    import glob

    possible_paths = [
        "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots",
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots",
        "./hf_models/Qwen2.5-14B-Instruct",
    ]

    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
        if os.path.exists(os.path.join(base_path, "config.json")):
            return base_path
        for subdir in glob.glob(os.path.join(base_path, "*")):
            if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "config.json")):
                print(f"Found model at: {subdir}")
                return subdir

    return model_name


def load_training_data(data_dir: str) -> List[Dict]:
    """Load training data from data folder"""
    data_path = Path(data_dir)
    all_data = []

    json_files = list(data_path.glob("*.json")) + list(data_path.glob("**/*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    # Filter to only include dicts
                    for item in content:
                        if isinstance(item, dict):
                            all_data.append(item)
                elif isinstance(content, dict):
                    # Could be {topic: [samples]} or a single sample
                    if "question" in content:
                        all_data.append(content)
                    else:
                        for key, value in content.items():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, dict):
                                        if "topic" not in item:
                                            item["topic"] = key
                                        all_data.append(item)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    # Also load from assets/topics_example.json
    examples_path = Path("assets/topics_example.json")
    if examples_path.exists():
        with open(examples_path) as f:
            examples = json.load(f)
            for topic, samples in examples.items():
                for sample in samples:
                    sample["topic"] = topic
                    all_data.append(sample)

    print(f"Loaded {len(all_data)} samples")
    return all_data


def format_for_question_agent(data: List) -> List[Dict]:
    """Format data for question generation training"""
    formatted = []

    for sample in data:
        # Skip if not a dict
        if not isinstance(sample, dict):
            continue

        topic = sample.get("topic", "General")
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        answer = sample.get("answer", sample.get("expected_answer", ""))
        explanation = sample.get("explanation", "")

        if not question or not choices:
            continue

        user_content = f"""Generate an EXTREMELY DIFFICULT MCQ on topic: {topic}.

Requirements:
1. Generate exactly FOUR choices labeled "A)", "B)", "C)", "D)".
2. Only ONE option should be correct.
3. Include brief explanation (under 100 words).

Output valid JSON with: topic, question, choices, answer, explanation"""

        assistant_content = json.dumps({
            "topic": topic,
            "question": question,
            "choices": choices,
            "answer": answer[0] if len(answer) > 0 else "A",
            "explanation": explanation
        }, indent=2)

        formatted.append({
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    return formatted


def format_for_answer_agent(data: List) -> List[Dict]:
    """Format data for answer generation training"""
    formatted = []

    for sample in data:
        # Skip if not a dict
        if not isinstance(sample, dict):
            continue

        question = sample.get("question", "")
        choices = sample.get("choices", [])
        answer = sample.get("answer", sample.get("expected_answer", ""))
        explanation = sample.get("explanation", "")

        if not question or not choices:
            continue

        choices_str = " ".join(choices)

        user_content = f"""Question: {question}
Choices: {choices_str}

Select the correct answer and provide brief reasoning (under 100 words).
Output valid JSON with: answer, reasoning"""

        ans_letter = answer[0].upper() if len(answer) > 0 else "A"

        assistant_content = json.dumps({
            "answer": ans_letter,
            "reasoning": explanation if explanation else "Based on logical analysis of the given information."
        }, indent=2)

        formatted.append({
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    return formatted


def train_sft(config: TrainConfig):
    """Main SFT training function using Unsloth"""
    print("=" * 60)
    print(f"SFT Training for {config.agent_type} agent")
    print(f"Using Unsloth (AMD ROCm optimized)")
    print("=" * 60)

    # Find model path
    model_path = find_model_path(config.model_name)
    print(f"Model path: {model_path}")

    # Load model with Unsloth (no bitsandbytes, use bfloat16 for ROCm)
    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # No 4-bit for ROCm
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Setup chat template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format data
    print(f"Loading data from {config.data_dir}...")
    raw_data = load_training_data(config.data_dir)

    if config.agent_type == "question":
        formatted_data = format_for_question_agent(raw_data)
    else:
        formatted_data = format_for_answer_agent(raw_data)

    print(f"Formatted {len(formatted_data)} training samples")

    if len(formatted_data) == 0:
        print("ERROR: No training data found!")
        return None

    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    dataset = standardize_sharegpt(dataset)

    # Format with chat template
    def formatting_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            if isinstance(convo, list) and all(isinstance(msg, dict) for msg in convo):
                text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    print(f"Final dataset size: {len(dataset)}")

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=config.output_dir,
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
        ),
    )

    # Train only on responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="user",
        response_part="assistant",
    )

    # Train
    print("\nStarting training...")
    FastLanguageModel.for_training(model)
    trainer.train()

    # Save model
    lora_path = os.path.join(config.output_dir, f"final_{config.agent_type}_lora")
    print(f"\nSaving LoRA adapters to {lora_path}...")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # Save merged model
    merged_path = os.path.join(config.output_dir, f"final_{config.agent_type}_merged")
    print(f"Saving merged model to {merged_path}...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print(f"\nTraining complete!")
    print(f"LoRA: {lora_path}")
    print(f"Merged: {merged_path}")

    return lora_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT Training with Unsloth")
    parser.add_argument("--agent_type", type=str, default="question", choices=["question", "answer"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./sft_checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    args = parser.parse_args()

    config = TrainConfig(
        agent_type=args.agent_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
    )

    train_sft(config)
