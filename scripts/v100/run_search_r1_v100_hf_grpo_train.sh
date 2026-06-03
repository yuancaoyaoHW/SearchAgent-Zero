#!/usr/bin/env bash
# HF-only GRPO training loop for V100.
#
# This is a runnable single-GPU training path for debugging and small-scale
# experiments. It does not start Ray, vLLM, SGLang, AgentLoopManager, or any
# async rollout server, and it does not execute Search-R1 tools.

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

CACHE_BASE="${CACHE_BASE:-/mnt/models/temp_cache}"
export HF_HOME="${HF_HOME:-$CACHE_BASE/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_BASE/hf_datasets}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

export MODEL_PATH="${MODEL_PATH:-/mnt/models/Qwen2.5-1.5B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-/mnt/models/search_r1_processed/train_search_r1.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/qwen2.5-1.5b-searchr1-hf-grpo-v100}"
export LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train_metrics.jsonl}"

export TRAIN_STEPS="${TRAIN_STEPS:-100}"
export SAVE_EVERY="${SAVE_EVERY:-25}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GROUP_SIZE="${GROUP_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export LR="${LR:-1e-6}"
export SEED="${SEED:-0}"
export CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-1.0}"
export TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"

mkdir -p "$OUTPUT_DIR"

python - <<'PY'
import json
import os
import re
import string
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_targets(row: dict) -> list[str]:
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        extra_info = row.get("extra_info") or {}
        ground_truth = extra_info.get("ground_truth")
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target")
    if isinstance(ground_truth, str):
        return [ground_truth]
    if isinstance(ground_truth, (list, tuple)):
        return [str(item) for item in ground_truth]
    raise KeyError(f"Missing reward_model.ground_truth/extra_info.ground_truth in row keys={list(row.keys())}")


def build_prompt(tokenizer, raw_prompt):
    if isinstance(raw_prompt, list):
        return tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
    if isinstance(raw_prompt, str):
        return raw_prompt
    raise TypeError(f"Unsupported prompt type: {type(raw_prompt)!r}")


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def reward_fn(response: str, targets: list[str], response_token_count: int, max_new_tokens: int) -> float:
    """Simple visible training reward for this HF-only path.

    It is intentionally local and deterministic: exact-match answer reward,
    answer-tag format reward, lexical diversity, and bounded length reward.
    It is not the full Search-R1 tool-use reward.
    """
    answer = normalize_answer(extract_answer(response))
    normalized_targets = [normalize_answer(target) for target in targets]
    exact = 1.0 if answer and answer in normalized_targets else 0.0
    has_answer_tag = 0.2 if re.search(r"<answer>.*?</answer>", response, flags=re.IGNORECASE | re.DOTALL) else 0.0
    words = response.split()
    unique_ratio = len(set(words)) / max(1, len(words))
    length_ratio = min(response_token_count, max_new_tokens) / max(1, max_new_tokens)
    return exact + has_answer_tag + 0.1 * unique_ratio + 0.05 * length_ratio


def response_logprob_loss(model, sequence, prompt_len, advantage):
    labels = sequence.clone()
    labels[:, :prompt_len] = -100
    outputs = model(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    token_log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    seq_log_prob = (token_log_probs * valid).sum() / valid.sum().clamp_min(1)
    return -advantage * seq_log_prob


def save_checkpoint(model, tokenizer, output_dir: Path, step: int):
    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    latest_file = output_dir / "latest_checkpoint.txt"
    latest_file.write_text(str(ckpt_dir), encoding="utf-8")
    print(f"[hf-grpo-train] saved checkpoint: {ckpt_dir}", flush=True)


def resolve_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"TRAIN_DTYPE must be float32 or float16, got {name}")


def assert_finite_model(model, step: int):
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            raise RuntimeError(f"Non-finite parameter after optimizer step {step}: {name}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this V100 training script")

    model_path = os.environ["MODEL_PATH"]
    train_data = os.environ["TRAIN_DATA"]
    output_dir = Path(os.environ["OUTPUT_DIR"])
    log_file = Path(os.environ["LOG_FILE"])
    train_steps = int(os.environ["TRAIN_STEPS"])
    save_every = int(os.environ["SAVE_EVERY"])
    batch_size = int(os.environ["BATCH_SIZE"])
    group_size = int(os.environ["GROUP_SIZE"])
    max_prompt_length = int(os.environ["MAX_PROMPT_LENGTH"])
    max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
    lr = float(os.environ["LR"])
    seed = int(os.environ["SEED"])
    clip_grad_norm = float(os.environ["CLIP_GRAD_NORM"])
    train_dtype = resolve_dtype(os.environ["TRAIN_DTYPE"])

    if train_steps <= 0:
        raise ValueError(f"TRAIN_STEPS must be > 0, got {train_steps}")
    if batch_size <= 0:
        raise ValueError(f"BATCH_SIZE must be > 0, got {batch_size}")
    if group_size <= 1:
        raise ValueError(f"GROUP_SIZE must be > 1 for GRPO, got {group_size}")

    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[hf-grpo-train] device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    print(f"[hf-grpo-train] model={model_path}")
    print(f"[hf-grpo-train] data={train_data}")
    print(f"[hf-grpo-train] output_dir={output_dir}")
    print(f"[hf-grpo-train] train_steps={train_steps} batch_size={batch_size} group_size={group_size}")
    print(f"[hf-grpo-train] train_dtype={train_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset("parquet", data_files=train_data, split="train")
    if len(dataset) < batch_size:
        raise RuntimeError(f"Need at least {batch_size} rows, found {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=train_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    log_handle = log_file.open("a", encoding="utf-8")

    try:
        for step in range(1, train_steps + 1):
            start_time = time.time()
            offset = ((step - 1) * batch_size) % len(dataset)
            indices = [(offset + i) % len(dataset) for i in range(batch_size)]
            rows = [dataset[int(index)] for index in indices]

            prompts = []
            targets = []
            data_sources = []
            for row in rows:
                if "prompt" not in row:
                    raise KeyError(f"Expected a 'prompt' column, got columns: {list(row.keys())}")
                prompts.append(build_prompt(tokenizer, row["prompt"]))
                targets.append(extract_targets(row))
                data_sources.append(row.get("data_source", "unknown"))

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_length,
            )
            padded_prompt_len = encoded["input_ids"].shape[1]
            repeated = {key: value.repeat_interleave(group_size, dim=0).cuda() for key, value in encoded.items()}
            repeated_prompt_lens = [padded_prompt_len] * (batch_size * group_size)

            model.eval()
            with torch.no_grad():
                generated = model.generate(
                    **repeated,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            model.train()

            rewards = []
            responses = []
            response_token_counts = []
            for i, sequence in enumerate(generated):
                group_index = i // group_size
                prompt_len = repeated_prompt_lens[i]
                response_ids = sequence[prompt_len:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                response_token_count = int(response_ids.ne(tokenizer.pad_token_id).sum().item())
                rewards.append(reward_fn(response_text, targets[group_index], response_token_count, max_new_tokens))
                responses.append(response_text)
                response_token_counts.append(response_token_count)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32).view(batch_size, group_size)
            mean = reward_tensor.mean(dim=1, keepdim=True)
            std = reward_tensor.std(dim=1, keepdim=True, unbiased=False)
            if torch.any(std < 1e-8):
                raise RuntimeError(f"Zero reward variance inside a GRPO group at step {step}: {reward_tensor.tolist()}")
            advantages = ((reward_tensor - mean) / std).view(-1).tolist()

            optimizer.zero_grad(set_to_none=True)
            total_loss_value = 0.0
            for i, sequence in enumerate(generated):
                sequence = sequence.unsqueeze(0).cuda()
                loss = response_logprob_loss(model, sequence, repeated_prompt_lens[i], advantages[i]) / generated.shape[0]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
                total_loss_value += float(loss.detach().cpu())
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            assert_finite_model(model, step)

            elapsed = time.time() - start_time
            record = {
                "step": step,
                "loss": total_loss_value,
                "grad_norm": float(grad_norm),
                "reward_mean": float(reward_tensor.mean().item()),
                "reward_min": float(reward_tensor.min().item()),
                "reward_max": float(reward_tensor.max().item()),
                "response_tokens_mean": sum(response_token_counts) / len(response_token_counts),
                "elapsed_sec": elapsed,
                "data_sources": data_sources,
            }
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()
            print(
                "[hf-grpo-train] "
                f"step={step} loss={total_loss_value:.6f} "
                f"reward_mean={record['reward_mean']:.6f} "
                f"reward_min={record['reward_min']:.6f} reward_max={record['reward_max']:.6f} "
                f"grad_norm={float(grad_norm):.6f} elapsed={elapsed:.2f}s",
                flush=True,
            )

            if step == 1:
                preview = responses[0][:500].replace("\n", "\\n")
                print(f"[hf-grpo-train] response_preview={preview}", flush=True)

            if save_every > 0 and step % save_every == 0:
                save_checkpoint(model, tokenizer, output_dir, step)

        save_checkpoint(model, tokenizer, output_dir, train_steps)
        print("[hf-grpo-train] done", flush=True)
    finally:
        log_handle.close()


if __name__ == "__main__":
    main()
PY
