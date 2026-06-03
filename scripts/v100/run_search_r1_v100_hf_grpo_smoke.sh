#!/usr/bin/env bash
# HF-only GRPO smoke for V100.
#
# This is intentionally not wired into verl.trainer.main_ppo. It does not start
# Ray, vLLM, SGLang, AgentLoopManager, or a rollout server. It checks whether a
# tiny local GRPO-style update can run with the Search-R1 parquet and HF model.

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
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GROUP_SIZE="${GROUP_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export LR="${LR:-1e-6}"
export SEED="${SEED:-0}"

python - <<'PY'
import os
import re
import string

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


def smoke_reward(response: str, targets: list[str], response_token_count: int, max_new_tokens: int) -> float:
    """Small visible reward for smoke only, not a production Search-R1 reward."""
    answer = normalize_answer(extract_answer(response))
    normalized_targets = [normalize_answer(target) for target in targets]
    exact = 1.0 if answer and answer in normalized_targets else 0.0
    has_answer_tag = 0.2 if re.search(r"<answer>.*?</answer>", response, flags=re.IGNORECASE | re.DOTALL) else 0.0
    unique_ratio = len(set(response.split())) / max(1, len(response.split()))
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this V100 smoke script")

    model_path = os.environ["MODEL_PATH"]
    train_data = os.environ["TRAIN_DATA"]
    batch_size = int(os.environ["BATCH_SIZE"])
    group_size = int(os.environ["GROUP_SIZE"])
    max_prompt_length = int(os.environ["MAX_PROMPT_LENGTH"])
    max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
    lr = float(os.environ["LR"])
    seed = int(os.environ["SEED"])
    torch.manual_seed(seed)

    print(f"[hf-grpo-smoke] device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    print(f"[hf-grpo-smoke] model={model_path}")
    print(f"[hf-grpo-smoke] data={train_data}")
    print(f"[hf-grpo-smoke] batch_size={batch_size} group_size={group_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset("parquet", data_files=train_data, split="train")
    if len(dataset) < batch_size:
        raise RuntimeError(f"Need at least {batch_size} rows, found {len(dataset)}")

    prompts = []
    targets = []
    for row in dataset.select(range(batch_size)):
        if "prompt" not in row:
            raise KeyError(f"Expected a 'prompt' column, got columns: {list(row.keys())}")
        prompts.append(build_prompt(tokenizer, row["prompt"]))
        targets.append(extract_targets(row))

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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
    for i, sequence in enumerate(generated):
        group_index = i // group_size
        prompt_len = repeated_prompt_lens[i]
        response_ids = sequence[prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        response_token_count = int(response_ids.ne(tokenizer.pad_token_id).sum().item())
        reward = smoke_reward(response_text, targets[group_index], response_token_count, max_new_tokens)
        rewards.append(reward)
        responses.append(response_text)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32).view(batch_size, group_size)
    mean = reward_tensor.mean(dim=1, keepdim=True)
    std = reward_tensor.std(dim=1, keepdim=True, unbiased=False)
    if torch.any(std < 1e-8):
        raise RuntimeError(f"Zero reward variance inside a GRPO group: rewards={reward_tensor.tolist()}")
    advantages = ((reward_tensor - mean) / std).view(-1).tolist()

    optimizer.zero_grad(set_to_none=True)
    total_loss_value = 0.0
    for i, sequence in enumerate(generated):
        sequence = sequence.unsqueeze(0).cuda()
        loss = response_logprob_loss(model, sequence, repeated_prompt_lens[i], advantages[i]) / generated.shape[0]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss}")
        total_loss_value += float(loss.detach().cpu())
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"[hf-grpo-smoke] rewards={reward_tensor.tolist()}")
    print(f"[hf-grpo-smoke] advantages={[round(x, 4) for x in advantages]}")
    print(f"[hf-grpo-smoke] loss={total_loss_value:.6f} grad_norm={float(grad_norm):.6f}")
    print("[hf-grpo-smoke] response_preview=" + responses[0][:500].replace("\n", "\\n"))
    print("[hf-grpo-smoke] done")


if __name__ == "__main__":
    main()
PY
