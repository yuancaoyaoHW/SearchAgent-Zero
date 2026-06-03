#!/usr/bin/env bash
# Minimal non-server Search-R1 smoke for V100.
#
# This does not start Ray, vLLM, SGLang, AgentLoopManager, or any async rollout
# server. It validates the local data/model/tokenizer path with one HF
# generate + one response-only training step.

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
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export LR="${LR:-1e-6}"

python - <<'PY'
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, raw_prompt):
    if isinstance(raw_prompt, list):
        return tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
    if isinstance(raw_prompt, str):
        return raw_prompt
    raise TypeError(f"Unsupported prompt type: {type(raw_prompt)!r}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this V100 smoke script")

    model_path = os.environ["MODEL_PATH"]
    train_data = os.environ["TRAIN_DATA"]
    max_prompt_length = int(os.environ["MAX_PROMPT_LENGTH"])
    max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
    lr = float(os.environ["LR"])

    print(f"[hf-smoke] device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    print(f"[hf-smoke] model={model_path}")
    print(f"[hf-smoke] data={train_data}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("parquet", data_files=train_data, split="train")
    if len(dataset) == 0:
        raise RuntimeError(f"No rows found in {train_data}")

    row = dataset[0]
    if "prompt" not in row:
        raise KeyError(f"Expected a 'prompt' column, got columns: {list(row.keys())}")

    prompt_text = build_prompt(tokenizer, row["prompt"])
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )
    encoded = {key: value.cuda() for key, value in encoded.items()}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = encoded["input_ids"].shape[1]
    if generated.shape[1] <= prompt_len:
        raise RuntimeError("Model generated no response tokens")

    attention_mask = torch.ones_like(generated)
    labels = generated.clone()
    labels[:, :prompt_len] = -100

    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=generated, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss}")
    loss.backward()
    optimizer.step()

    response = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
    print(f"[hf-smoke] prompt_tokens={prompt_len} response_tokens={generated.shape[1] - prompt_len}")
    print(f"[hf-smoke] loss={loss.item():.6f}")
    print("[hf-smoke] response_preview=" + response[:500].replace("\n", "\\n"))
    print("[hf-smoke] done")


if __name__ == "__main__":
    main()
PY
