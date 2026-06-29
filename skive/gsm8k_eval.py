# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fair SKIVE eval: GSM8K accuracy on a reasoning model.

Compares FullKV vs vk_ratio vs value_attention (SKIVE's metric) by TASK
ACCURACY (not closeness to FullKV). One metric per process (value_attention
needs SKIVE_METRIC set before import for the query-capture flag).

Usage: python gsm8k_eval.py <fullkv|vk_ratio|value_attention> <budget> [n]
"""

import os
import sys

import regex as re

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
from vllm import LLM, SamplingParams

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Qwen2 arch -> V1 runner
mode = sys.argv[1] if len(sys.argv) > 1 else "fullkv"
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 16
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
evict = mode != "fullkv"


def gold_of(ans):
    m = re.search(r"####\s*([-0-9,\.]+)", ans)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def pred_of(text):
    # prefer a boxed / "answer" number; else the last number in the text
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def main():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{N}]")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        kv_evict_enabled=evict,
        kv_evict_budget=(budget if evict else 1),
        kv_evict_num_sink_blocks=2,
        kv_evict_num_local_blocks=8,
    )
    tok = llm.get_tokenizer()
    prompts, golds = [], []
    for ex in ds:
        q = ex["question"] + (
            "\nPlease reason step by step and put your final answer after '####'."
        )
        prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        golds.append(gold_of(ex["answer"]))

    sp = SamplingParams(temperature=0.0, max_tokens=1024)
    outs = llm.generate(prompts, sp)

    correct = 0
    for o, g in zip(outs, golds):
        p = pred_of(o.outputs[0].text)
        try:
            ok = p is not None and g is not None and abs(float(p) - float(g)) < 1e-4
        except ValueError:
            ok = p == g
        correct += int(ok)
    acc = 100.0 * correct / len(golds)
    print(
        f"RESULT mode={mode} budget={budget if evict else '-'} "
        f"n={len(golds)} accuracy={acc:.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
