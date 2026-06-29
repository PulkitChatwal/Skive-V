"""Stage 6 benchmark: FullKV vs block-wise eviction at matched KV budget.

A 0.5B model on a 23GB L4 is NOT naturally KV-bound, so we create a controlled
KV-bound regime with --blocks (num_gpu_blocks_override) -- this simulates the
memory pressure of a bigger model / longer context, where eviction matters.
The workload (nreqs concurrent requests, each a fixed-length long generation)
is identical across runs; only the eviction config changes.

Reported per run:
  * throughput  = total output tokens / wall time   (tok/s)
  * mean TPOT   = mean time-per-output-token across requests (ms)
  * elapsed     = wall time for the whole workload (s)
  * blocks/seq  = KV blocks a full-length sequence occupies (FullKV vs capped)

Run via stage6/run_bench.sh which sweeps FullKV + several budgets.
ALWAYS run with VLLM_USE_V2_MODEL_RUNNER=0 (eviction is V1-runner only).
"""

import argparse
import json
import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evict", action="store_true")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--sink", type=int, default=2)
    ap.add_argument("--local", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=1500)   # KV budget (pressure)
    ap.add_argument("--nreqs", type=int, default=96)
    ap.add_argument("--prompt-len", type=int, default=200)
    ap.add_argument("--gen-len", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=1024)
    args = ap.parse_args()

    llm = LLM(
        model=MODEL, dtype="bfloat16", seed=0, enforce_eager=True,
        enable_prefix_caching=False, gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        num_gpu_blocks_override=args.blocks,
        kv_evict_enabled=args.evict, kv_evict_budget=args.budget,
        kv_evict_num_sink_blocks=args.sink, kv_evict_num_local_blocks=args.local,
    )
    tok = llm.get_tokenizer()
    # Distinct long prompts (~prompt-len tokens) so KV is not shared.
    base = ("The history of science and technology spans many centuries and "
            "countless discoveries across physics, chemistry, biology. ")
    prompts = []
    for i in range(args.nreqs):
        s = f"Document {i}. " + base * 40
        ids = tok(s).input_ids[: args.prompt_len]
        prompts.append(tok.decode(ids))

    # Fixed-length generation for a clean throughput measurement.
    sp = SamplingParams(temperature=0.0, min_tokens=args.gen_len,
                        max_tokens=args.gen_len, ignore_eos=True)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    tpots = []
    for o in outs:
        m = o.metrics
        if m and m.first_token_time and m.finished_time:
            n = len(o.outputs[0].token_ids)
            if n > 1:
                tpots.append((m.finished_time - m.first_token_time) / (n - 1))
    mean_tpot_ms = 1000 * sum(tpots) / len(tpots) if tpots else float("nan")
    full_seq_blocks = -(-(args.prompt_len + args.gen_len) // 16)  # ceil
    eff_blocks = args.budget if args.evict else full_seq_blocks

    res = {
        "mode": "evict" if args.evict else "fullkv",
        "budget": args.budget if args.evict else None,
        "blocks_override": args.blocks,
        "nreqs": args.nreqs,
        "out_tokens": total_out,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(total_out / elapsed, 1),
        "mean_tpot_ms": round(mean_tpot_ms, 2),
        "blocks_per_seq": eff_blocks,
        "fullkv_blocks_per_seq": full_seq_blocks,
    }
    print("BENCH_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
