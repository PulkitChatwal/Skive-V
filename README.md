# Skive-V — vLLM + value-aware block-wise KV-cache eviction

Skive-V is a fork of **vLLM `v0.23.0`** (commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`)
that adds **SKIVE**: value-aware, block-wise KV-cache eviction. It lets you run
**more concurrent requests / longer context on the same GPU** by capping each
sequence's KV footprint — trading a controlled amount of output quality for
memory headroom.

> **One-line summary:** flag **off ⇒ byte-identical to stock vLLM**; flag on ⇒
> fewer KV blocks per sequence (≈3–4× more concurrency/context when KV-bound),
> at some quality cost. It is **not** a universal upgrade — see *When to use it*.

## How it works (no kernel changes)
- **Score** each KV block by the proxy `‖V_block‖₂ / ‖K_block‖₂` (aggregated over
  layers) directly from the cache — lower = evict first.
- **Evict** the lowest-importance non-protected blocks by repointing their block
  table entry to vLLM's zeroed **null block** and freeing the physical block.
  Sequence length and token **positions are left unchanged**, so RoPE stays
  exact for every retained token (evicted positions read zeros — a bounded
  attention-sink). The first `num_sink_blocks` and last `num_local_blocks` are
  protected.
- All changes are pure Python in `vllm/` + a vendored `vllm/kv_evict/` package.

## Build (from source)
SKIVE is pure Python, but this is a full vLLM source tree, so it builds like
upstream vLLM. On a CUDA box:

```bash
git clone https://github.com/PulkitChatwal/Skive-V.git && cd Skive-V
pip install cmake ninja setuptools-rust          # + a Rust toolchain (cargo)
python use_existing_torch.py                      # build against your torch
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST="8.9" \
    pip install -e . --no-build-isolation         # set arch to YOUR GPU (8.9 = L4)
```
(Build time is dominated by CUDA kernel compilation; limiting `TORCH_CUDA_ARCH_LIST`
to your GPU's arch keeps it to ~1 hr.)

## Usage
**Requirement:** eviction hooks live in vLLM's **V1 model runner**, so always set
`VLLM_USE_V2_MODEL_RUNNER=0` (some architectures, e.g. Llama, default to V2 —
the engine will *raise loudly* if eviction is enabled on V2).

```python
import os
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_prefix_caching=False,        # supported; see notes
    kv_evict_enabled=True,              # default False == stock vLLM
    kv_evict_budget=16,                 # max KV blocks kept per sequence
    kv_evict_num_sink_blocks=2,         # leading blocks never evicted
    kv_evict_num_local_blocks=4,        # trailing (recent) blocks never evicted
)
print(llm.generate(["Hello"], SamplingParams(max_tokens=64))[0].outputs[0].text)
```
CLI equivalents: `--kv-evict-enabled --kv-evict-budget 16 --kv-evict-num-sink-blocks 2 --kv-evict-num-local-blocks 4`.

## When to use it
- ✅ **You are KV-bound** (long context / high concurrency filling the GPU) and can
  tolerate some quality loss → fits ~3–4× more on the same VRAM; measured
  **+24% to +60% throughput** in a KV-bound benchmark.
- ❌ **You are not KV-bound** → eviction only adds per-step overhead (measured
  **−8% to −16%**); just leave it off (or don't use this fork).
- ⚠️ **Quality is lossy** and worst at the aggressive budgets that save the most
  memory (token-overlap ≈0.78 vs FullKV at ~80% budget on a 0.5B model; coherent,
  never gibberish). Validate on *your* task before relying on it.

## Status / caveats
- Pinned to vLLM `0.23.0`. The eviction patches are version-specific.
- Validated on Qwen2.5-0.5B and TinyLlama-1.1B (GQA + a second arch).
- Prefix caching is supported (shared blocks are ref-counted, not corrupted).
- Preemption: eviction reduces the memory pressure that causes it.
- Quality numbers above are token-divergence vs FullKV, **not** a task-accuracy
  eval — run a real eval for your workload.

## Attribution & license
Based on [vLLM](https://github.com/vllm-project/vllm) `v0.23.0`, Apache-2.0.
SKIVE modifications by PulkitChatwal. See `LICENSE` (Apache-2.0, retained from
vLLM) and `NOTICE`.

## KV-eviction metric: `vk_ratio` vs `value_attention`

Two importance metrics decide which block to evict (select via the
`SKIVE_METRIC` environment variable; vLLM-style env knob):

| `SKIVE_METRIC` | score | notes |
|---|---|---|
| `vk_ratio` (default) | `‖V_block‖ / ‖K_block‖` from the cached KV | cheap, no query needed |
| `value_attention` | `Σ_tok Σ_head softmax(q·k)·‖v‖` (SKIVE's `‖p·v‖₁`, current query) | better accuracy under tight budgets; extra overhead |

```bash
# enable SKIVE's value x attention metric
export SKIVE_METRIC=value_attention
export VLLM_USE_V2_MODEL_RUNNER=0      # eviction is V1-runner only
```

### Measured: GSM8K accuracy (DeepSeek-R1-Distill-Qwen-1.5B, n=200, max_tokens=1024)
| Config | KV budget | Accuracy |
|---|---|---|
| FullKV | 100% | 54.0% |
| `vk_ratio` | ~23% | 21.5% |
| **`value_attention`** | ~23% | **30.5%** |
| `vk_ratio` | ~45% | 51.0% |
| **`value_attention`** | ~45% | **53.0%** |

- `value_attention` beats `vk_ratio` at every budget — by **+9 points** when
  aggressive (~23% budget), and is **near-lossless at ~45%** (53.0 vs 54.0).
- **Recommendation:** use `value_attention` for accuracy / aggressive budgets;
  keep `vk_ratio` (default) for lowest overhead.
- Caveats: validated on one model / one benchmark, `max_tokens=1024` (some CoT
  truncation), block- (not token-) granularity. Quality only — SKIVE's *speed*
  claim needs the fused kernel (not implemented here).

See `skive/metric_ab_findings.md` and `skive/gsm8k_eval.py` for details.
