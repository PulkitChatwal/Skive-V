# Stage 6 — Benchmark results

**Setup:** vLLM 0.23.0, NVIDIA L4 (23GB), Qwen2.5-0.5B-Instruct, `enforce_eager`,
prefix caching off, `VLLM_USE_V2_MODEL_RUNNER=0`. Workload: 96 concurrent
requests, ~200-token prompt + 200 fixed output tokens each (19,200 output
tokens total). KV pressure is controlled with `num_gpu_blocks_override` (a 0.5B
on a 23GB L4 is not naturally KV-bound). FullKV uses 25 KV blocks per
full-length sequence; eviction caps each at `budget`.

TPOT not reported — `RequestOutput.metrics` timing fields were unpopulated in
this configuration; throughput + wall time are the reliable signals.

## A. NOT KV-bound (blocks=1500 → FullKV fits ~60 concurrent)
| mode | budget | blocks/seq | throughput (tok/s) | elapsed (s) | vs FullKV |
|------|-------:|-----------:|-------------------:|------------:|----------:|
| FullKV | — | 25 | **2171** | 8.84 | — |
| evict | 20 | 20 | 1824 | 10.53 | **−16%** |
| evict | 13 | 13 | 2008 | 9.56 | −8% |
| evict |  8 |  8 | 1937 | 9.91 | −11% |

When the GPU is **not** KV-bound, eviction is **slower** — the per-step scoring
(GPU norms + a CPU sync on eviction) and block bookkeeping cost throughput, with
no concurrency benefit to offset it. There is no free lunch here.

## B. KV-bound (blocks=384 → FullKV fits only ~15 concurrent)
| mode | budget | blocks/seq | throughput (tok/s) | elapsed (s) | vs FullKV |
|------|-------:|-----------:|-------------------:|------------:|----------:|
| FullKV | — | 25 | 834 | 23.03 | — |
| evict | 8 | 8 | **1038** | 18.50 | **+24%** |
| evict | 6 | 6 | **1332** | 14.41 | **+60%** |

When KV memory is the bottleneck, eviction caps each sequence's footprint, so
many more requests run concurrently (FullKV serializes into ~6 waves; eviction
into ~2), and throughput rises. The +60% case is legitimate (driven by
concurrency, outputs coherent — not a broken run) but comes at the most
aggressive budget and therefore the largest quality loss.

## Memory savings (deterministic)
Per full-length sequence: FullKV = 25 blocks; eviction = `budget` (8 → −68%,
6 → −76%). i.e. eviction fits **~3–4× more concurrent sequences** (or ~3–4×
longer context) in the same KV memory — this is the real, reliable win.

## Quality cost (from Stages 4–5, same proxy metric)
Eviction is lossy. Greedy outputs vs FullKV: ~mean-LCP **0.78 at ~80% budget**,
**0.68 at ~25% budget** on Qwen2.5-0.5B; coherent (0 degenerate) throughout.
Aggressive budgets (the ones that win most on throughput/memory) lose the most
quality. Short-sequence / small-model greedy decoding amplifies divergence; the
proxy may behave better on larger models / longer contexts (not benchmarked).

## Honest bottom line
- **Memory: real and reliable** — 68–76% less KV per sequence at aggressive
  budgets → fit ~3–4× more concurrency / longer context on the same GPU.
- **Throughput: regime-dependent** — a **loss (−8 to −16%)** when not KV-bound
  (overhead), a **gain (+24% to +60%)** when genuinely KV-bound. Tens-of-percent,
  not 5×, as expected.
- **Quality: a real cost** — graceful but lossy; worst at the budgets that win
  most on speed/memory.
- **Use it when** you are KV-bound (long context / high concurrency on a large
  model) and can tolerate some quality loss; **don't** enable it on workloads
  that already fit comfortably — it will only add overhead.
- Not benchmarked: a true large-model (e.g. 7B) long-context run, where the
  memory→concurrency→throughput win should be strongest. Would need more GPU.
