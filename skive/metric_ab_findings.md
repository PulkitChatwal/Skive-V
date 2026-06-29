# Metric evaluation: `vk_ratio` vs SKIVE-style `value_attention`

Two metrics are available (select via `SKIVE_METRIC`):
- `vk_ratio` (default): `‖V_block‖ / ‖K_block‖` from the cached KV. Cheap, no
  query needed.
- `value_attention` (SKIVE's metric): `Σ_token Σ_head softmax(q·k)·‖v‖` using
  the current decode query (captured kernel-free via patch I). Closer to
  SKIVE's `‖p·v‖₁`.

## ⚠️ First A/B was the WRONG test
An initial A/B used **0.5B model + short summarization**, scored by
**closeness-to-FullKV (mean-LCP)**. That wrongly favored `vk_ratio`
(0.681 vs 0.580) because (a) the regime wasn't reasoning, and (b) LCP penalizes
the very divergence SKIVE is *designed* to produce. It does **not** judge SKIVE
fairly. See the GSM8K results below for the fair test.

## Fair test: GSM8K accuracy on a reasoning model
DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 arch → V1 runner), accuracy on the GSM8K
test set, max_tokens=1024, sink=2, local=8. Higher = better.

### n=200 (confirmed)
| Config | KV budget | Accuracy |
| --- | --- | --- |
| FullKV | 100% | 54.0% |
| vk_ratio | 16 (~23%) | 21.5% |
| **value_attention (SKIVE)** | 16 (~23%) | **30.5%** |
| vk_ratio | 32 (~45%) | 51.0% |
| **value_attention (SKIVE)** | 32 (~45%) | **53.0%** |

(n=40 earlier run agreed directionally: 40 / 20 / 30 / 50 / 47.5.)

## Conclusions (corrected & confirmed)
1. **SKIVE's `value_attention` beats the `vk_ratio` proxy at every budget** —
   decisively when aggressive (16: **30.5% vs 21.5%**, +9 pts), marginally at
   moderate (32: 53.0% vs 51.0%). The metric matters most when squeezing hard.
2. **Near-lossless at ~45% budget:** value_attention 53.0% vs FullKV 54.0%
   (−1 pt) — matches SKIVE's near-lossless claim.
3. **Correction:** the n=40 "eviction BEATS FullKV" was small-sample noise
   (n=40 FullKV was only 40%). At n=200 FullKV is 54% and eviction is
   near-lossless, **not** better. We did not reproduce a robust noise-filtering
   *gain* over FullKV at this scale/model — only near-losslessness.
4. `value_attention` costs extra overhead (per-step query capture + an
   attention recompute at eviction); `vk_ratio` is cheaper.

## Recommendation
- **For accuracy / aggressive budgets:** use `SKIVE_METRIC=value_attention`
  (clearly better, near-lossless at moderate budgets).
- **For lowest overhead / non-KV-bound:** keep `vk_ratio` (default).
- Caveats: one model (R1-Distill-Qwen-1.5B), one benchmark (GSM8K),
  max_tokens=1024 (some CoT truncation), block- (not token-) granularity.
  Directionally strong; broaden models/benchmarks before treating as final.

## When to use which
- Default stays `vk_ratio` (lower overhead, ties at moderate budgets).
- Use `SKIVE_METRIC=value_attention` for **aggressive budgets** / reasoning
  workloads, where it clearly helps.
- The earlier summarization/LCP result is retained only as a cautionary example
  of an unfair eval; the GSM8K accuracy numbers are the verdict.
