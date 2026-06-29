"""vLLM-side integration glue for KV-eviction (Stages 4b/4c).

This module is imported *inside* the running vLLM worker process. It must not
import vLLM at module load time (it's called from within vLLM), only torch.

Stage 4b: worker-side per-block scoring from the live KV cache, plus a
read-only debug logger used to validate the slicing/layout on real data
WITHOUT mutating anything.
"""

from __future__ import annotations

import torch

import os

from .manager import EvictionConfig
from .scoring import DEFAULT_EPS, score_blocks_value_attention

# vLLM's reserved placeholder block (block_pool.py: "placeholder block with
# block_id=0"). Pointing a block-table entry here frees the real block while
# reads return zeros (the cache is .zero_()'d at init and block 0 is never
# written). Under full attention this is a bounded zero-value attention sink --
# it does NOT shift positions, so RoPE stays exact for every retained token.
NULL_BLOCK_ID = 0


def score_request_blocks(
    kv_caches: list[torch.Tensor],
    block_ids,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Per-block importance for one request, aggregated over all layers.

    Each entry of ``kv_caches`` is a FlashAttention KV tensor of shape
    ``(num_blocks, 2, block_size, num_kv_heads, head_size)`` (NHD layout; dim 1
    index 0 = K, 1 = V). For physical blocks ``block_ids`` we compute

        score(b) = ||V_b||_2 / (||K_b||_2 + eps)

    where the norm is the L2 over every element of that block across ALL
    attention layers (i.e. the layers' contributions are concatenated). Lower
    score = evict first. Returns a 1-D float32 tensor of length len(block_ids),
    on the kv-cache device.
    """
    idx = torch.as_tensor(list(block_ids), dtype=torch.long)
    n = idx.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.float32)

    device = None
    sum_k = None
    sum_v = None
    for kc in kv_caches:
        # Only full-attention KV tensors (skip e.g. mamba state tensors).
        if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
            continue
        if device is None:
            device = kc.device
            idx = idx.to(device)
            sum_k = torch.zeros(n, dtype=torch.float32, device=device)
            sum_v = torch.zeros(n, dtype=torch.float32, device=device)
        k_blocks = kc[idx, 0].reshape(n, -1).to(torch.float32)
        v_blocks = kc[idx, 1].reshape(n, -1).to(torch.float32)
        sum_k += (k_blocks * k_blocks).sum(dim=1)
        sum_v += (v_blocks * v_blocks).sum(dim=1)

    if sum_k is None:  # no attention KV caches found
        return torch.empty(0, dtype=torch.float32)
    return torch.sqrt(sum_v) / (torch.sqrt(sum_k) + eps)


def _row_block_ids(model_runner, req_index: int):
    """Physical block ids currently held by request row `req_index` (group 0)."""
    bt = model_runner.input_batch.block_table[0]  # group 0 BlockTable
    n = int(bt.num_blocks_per_row[req_index])
    if n <= 0:
        return []
    return bt.block_table.np[req_index, :n].tolist()


def debug_log_scores(model_runner) -> bool:
    """Stage 4b validation: score the first request that has >= 2 blocks and
    log the result. READ-ONLY -- mutates nothing. Returns True once it has
    logged (so the caller can stop invoking it).
    """
    ib = model_runner.input_batch
    num_reqs = ib.num_reqs if hasattr(ib, "num_reqs") else len(ib.req_ids)
    for ri in range(num_reqs):
        block_ids = _row_block_ids(model_runner, ri)
        if len(block_ids) < 2:
            continue
        scores = score_request_blocks(model_runner.kv_caches, block_ids)
        vals = [round(x, 4) for x in scores.tolist()]
        finite = bool(torch.isfinite(scores).all().item())
        msg = (
            f"[SKIVE 4b] req_index={ri} blocks={len(block_ids)} "
            f"finite={finite} scores={vals}"
        )
        print(msg, flush=True)
        try:
            from vllm.logger import init_logger

            init_logger(__name__).info(msg)
        except Exception:
            pass
        return True
    return False


# --------------------------------------------------------------------------
# Stage 4c: actual eviction via null-block replacement (worker-side).
# --------------------------------------------------------------------------
def build_eviction_config(cache_config) -> EvictionConfig | None:
    """Construct an EvictionConfig from the vLLM cache_config, or None if
    eviction is disabled / misconfigured (in which case we never evict)."""
    if not getattr(cache_config, "kv_evict_enabled", False):
        return None
    budget = getattr(cache_config, "kv_evict_budget", None)
    if budget is None:
        return None
    try:
        return EvictionConfig(
            block_size=cache_config.block_size,
            kv_budget=int(budget),
            num_sink_blocks=int(getattr(cache_config, "kv_evict_num_sink_blocks", 0)),
            num_local_blocks=int(getattr(cache_config, "kv_evict_num_local_blocks", 0)),
            metric=os.environ.get("SKIVE_METRIC", "vk_ratio"),
        )
    except ValueError:
        return None


def score_request_blocks_va(model_runner, req_index, real_block_ids,
                            eps: float = DEFAULT_EPS):
    """SKIVE-style value x attention score for a request's real blocks,
    aggregated over layers. Returns a 1-D tensor (len == #real blocks) or None
    if the per-layer query wasn't captured (caller falls back to vk_ratio)."""
    try:
        import torch
        bs = int(model_runner.cache_config.block_size)
        q_row = int(model_runner.query_start_loc.np[req_index + 1]) - 1
        layers = [
            m for _, m in
            model_runner.compilation_config.static_forward_context.items()
            if type(m).__name__ in ("Attention", "MLAAttention")
        ]
        kv_caches = model_runner.kv_caches
        total = None
        for li, layer in enumerate(layers):
            q = getattr(layer, "_skive_q", None)
            if q is None or li >= len(kv_caches):
                return None
            kc = kv_caches[li]
            if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
                continue
            dev = kc.device
            idx = torch.as_tensor(real_block_ids, dtype=torch.long, device=dev)
            qr = q[q_row].to(dev)                       # [num_heads, head_size]
            kb = kc[idx, 0]                              # [nb, bs, Hkv, D]
            vb = kc[idx, 1]
            nb = kb.shape[0]
            k_all = kb.reshape(nb * bs, kb.shape[2], kb.shape[3])
            v_all = vb.reshape(nb * bs, vb.shape[2], vb.shape[3])
            nqpk = getattr(layer.impl, "num_queries_per_kv",
                           qr.shape[0] // kb.shape[2])
            scale = getattr(layer.impl, "scale", None)
            s = score_blocks_value_attention(qr, k_all, v_all, bs, nqpk, scale, eps)
            total = s if total is None else total + s
        return total
    except Exception:
        return None


def _ensure_null_zeroed(model_runner) -> None:
    """One-time defensive zero of the null block's KV across all layers."""
    if getattr(model_runner, "_skive_null_zeroed", False):
        return
    for kc in model_runner.kv_caches:
        if kc is not None and kc.ndim == 5 and kc.shape[1] == 2:
            kc[NULL_BLOCK_ID].zero_()
    model_runner._skive_null_zeroed = True


def evict_request_blocks(
    model_runner, cfg: EvictionConfig, req_index: int
) -> list[int]:
    """Null-replace the lowest-importance non-protected real blocks of one
    request until it is back within budget. Returns the logical row indices
    that were nulled (so the scheduler can free the same physical blocks).

    Positions/length are left UNCHANGED (RoPE stays exact); only the block
    table entry is repointed to the null block.
    """
    bt = model_runner.input_batch.block_table[0]  # group 0
    n = int(bt.num_blocks_per_row[req_index])
    if n == 0:
        return []
    row = bt.block_table.np[req_index]
    block_ids_full = row[:n].tolist()

    # Real (non-null) blocks, in logical order, with their row indices.
    real_pos = [i for i in range(n) if block_ids_full[i] != NULL_BLOCK_ID]
    num_real = len(real_pos)
    if num_real <= cfg.kv_budget:
        return []

    # Protected over the REAL ordering: first num_sink + last num_local.
    sink, local = cfg.num_sink_blocks, cfg.num_local_blocks
    protected_k = set(range(0, min(sink, num_real))) | set(
        range(max(0, num_real - local), num_real)
    )
    candidates_k = [k for k in range(num_real) if k not in protected_k]
    if not candidates_k:
        return []

    num_to_evict = min(num_real - cfg.kv_budget, len(candidates_k))

    # Score only the real blocks (one .tolist() sync, only when over budget).
    real_block_ids = [block_ids_full[real_pos[k]] for k in range(num_real)]
    scores = None
    if getattr(cfg, "metric", "vk_ratio") == "value_attention":
        sv = score_request_blocks_va(model_runner, req_index, real_block_ids)
        if sv is not None:
            scores = sv.tolist()
    if scores is None:  # default proxy, or fallback if query not captured
        scores = score_request_blocks(model_runner.kv_caches, real_block_ids).tolist()

    # Lowest-importance candidates first (stable on ties via index).
    chosen = sorted(candidates_k, key=lambda k: (scores[k], k))[:num_to_evict]
    nulled_rows = []
    for k in chosen:
        j = real_pos[k]
        row[j] = NULL_BLOCK_ID  # repoint worker table to null
        nulled_rows.append(j)
    return nulled_rows


def skive_post_step(model_runner) -> None:
    """Stable post-decode entrypoint called from the patched gpu_model_runner.

    Nulls evicted entries in the worker block table and records (req_id, j)
    pairs on ``model_runner._skive_pending_free`` for EngineCore to free on the
    scheduler side (4c-ii). Flag-gated upstream; a None config => no-op."""
    cfg = build_eviction_config(model_runner.cache_config)
    if cfg is None:
        return
    _ensure_null_zeroed(model_runner)

    ib = model_runner.input_batch
    num_reqs = ib.num_reqs if hasattr(ib, "num_reqs") else len(ib.req_ids)
    pending = getattr(model_runner, "_skive_pending_free", [])
    total = 0
    for ri in range(num_reqs):
        req_id = ib.req_ids[ri]
        if req_id is None:
            continue
        for j in evict_request_blocks(model_runner, cfg, ri):
            pending.append((req_id, j))
            total += 1
    model_runner._skive_pending_free = pending
    if total:
        c = getattr(model_runner, "_skive_evicted_total", 0) + total
        model_runner._skive_evicted_total = c
        print(f"[SKIVE 4c] evicted {total} block(s) this step; cumulative={c}",
              flush=True)


def _skive_pop_pending(worker):
    """Worker-side: return and clear the pending (req_id, logical_index) frees.

    Used via collective_rpc from EngineCore. ``worker`` is the (wrapped) worker;
    attribute access delegates to the real worker, so .model_runner resolves."""
    mr = getattr(worker, "model_runner", None)
    if mr is None:
        return []
    pending = getattr(mr, "_skive_pending_free", None)
    if not pending:
        return []
    mr._skive_pending_free = []
    return list(pending)


def skive_reclaim(kv_cache_manager, pending) -> int:
    """Scheduler-side: free the physical blocks the worker evicted, mirroring
    them to null_block in req_to_blocks (group 0). Returns #blocks freed.

    This mirrors vLLM's own sliding-window path (remove_skipped_blocks): null
    substitution + BlockPool.free_blocks. Prefix caching is OFF in this stage,
    so there are no shared/cached-block ref-count complications.
    """
    block_pool = kv_cache_manager.block_pool
    null_block = block_pool.null_block
    managers = kv_cache_manager.coordinator.single_type_managers
    if not managers:
        return 0
    mgr = managers[0]  # group 0 (single full-attention group); multi-group=Stage 5
    freed = 0
    for req_id, j in pending:
        blocks = mgr.req_to_blocks.get(req_id)
        if not blocks or j >= len(blocks):
            continue
        old = blocks[j]
        if getattr(old, "is_null", False):
            continue
        blocks[j] = null_block
        block_pool.free_blocks([old])
        freed += 1
    if freed:
        c = getattr(kv_cache_manager, "_skive_freed_total", 0) + freed
        kv_cache_manager._skive_freed_total = c
        print(f"[SKIVE 4c-ii] freed {freed} physical block(s); cumulative={c}; "
              f"pool_free={block_pool.get_num_free_blocks()}", flush=True)
    return freed
