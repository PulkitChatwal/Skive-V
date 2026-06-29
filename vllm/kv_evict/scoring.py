"""Value-aware KV-block importance scoring (SKIVE proxy metric).

Stage 1 of the integration: a *pure* function, with no GPU and no vLLM
dependency, that turns the K/V tensors of some KV-cache blocks into one
importance score per block.

The metric is the proxy chosen for this project (NOT the kernel-fused
``||p_i * v_i||_1`` variant, which is explicitly out of scope)::

    score(block) = ||V_block||_2 / ||K_block||_2

where ``||.||_2`` is the L2 (Frobenius) norm over *all* elements of the
block's tensor. Lower score => evict first.

Intuition: a block whose value vectors carry large magnitude relative to
their key magnitude tends to contribute more to attention outputs, so it is
worth retaining. The ratio is computed directly from the KV already sitting
in cache, so it requires no attention-kernel changes.

Numerical notes:
  * Norms are accumulated in float32 even for fp16/bf16 inputs, to avoid
    overflow/underflow; the returned scores are float32.
  * ``eps`` guards the division so an all-zero K block can never produce
    inf/nan.
"""

from __future__ import annotations

import torch

# Small constant added to the key-norm denominator to avoid divide-by-zero.
DEFAULT_EPS: float = 1e-6


def score_blocks(
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Compute one importance score per KV block.

    Args:
        k_blocks: key tensor for a set of blocks. The first dimension indexes
            blocks; all remaining dimensions are reduced into the per-block
            norm. Typical real shape is ``[num_blocks, block_size, num_kv_heads,
            head_dim]``, but any shape with ``ndim >= 2`` works.
        v_blocks: value tensor, same shape as ``k_blocks``.
        eps: positive constant added to the key norm before dividing.

    Returns:
        A 1-D ``float32`` tensor of shape ``[num_blocks]`` on the same device
        as the inputs. Element ``i`` is ``||V_i||_2 / (||K_i||_2 + eps)``.
        Lower means "evict first".

    Raises:
        ValueError: if shapes mismatch, inputs are not at least 2-D, or
            ``eps`` is not strictly positive.
    """
    if k_blocks.shape != v_blocks.shape:
        raise ValueError(
            f"k_blocks and v_blocks must have identical shapes, got "
            f"{tuple(k_blocks.shape)} vs {tuple(v_blocks.shape)}"
        )
    if k_blocks.ndim < 2:
        raise ValueError(
            f"expected at least 2 dims (num_blocks, ...), got ndim="
            f"{k_blocks.ndim}"
        )
    if not eps > 0:
        raise ValueError(f"eps must be strictly positive, got {eps}")

    num_blocks = k_blocks.shape[0]
    if num_blocks == 0:
        # Preserve device; empty score vector.
        return torch.empty(0, dtype=torch.float32, device=k_blocks.device)

    # Flatten each block to a vector, accumulate norm in float32 for stability.
    k_flat = k_blocks.reshape(num_blocks, -1).to(torch.float32)
    v_flat = v_blocks.reshape(num_blocks, -1).to(torch.float32)

    k_norm = torch.linalg.vector_norm(k_flat, dim=1)
    v_norm = torch.linalg.vector_norm(v_flat, dim=1)

    return v_norm / (k_norm + eps)
