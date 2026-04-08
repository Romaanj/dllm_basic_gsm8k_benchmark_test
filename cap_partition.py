from __future__ import annotations

from typing import List


def context_aware_partition(
    L: int,
    K: int,
    P: int,
    alpha: float = 1.0,
    B_min: int = 8,
    max_iter: int = 50,
) -> List[int]:
    """Context-Aware Partition (CAP).

    Block sizes are allocated inversely proportional to available left context:
      B_k is proportional to 1 / C_k^alpha,
      where C_k = P + sum(B_0 .. B_{k-1}).
    """
    L = int(L)
    K = int(K)
    P = int(P)
    alpha = float(alpha)
    B_min = int(B_min)
    max_iter = int(max_iter)

    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if P < 0:
        raise ValueError(f"P must be >= 0, got {P}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if B_min <= 0:
        raise ValueError(f"B_min must be > 0, got {B_min}")
    if B_min * K > L:
        raise ValueError(
            f"Infeasible CAP setup: B_min*K={B_min * K} exceeds L={L}"
        )

    if alpha == 0:
        # Uniform partition.
        base = L // K
        remainder = L - base * K
        return [base + (1 if i < remainder else 0) for i in range(K)]

    # Iterative fixed-point solver.
    blocks = [L / K] * K

    for _ in range(max_iter):
        contexts = []
        cumsum = 0.0
        for k in range(K):
            contexts.append(P + cumsum)
            cumsum += blocks[k]

        # Avoid division by zero when prompt length is 0.
        weights = [1.0 / (max(c, 1e-9) ** alpha) for c in contexts]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0] * K
            total_w = float(K)

        new_blocks = [max(B_min, L * w / total_w) for w in weights]

        # Renormalize to sum exactly L in float space.
        total = sum(new_blocks)
        new_blocks = [b * L / total for b in new_blocks]

        if max(abs(a - b) for a, b in zip(blocks, new_blocks)) < 0.01:
            blocks = new_blocks
            break
        blocks = new_blocks

    blocks_int = [int(round(b)) for b in blocks]
    blocks_int = [max(B_min, b) for b in blocks_int]

    # Fix sum to exactly L.
    diff = L - sum(blocks_int)
    if diff > 0:
        blocks_int[0] += diff
    elif diff < 0:
        blocks_int[0] += diff
        if blocks_int[0] < B_min:
            blocks_int[0] = B_min
            remaining = L - sum(blocks_int)
            for i in range(K - 1, -1, -1):
                if remaining == 0:
                    break
                reducible = blocks_int[i] - B_min
                if reducible <= 0:
                    continue
                # remaining is negative here; apply negative adjustment.
                adjust = max(remaining, -reducible)
                blocks_int[i] += adjust
                remaining -= adjust

    if sum(blocks_int) != L:
        raise AssertionError(f"CAP sum mismatch: {sum(blocks_int)} != {L}")
    if any(b < B_min for b in blocks_int):
        raise AssertionError(f"CAP minimum violation: {blocks_int}")

    return blocks_int
