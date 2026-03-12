"""Fused Triton kernels for small-matrix batched operations in CRBA/ABA."""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_xtmx_add_kernel(
        Xt_ptr,
        M_ptr,
        X_ptr,
        Out_ptr,
        B,
        xt_bs,
        m_bs,
        x_bs,
        out_bs,
        BLOCK_B: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """Compute Out[b] += (Xt[b] @ M[b] @ X[b]) for 6x6 matrices.

        Each program handles BLOCK_B batch elements. The two 6x6 matmuls are
        fully unrolled and processed row-by-row to keep register pressure low.
        No atomics needed — each batch element is independent.
        """
        pid = tl.program_id(0)
        b_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        mask = b_offs < B

        # Process output matrix row by row
        for i in range(6):
            # Step 1: Compute row i of tmp = Xt @ M
            # tmp[j] = sum_k Xt[i,k] * M[k,j]  for j = 0..5
            tmp0 = tl.zeros([BLOCK_B], dtype=DTYPE)
            tmp1 = tl.zeros([BLOCK_B], dtype=DTYPE)
            tmp2 = tl.zeros([BLOCK_B], dtype=DTYPE)
            tmp3 = tl.zeros([BLOCK_B], dtype=DTYPE)
            tmp4 = tl.zeros([BLOCK_B], dtype=DTYPE)
            tmp5 = tl.zeros([BLOCK_B], dtype=DTYPE)

            for k in range(6):
                xt_val = tl.load(Xt_ptr + b_offs * xt_bs + i * 6 + k, mask=mask, other=0.0)
                tmp0 += xt_val * tl.load(M_ptr + b_offs * m_bs + k * 6 + 0, mask=mask, other=0.0)
                tmp1 += xt_val * tl.load(M_ptr + b_offs * m_bs + k * 6 + 1, mask=mask, other=0.0)
                tmp2 += xt_val * tl.load(M_ptr + b_offs * m_bs + k * 6 + 2, mask=mask, other=0.0)
                tmp3 += xt_val * tl.load(M_ptr + b_offs * m_bs + k * 6 + 3, mask=mask, other=0.0)
                tmp4 += xt_val * tl.load(M_ptr + b_offs * m_bs + k * 6 + 4, mask=mask, other=0.0)
                tmp5 += xt_val * tl.load(M_ptr + b_offs * m_bs + k * 6 + 5, mask=mask, other=0.0)

            # Step 2: Compute row i of result = tmp @ X, accumulate into Out
            # result[col] = sum_j tmp[j] * X[j, col]  for col = 0..5
            for col in range(6):
                result = (
                    tmp0 * tl.load(X_ptr + b_offs * x_bs + 0 * 6 + col, mask=mask, other=0.0)
                    + tmp1 * tl.load(X_ptr + b_offs * x_bs + 1 * 6 + col, mask=mask, other=0.0)
                    + tmp2 * tl.load(X_ptr + b_offs * x_bs + 2 * 6 + col, mask=mask, other=0.0)
                    + tmp3 * tl.load(X_ptr + b_offs * x_bs + 3 * 6 + col, mask=mask, other=0.0)
                    + tmp4 * tl.load(X_ptr + b_offs * x_bs + 4 * 6 + col, mask=mask, other=0.0)
                    + tmp5 * tl.load(X_ptr + b_offs * x_bs + 5 * 6 + col, mask=mask, other=0.0)
                )
                out_off = b_offs * out_bs + i * 6 + col
                old_val = tl.load(Out_ptr + out_off, mask=mask, other=0.0)
                tl.store(Out_ptr + out_off, old_val + result, mask=mask)


def fused_xtmx_add(Xt, M, X, out):
    """Compute out += Xt @ M @ X for batched 6x6 matrices using a fused Triton kernel.

    Replaces two cuBLAS bmm calls with a single fused kernel. Each thread
    handles one batch element, computing the full 6x6 double-matmul in
    registers with zero intermediate allocations.

    Args:
        Xt: (B, 6, 6) — left transform (may be strided in batch dim)
        M:  (B, 6, 6) — middle matrix (contiguous)
        X:  (B, 6, 6) — right transform (may be strided in batch dim)
        out: (B, 6, 6) — output, accumulated in-place (contiguous)
    """
    if not HAS_TRITON or not Xt.is_cuda:
        out += Xt @ M @ X
        return

    B = Xt.shape[0]
    BLOCK_B = 128
    grid = ((B + BLOCK_B - 1) // BLOCK_B,)

    triton_dtype = tl.float64 if Xt.dtype == torch.float64 else tl.float32
    _fused_xtmx_add_kernel[grid](
        Xt,
        M,
        X,
        out,
        B,
        Xt.stride(0),
        M.stride(0),
        X.stride(0),
        out.stride(0),
        BLOCK_B=BLOCK_B,
        DTYPE=triton_dtype,
    )
