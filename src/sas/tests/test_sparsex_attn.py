import torch
import torch.nn.functional as F
import pytest
from sas.core.kernels.sparsex_attention import (
    sparsex_attn,
    sparsex_attn_ref,
)
from sas.tests._utils import get_tols
from sas.utils import ceildiv


def make_inputs(seqlens_q, seqlens_k, nheads_q, nheads_kv, head_dim, block_size, topk,
                dtype, device="cuda"):
    """Helper to create random inputs for sparsex attention tests.

    Args:
        seqlens_q: (bsz,) tensor of query sequence lengths
        seqlens_k: (bsz,) tensor of key/value sequence lengths
        nheads_q: number of query heads
        nheads_kv: number of key/value heads
        head_dim: dimension per head
        block_size: block size for sparse attention
        topk: number of top-k blocks per query
        dtype: data type
        device: torch device

    Returns:
        (q, k, v, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
         block_logsm, logsm_th, sm_scale)

        block_logsm: [nheads_q, total_q, max_kb]
        logsm_th:    [nheads_q, total_q]
    """
    cu_seqlens_q = torch.cumsum(F.pad(seqlens_q, (1, 0), value=0), dim=0).to(device)
    cu_seqlens_k = torch.cumsum(F.pad(seqlens_k, (1, 0), value=0), dim=0).to(device)
    max_sq = seqlens_q.max().item()
    max_sk = seqlens_k.max().item()

    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    q = torch.randn(total_q, nheads_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, nheads_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, nheads_kv, head_dim, device=device, dtype=dtype)

    sm_scale = head_dim ** -0.5

    max_kb = ceildiv(max_sk, block_size)
    block_logsm = 10 * torch.randn(nheads_q, total_q, max_kb,
        device=device, dtype=dtype)
    # Causal mask: per-sequence positions with right-aligned q/kv offset.
    # For query at intra-seq Q-position p in sequence i, its position in
    # K-coordinate space is p + (seqlens_k[i] - seqlens_q[i]).
    # kb_id >= that block should be masked (self + future).
    bsz = seqlens_q.size(0)
    q_k_offsets = seqlens_k - seqlens_q  # [bsz], right-aligned offset
    intra_k_pos = torch.zeros(total_q, device=device, dtype=torch.long)
    for i in range(bsz):
        s, e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        offset_i = q_k_offsets[i].item()
        intra_k_pos[s:e] = torch.arange(e - s, device=device) + offset_i
    block_ids_in_k = intra_k_pos // block_size  # [total_q], kv-block index for each query
    mask = block_ids_in_k[:, None] <= torch.arange(max_kb, device=device)[None, :]  # [total_q, max_kb]
    block_logsm.masked_fill_(mask.unsqueeze(0), float('-inf'))

    # logsm_th: [nheads_q, total_q], topk th largest logsm value per query
    actual_topk = min(topk, block_logsm.size(-1))
    logsm_th = torch.kthvalue(block_logsm, k=max_kb - actual_topk + 1, dim=-1).values
    # replace -inf with torch.finfo(dtype).min to prevent -inf positions being included
    # note that, -inf == -inf is True, but torch.infinfo(dtype).min == -inf is False
    logsm_th = torch.where(logsm_th == float('-inf'), torch.finfo(dtype).min, logsm_th)

    return (q, k, v, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
            block_logsm, logsm_th, sm_scale)


def _rand_seqlens(bsz, block_size, min_blocks=0, max_blocks=8, device="cuda"):
    """Generate random sequence lengths."""
    return torch.randint(
        min_blocks * block_size + 1, max_blocks * block_size + 1,
        (bsz,), dtype=torch.long, device=device,
    )


# ---- Edge case tests ----

def test_sparsex_attn_edge_cases():
    """Test various edge cases for correctness."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    dtype = torch.float16
    nheads_q = 4
    nheads_kv = 4
    head_dim = 64

    cases = [
        # (name, seqlens_q, seqlens_k, block_size, topk)
        # ---- Equal q/k lengths ----
        ("Single block, single seq", [128], [128], 128, 2),
        ("Two blocks exact", [256], [256], 128, 2),
        ("Partial last block", [200], [200], 128, 2),
        ("Batch varying lengths", [128, 256, 384], [128, 256, 384], 128, 2),
        ("Batch varying misaligned", [150, 300, 450], [150, 300, 450], 128, 3),
        ("Many blocks sparse", [128 * 8], [128 * 8], 128, 4),
        # ---- Unequal q/k lengths (KV cache / cross-attn) ----
        ("Single token decode, aligned", [1], [128 * 3], 128, 2),
        ("Single token decode, misaligned", [1], [128 * 3 + 20], 128, 2),
        ("Short q, long k", [10], [128 * 5], 128, 3),
        ("Batch mixed decode+prefill", [1, 128, 32], [128 * 4, 128 * 2, 128], 128, 2),
        ("q sub-block, k multi-block", [50], [128 * 3 + 50], 128, 2),
    ]

    for name, seqlens_q, seqlens_k, bs, topk in cases:
        seqlens_q_t = torch.tensor(seqlens_q, dtype=torch.long, device=device)
        seqlens_k_t = torch.tensor(seqlens_k, dtype=torch.long, device=device)

        (q, k, v, cu_q, cu_k, max_sq, max_sk,
         block_logsm, logsm_th, sm_scale) = make_inputs(
            seqlens_q_t, seqlens_k_t, nheads_q, nheads_kv, head_dim, bs, topk, dtype, device
        )

        try:
            out_ref = sparsex_attn_ref(
                q, k, v, cu_q, cu_k, max_sq, max_sk,
                bs, block_logsm, logsm_th, sm_scale,
            )

            out_ker, lse_ker = sparsex_attn(
                q, k, v, cu_q, cu_k, max_sq, max_sk,
                bs, block_logsm, logsm_th, sm_scale,
            )

            assert out_ref.shape == out_ker.shape, f"[{name}] Shape mismatch"
            assert not torch.isnan(out_ref).any(), f"[{name}] NaN in reference"
            assert not torch.isnan(out_ker).any(), f"[{name}] NaN in kernel"

            tols = get_tols(dtype, op_type="matmul")
            atol, rtol = tols['atol'], tols['rtol']
            torch.testing.assert_close(out_ker.float(), out_ref.float(), atol=atol, rtol=rtol)

            # Backward check
            q_ref = q.detach().clone().requires_grad_(True)
            k_ref = k.detach().clone().requires_grad_(True)
            v_ref = v.detach().clone().requires_grad_(True)
            out_ref2 = sparsex_attn_ref(
                q_ref, k_ref, v_ref, cu_q, cu_k, max_sq, max_sk,
                bs, block_logsm, logsm_th, sm_scale,
            )
            out_ref2.sum().backward()

            q_ker = q.detach().clone().requires_grad_(True)
            k_ker = k.detach().clone().requires_grad_(True)
            v_ker = v.detach().clone().requires_grad_(True)
            blsm_ker = block_logsm.detach().clone().requires_grad_(True)
            out_ker2, _ = sparsex_attn(
                q_ker, k_ker, v_ker, cu_q, cu_k, max_sq, max_sk,
                bs, blsm_ker, logsm_th, sm_scale,
            )
            out_ker2.sum().backward()

            torch.testing.assert_close(q_ker.grad, q_ref.grad, atol=atol, rtol=rtol)
            torch.testing.assert_close(k_ker.grad, k_ref.grad, atol=atol, rtol=rtol)
            torch.testing.assert_close(v_ker.grad, v_ref.grad, atol=atol, rtol=rtol)

        except Exception as e:
            pytest.fail(f"Edge case '{name}' failed: {e}")


# ---- Forward tests ----

@pytest.mark.parametrize("bsz", [1, 2, 3])
@pytest.mark.parametrize("nheads_q,nheads_kv", [(4, 4), (4, 1)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparsex_attn_fwd(bsz, nheads_q, nheads_kv, head_dim, block_size, topk, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(42)

    seqlens = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=6, device=device)

    (q, k, v, cu_q, cu_k, max_sq, max_sk,
     block_logsm, logsm_th, sm_scale) = make_inputs(
        seqlens, seqlens, nheads_q, nheads_kv, head_dim, block_size, topk, dtype, device
    )

    out_ref = sparsex_attn_ref(
        q, k, v, cu_q, cu_k, max_sq, max_sk,
        block_size, block_logsm, logsm_th, sm_scale,
    )

    out_ker, _ = sparsex_attn(
        q, k, v, cu_q, cu_k, max_sq, max_sk,
        block_size, block_logsm, logsm_th, sm_scale,
    )

    assert out_ref.shape == out_ker.shape, f"Shape mismatch: ref={out_ref.shape}, ker={out_ker.shape}"

    tols = get_tols(dtype, op_type="matmul")
    atol, rtol = tols['atol'], tols['rtol']
    torch.testing.assert_close(out_ker.float(), out_ref.float(), atol=atol, rtol=rtol)


# ---- Backward tests ----

@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("nheads_q,nheads_kv", [(4, 4), (4, 1)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparsex_attn_bwd(bsz, nheads_q, nheads_kv, head_dim, block_size, topk, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(42)

    seqlens = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=5, device=device)

    (q, k, v, cu_q, cu_k, max_sq, max_sk,
     block_logsm, logsm_th, sm_scale) = make_inputs(
        seqlens, seqlens, nheads_q, nheads_kv, head_dim, block_size, topk, dtype, device
    )

    # Reference
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    blsm_ref = block_logsm.detach().clone().requires_grad_(True)
    out_ref = sparsex_attn_ref(
        q_ref, k_ref, v_ref, cu_q, cu_k, max_sq, max_sk,
        block_size, blsm_ref, logsm_th, sm_scale,
    )
    out_ref.sum().backward()

    # Kernel
    q_ker = q.detach().clone().requires_grad_(True)
    k_ker = k.detach().clone().requires_grad_(True)
    v_ker = v.detach().clone().requires_grad_(True)
    blsm_ker = block_logsm.detach().clone().requires_grad_(True)
    out_ker, _ = sparsex_attn(
        q_ker, k_ker, v_ker, cu_q, cu_k, max_sq, max_sk,
        block_size, blsm_ker, logsm_th, sm_scale,
    )
    out_ker.sum().backward()

    tols = get_tols(dtype, op_type="matmul")
    atol, rtol = tols['atol'], tols['rtol']

    torch.testing.assert_close(q_ker.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_ker.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_ker.grad, v_ref.grad, atol=atol, rtol=rtol)

    # Compare block_logsm gradient
    if blsm_ker.grad is not None and blsm_ref.grad is not None:
        torch.testing.assert_close(blsm_ker.grad, blsm_ref.grad, atol=atol, rtol=rtol)


# ---- Large-tensor smoke test (real 128K / 32-head setting) ----
# Total 128K tokens, nheads_q=32, head_dim=128, block_size=64. Reproduces the
# real workload where the sequence is several docs packed together (varlen
# with multiple cu_seqlens segments). Just checks forward + backward run.

def _free_gpu_bytes():
    free, _ = torch.cuda.mem_get_info()
    return free


# "single": one 128K segment; "packed_*": multiple docs packed into 128K total.
@pytest.mark.parametrize("layout", ["single", "packed_equal", "packed_varlen"])
def test_sparsex_attn_large_128k(layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    dtype = torch.float16
    # Release cached blocks from previous parametrized cases so the free-memory
    # check below sees the true amount available, not what's held in the pool.
    torch.cuda.empty_cache()

    total = 128 * 1024
    nheads_q = 32
    nheads_kv = 8
    head_dim = 128
    block_size = 64

    if layout == "single":
        seqlens = torch.tensor([total], dtype=torch.long, device=device)
    elif layout == "packed_equal":
        # 16 docs of 8K each, packed into 128K
        seqlens = torch.full((16,), total // 16, dtype=torch.long, device=device)
    else:  # packed_varlen: one long doc + many shorter ones, still summing to 128K
        torch.manual_seed(1)
        parts = [64 * 1024]  # one long 64K doc keeps max_seqlen large
        remaining = total - parts[0]
        while remaining > 0:
            s = min(remaining, int(torch.randint(512, 4096, (1,)).item()))
            parts.append(s)
            remaining -= s
        seqlens = torch.tensor(parts, dtype=torch.long, device=device)

    max_blocks = ceildiv(seqlens.max().item(), block_size)
    topk = max_blocks  # select all blocks -> dense causal

    needed = int(total * nheads_q * (head_dim + max_blocks) * 2 * 4)
    if _free_gpu_bytes() < needed:
        pytest.skip(f"needs ~{needed / 1e9:.1f} GB free GPU memory")

    (q, k, v, cu_q, cu_k, max_sq, max_sk,
     block_logsm, logsm_th, sm_scale) = make_inputs(
        seqlens, seqlens, nheads_q, nheads_kv, head_dim, block_size, topk, dtype, device
    )
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    out, _ = sparsex_attn(
        q, k, v, cu_q, cu_k, max_sq, max_sk,
        block_size, block_logsm, logsm_th, sm_scale,
    )
    out.sum().backward()

    assert out.shape == (total, nheads_q, head_dim)
    assert q.grad is not None and k.grad is not None and v.grad is not None


# ---- Shape tests ----

@pytest.mark.parametrize("bsz", [1, 3])
@pytest.mark.parametrize("nheads_q", [4])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_sparsex_attn_shapes(bsz, nheads_q, head_dim, block_size, dtype):
    """Test that output shapes are correct for various configurations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"

    seqlens = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=4, device=device)
    topk = 2

    (q, k, v, cu_q, cu_k, max_sq, max_sk,
     block_logsm, logsm_th, sm_scale) = make_inputs(
        seqlens, seqlens, nheads_q, nheads_q, head_dim, block_size, topk, dtype, device
    )

    out, lse = sparsex_attn(
        q, k, v, cu_q, cu_k, max_sq, max_sk,
        block_size, block_logsm, logsm_th, sm_scale,
    )

    total_q = q.size(0)
    assert out.shape == (total_q, nheads_q, head_dim), \
        f"Expected output shape ({total_q}, {nheads_q}, {head_dim}), got {out.shape}"
    assert lse.shape == (nheads_q, total_q), \
        f"Expected lse shape ({nheads_q}, {total_q}), got {lse.shape}"


# ---- KV cache tests (seqlens_q != seqlens_k) ----

@pytest.mark.parametrize("nheads_q,nheads_kv", [(4, 4), (4, 1)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.xfail(reason="q/kv unequal-length kernel support not yet debugged")
def test_sparsex_attn_kv_cache_fwd(nheads_q, nheads_kv, head_dim, block_size, topk, dtype):
    """Test forward with KV caching: seqlens_q << seqlens_k, right-aligned causal mask."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(42)

    bsz = 3
    seqlens_k = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=8, device=device)
    seqlens_q = torch.randint(1, block_size + 1, (bsz,), dtype=torch.long, device=device)
    seqlens_q = torch.min(seqlens_q, seqlens_k)


    (q, k, v, cu_q, cu_k, max_sq, max_sk,
     block_logsm, logsm_th, sm_scale) = make_inputs(
        seqlens_q, seqlens_k, nheads_q, nheads_kv, head_dim, block_size, topk, dtype, device
    )

    out_ref = sparsex_attn_ref(
        q, k, v, cu_q, cu_k, max_sq, max_sk,
        block_size, block_logsm, logsm_th, sm_scale,
    )

    out_ker, _ = sparsex_attn(
        q, k, v, cu_q, cu_k, max_sq, max_sk,
        block_size, block_logsm, logsm_th, sm_scale,
    )

    assert out_ref.shape == out_ker.shape, f"Shape mismatch: ref={out_ref.shape}, ker={out_ker.shape}"

    tols = get_tols(dtype, op_type="matmul")
    # The ref uses batched 4D matmul with gate addition in original dtype, while the
    # kernel uses tiled fp32 accumulation. Use 3x tolerance to account for this.
    atol, rtol = tols['atol'], tols['rtol']
    torch.testing.assert_close(out_ker.float(), out_ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("nheads_q,nheads_kv", [(4, 4), (4, 1)])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.xfail(reason="q/kv unequal-length kernel support not yet debugged")
def test_sparsex_attn_kv_cache_bwd(nheads_q, nheads_kv, head_dim, block_size, topk, dtype):
    """Test backward with KV caching: seqlens_q != seqlens_k, right-aligned causal mask."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(42)

    bsz = 3
    seqlens_k = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=6, device=device)
    seqlens_q = torch.randint(1, block_size + 1, (bsz,), dtype=torch.long, device=device)
    seqlens_q = torch.min(seqlens_q, seqlens_k)


    (q, k, v, cu_q, cu_k, max_sq, max_sk,
     block_logsm, logsm_th, sm_scale) = make_inputs(
        seqlens_q, seqlens_k, nheads_q, nheads_kv, head_dim, block_size, topk, dtype, device
    )

    # Reference
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = sparsex_attn_ref(
        q_ref, k_ref, v_ref, cu_q, cu_k, max_sq, max_sk,
        block_size, block_logsm, logsm_th, sm_scale,
    )
    out_ref.sum().backward()

    # Kernel
    q_ker = q.detach().clone().requires_grad_(True)
    k_ker = k.detach().clone().requires_grad_(True)
    v_ker = v.detach().clone().requires_grad_(True)
    blsm_ker = block_logsm.detach().clone().requires_grad_(True)
    out_ker, _ = sparsex_attn(
        q_ker, k_ker, v_ker, cu_q, cu_k, max_sq, max_sk,
        block_size, blsm_ker, logsm_th, sm_scale,
    )
    out_ker.sum().backward()

    tols = get_tols(dtype, op_type="matmul")
    atol, rtol = tols['atol'], tols['rtol']

    torch.testing.assert_close(q_ker.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_ker.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_ker.grad, v_ref.grad, atol=atol, rtol=rtol)
