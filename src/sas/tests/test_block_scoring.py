import torch
import torch.nn.functional as F
import pytest
from sas.core.kernels.block_scoring import (
    fused_causal_matmul_logsoftmax,
    fused_causal_matmul_logsoftmax_ref,
)
from sas.tests._utils import get_tols
from sas.utils import ceildiv


def make_inputs(seqlen_q:torch.LongTensor, seqlen_k:torch.LongTensor,
                nheads:int, head_dim:int, block_size:int, dtype:torch.dtype, device="cuda"):
    """Helper to create random inputs for block scoring tests.

    Args:
        seqlen_q: (bsz,) tensor of query sequence lengths
        seqlen_k: (bsz,) tensor of key sequence lengths (for kb)
        nheads: number of attention heads
        head_dim: dimension of each head
        dtype: data type for q and kb tensors
        device: torch device
    Returns:
        (q, kb, cu_seqlens_q, cu_seqlens_kb, q_offs, max_sq, max_skb) \
        (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]): \
            Tensors for block scoring tests.
    """
    q_offs = seqlen_k - seqlen_q
    seqlen_kb = ceildiv(seqlen_k, block_size)
    cu_seqlens_q = torch.cumsum(F.pad(seqlen_q, (1, 0), value=0), dim=0).to(device)
    cu_seqlens_kb = torch.cumsum(F.pad(seqlen_kb, (1, 0), value=0), dim=0).to(device)
    max_sq = seqlen_q.max().item()
    max_skb = seqlen_kb.max().item()
    q = torch.randn(cu_seqlens_q[-1].item(), nheads, head_dim, device=device, dtype=dtype).requires_grad_(True)
    kb = torch.randn(cu_seqlens_kb[-1].item(), nheads, head_dim, device=device, dtype=dtype).requires_grad_(True)
    
    return q, kb, cu_seqlens_q, cu_seqlens_kb, q_offs, max_sq, max_skb


def _rand_seqlens(bsz, block_size, min_blocks=1, max_blocks=8, device="cuda"):
    """Generate random sequence lengths (multiples of 1, NOT block_size)."""
    return torch.randint(
        min_blocks * block_size, max_blocks * block_size + 1,
        (bsz,), dtype=torch.long, device=device,
    )


# ---- Edge case tests ----

def test_fused_causal_matmul_logsoftmax_edge_cases():
    """Test various edge cases for correctness."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    dtype = torch.float16
    nheads = 2
    head_dim = 64

    cases = [
        # (name, bsz, seqlens_k, seqlens_q, block_size)
        # first token in different blocks
        ("First token in the first block (Single token)", 1, [1], [1], 64),
        ("First token of the second block", 1, [65], [65], 64),
        ("First token of the 10th block", 1, [64 * 10+1], [64 * 10+1], 64),
        # last token in different blocks
        ("Last token of the first block", 1, [64], [64], 64),
        ("Last token of the second block", 1, [128], [128], 64),
        ("Last token of the 10th block", 1, [64 * 10], [64 * 10], 64),
        # middle token in different blocks
        ("Middle token of the first block", 1, [31], [31], 64),
        ("Middle token of the second block", 1, [64 + 15], [64 + 15], 64),
        ("Middle token of the 10th block", 1, [64 * 10+33], [64 * 10+33], 64),
        
        # batched cases with varying sequence lengths (seqlens_q == seqlens_k for simplicity, but varying across batch)
        ("Batched varying lengths - blocks aligned", 4, [64, 128, 256, 320], [64, 128, 256, 320], 64),
        ("Batched varying lengths - blocks misaligned", 3, [10, 70, 130], [10, 70, 130], 64),
        ("Batched varying lengths - hybrid aligned & misaligned", 4, [5, 64, 150, 300], [5, 64, 150, 300], 64),
        ("All sequences single block", 2, [64, 32], [64, 32], 64),

        # super long sequences (multiple blocks) to test multi-block logic and numerical stability
        ("Super long sequence - aligned to block", 1, [64 * 1024], [64 * 1024], 64),
        ("Super long sequence - misaligned to block", 1, [64 * 1024 + 15], [64 * 1024 + 15], 64),

        # cases with seqlen_q != seqlen_k (KV cache decode scenarios)
        ("Fits in one block, extend multiple", 1, [52], [10], 64),
        ("Single token decode, historical context block aligned", 1, [64 * 3], [1], 64),
        ("Single token decode, historical context block misaligned", 1, [64 * 3 + 20], [1], 64),   
        ("KV cache decode batch", 3, [64 * 4 + 1, 64 * 2 + 5, 64 * 6], [1, 1, 1], 64),
        ("Mixed decode + prefill", 3, [12, 128 * 2 + 5, 128], [1, 128, 32], 128),
        ("Single query sees many blocks", 1, [64 * 8+31], [1], 64),
    ]

    for name, bsz, seqlens_k, seqlens_q, bs in cases:
        seqlen_q = torch.tensor(seqlens_q, dtype=torch.long, device=device)
        seqlen_k = torch.tensor(seqlens_k, dtype=torch.long, device=device)

        q, kb, cu_q, cu_kb, q_pos_offset, max_sq, max_skb = make_inputs(
            seqlen_q, seqlen_k, nheads, head_dim, bs, dtype, device
        )
        sm_scale = head_dim ** -0.5

        try:
            q_ref = q.detach().clone().requires_grad_(True)
            kb_ref = kb.detach().clone().requires_grad_(True)
            ref_out = fused_causal_matmul_logsoftmax_ref(
                q_ref, kb_ref, cu_q, cu_kb, max_sq, max_skb, bs, sm_scale,
                q_position_offset=q_pos_offset,
            )

            q_ker = q.detach().clone().requires_grad_(True)
            kb_ker = kb.detach().clone().requires_grad_(True)
            ker_out = fused_causal_matmul_logsoftmax(
                q_ker, kb_ker, cu_q, cu_kb, max_sq, max_skb, bs, sm_scale,
                q_position_offset=q_pos_offset,
            )

            tols = get_tols(dtype, op_type="matmul")
            atol, rtol = tols['atol'], tols['rtol']

            # shape check
            assert ref_out.shape == ker_out.shape, f"[{name}] Shape mismatch"

            # check nan
            assert not torch.isnan(ref_out).any(), f"[{name}] NaN in reference output"
            assert not torch.isnan(ker_out).any(), f"[{name}] NaN in kernel output"

            # check forward values (ref now outputs strict -inf at masked positions,
            # so we can compare directly without masking)
            torch.testing.assert_close(ker_out.float(), ref_out.float(), atol=atol, rtol=rtol)

            # Test backward
            # Mask grad_output at -inf positions: ref uses torch.log_softmax whose
            # backward includes g_j from -inf positions in delta=Σg_j, while the
            # Triton kernel skips them via causal_mask. Zeroing these g_j aligns both.
            grad_output = torch.randn_like(ker_out)
            grad_output[ker_out <= torch.finfo(dtype).min] = 0.0

            ref_out.backward(grad_output)
            ker_out.backward(grad_output)

            torch.testing.assert_close(q_ker.grad, q_ref.grad, atol=atol, rtol=rtol)
            torch.testing.assert_close(kb_ker.grad, kb_ref.grad, atol=atol, rtol=rtol)

        except Exception as e:
            pytest.fail(f"Edge case '{name}' failed: {e}")

# ---- Forward tests (training) ----

@pytest.mark.parametrize("bsz", [1, 2, 3])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_causal_matmul_logsoftmax_fwd(bsz, nheads, head_dim, block_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"

    seqlen_q = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=6, device=device)
    seqlen_k = seqlen_q.clone()

    q, kb, cu_q, cu_kb, q_offs, max_sq, max_skb = make_inputs(
        seqlen_q, seqlen_k, nheads, head_dim, block_size, dtype, device
    )
    sm_scale = head_dim ** -0.5

    q_ref = q.detach().clone().requires_grad_(True)
    kb_ref = kb.detach().clone().requires_grad_(True)
    ref_out = fused_causal_matmul_logsoftmax_ref(
        q_ref, kb_ref, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale
    )

    ker_out = fused_causal_matmul_logsoftmax(
        q, kb, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale
    )

    assert ref_out.shape == ker_out.shape, f"Shape mismatch: ref={ref_out.shape}, ker={ker_out.shape}"

    tols = get_tols(dtype, op_type="matmul")
    atol, rtol = tols['atol'], tols['rtol']

    torch.testing.assert_close(ker_out.float(), ref_out.float(), atol=atol, rtol=rtol)


# ---- Backward tests (training) ----

@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_causal_matmul_logsoftmax_bwd(bsz, nheads, head_dim, block_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"

    seqlen_q = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=5, device=device)
    seqlen_k = seqlen_q.clone()

    q, kb, cu_q, cu_kb, q_offs, max_sq, max_skb = make_inputs(
        seqlen_q, seqlen_k, nheads, head_dim, block_size, dtype, device
    )
    sm_scale = head_dim ** -0.5

    q_ref = q.detach().clone().requires_grad_(True)
    kb_ref = kb.detach().clone().requires_grad_(True)
    ref_out = fused_causal_matmul_logsoftmax_ref(
        q_ref, kb_ref, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale
    )

    q_ker = q.detach().clone().requires_grad_(True)
    kb_ker = kb.detach().clone().requires_grad_(True)
    ker_out = fused_causal_matmul_logsoftmax(
        q_ker, kb_ker, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale
    )

    # Mask grad_output at -inf positions (see edge_cases test for explanation)
    grad_output = torch.randn_like(ker_out)
    grad_output[ker_out <= torch.finfo(dtype).min] = 0.0

    ref_out.backward(grad_output)
    ker_out.backward(grad_output)

    tols = get_tols(dtype, op_type="matmul")
    atol, rtol = tols['atol'], tols['rtol']

    torch.testing.assert_close(q_ker.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(kb_ker.grad, kb_ref.grad, atol=atol, rtol=rtol)


# ---- Shape tests ----

@pytest.mark.parametrize("bsz", [1, 3])
@pytest.mark.parametrize("nheads", [2])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_causal_matmul_logsoftmax_shapes(bsz, nheads, head_dim, block_size, dtype):
    """Test that output shapes are correct for various configurations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"

    seqlen_q = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=4, device=device)
    seqlen_k = seqlen_q.clone()

    q, kb, cu_q, cu_kb, q_offs, max_sq, max_skb = make_inputs(
        seqlen_q, seqlen_k, nheads, head_dim, block_size, dtype, device
    )
    sm_scale = head_dim ** -0.5

    out = fused_causal_matmul_logsoftmax(
        q, kb, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale
    )

    total_q = q.size(0)
    assert out.shape == (nheads, total_q, max_skb), \
        f"Expected shape ({nheads}, {total_q}, {max_skb}), got {out.shape}"

    # First block_size positions in each sequence should have all -inf
    for ib in range(bsz):
        q_start = cu_q[ib].item()
        first_row = out[:, q_start, :]
        assert (first_row == float('-inf')).all(), \
            f"First query position should have all -inf (no previous blocks), got {first_row}"


# ---- KV caching tests (seqlen_q != seqlen_kb) ----

@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_causal_matmul_logsoftmax_kv_cache_fwd(nheads, head_dim, block_size, dtype):
    """Test forward with KV caching: seqlen_q << seqlen_kb, right-aligned causal mask."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"

    bsz = 3
    # Random context lengths (2-8 blocks), random new query lengths (1 to block_size)
    seqlen_k = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=8, device=device)
    seqlen_q = torch.randint(1, block_size + 1, (bsz,), dtype=torch.long, device=device)
    # Ensure seqlen_q <= seqlen_k
    seqlen_q = torch.min(seqlen_q, seqlen_k)

    q, kb, cu_q, cu_kb, q_pos_offset, max_sq, max_skb = make_inputs(
        seqlen_q, seqlen_k, nheads, head_dim, block_size, dtype, device
    )
    sm_scale = head_dim ** -0.5

    q_ref = q.detach().clone().requires_grad_(True)
    kb_ref = kb.detach().clone().requires_grad_(True)
    ref_out = fused_causal_matmul_logsoftmax_ref(
        q_ref, kb_ref, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale,
        q_position_offset=q_pos_offset,
    )

    ker_out = fused_causal_matmul_logsoftmax(
        q, kb, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale,
        q_position_offset=q_pos_offset,
    )

    assert ref_out.shape == ker_out.shape, f"Shape mismatch: ref={ref_out.shape}, ker={ker_out.shape}"

    tols = get_tols(dtype, op_type="matmul")
    atol, rtol = tols['atol'], tols['rtol']

    torch.testing.assert_close(ker_out.float(), ref_out.float(), atol=atol, rtol=rtol)

    # Verify causal mask visibility: for the first query token of each sequence,
    # blocks with kb_id >= (q_pos_offset + 0) // block_size should be masked.
    for ib in range(bsz):
        q_start = cu_q[ib].item()
        row = ker_out[:, q_start, :]  # [nheads, max_skb]
        seqlen_kb_ib = cu_kb[ib + 1].item() - cu_kb[ib].item()
        expected_visible = q_pos_offset[ib].item() // block_size
        for h in range(nheads):
            for kb_id in range(seqlen_kb_ib):
                val = row[h, kb_id].item()
                if kb_id < expected_visible:
                    assert val > float('-inf'), \
                        f"seq={ib}, head={h}, kb_id={kb_id} should be visible (expected_visible={expected_visible}), got -inf"
                else:
                    assert val == float('-inf'), \
                        f"seq={ib}, head={h}, kb_id={kb_id} should be masked (expected_visible={expected_visible}), got {val}"


@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_causal_matmul_logsoftmax_kv_cache_bwd(nheads, head_dim, block_size, dtype):
    """Test backward with KV caching: seqlen_q != seqlen_kb, right-aligned causal mask."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"

    bsz = 3
    seqlen_k = _rand_seqlens(bsz, block_size, min_blocks=2, max_blocks=6, device=device)
    seqlen_q = torch.randint(1, block_size + 1, (bsz,), dtype=torch.long, device=device)
    seqlen_q = torch.min(seqlen_q, seqlen_k)

    q, kb, cu_q, cu_kb, q_pos_offset, max_sq, max_skb = make_inputs(
        seqlen_q, seqlen_k, nheads, head_dim, block_size, dtype, device
    )
    sm_scale = head_dim ** -0.5

    q_ref = q.detach().clone().requires_grad_(True)
    kb_ref = kb.detach().clone().requires_grad_(True)
    ref_out = fused_causal_matmul_logsoftmax_ref(
        q_ref, kb_ref, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale,
        q_position_offset=q_pos_offset,
    )

    q_ker = q.detach().clone().requires_grad_(True)
    kb_ker = kb.detach().clone().requires_grad_(True)
    ker_out = fused_causal_matmul_logsoftmax(
        q_ker, kb_ker, cu_q, cu_kb, max_sq, max_skb, block_size, sm_scale,
        q_position_offset=q_pos_offset,
    )

    # Mask grad_output at -inf positions (see edge_cases test for explanation)
    grad_output = torch.randn_like(ker_out)
    grad_output[ker_out <= torch.finfo(dtype).min] = 0.0

    ref_out.backward(grad_output)
    ker_out.backward(grad_output)

    tols = get_tols(dtype, op_type="matmul")
    atol, rtol = tols['atol'], tols['rtol']

    torch.testing.assert_close(q_ker.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(kb_ker.grad, kb_ref.grad, atol=atol, rtol=rtol)


# ---- Large-tensor smoke test (real 128K / 32-head setting) ----
# Total 128K tokens, nheads=32, head_dim=128, block_size=64. Reproduces the real
# workload where the sequence is several docs packed together (varlen with
# multiple cu_seqlens segments). Just checks forward + backward run.

def _free_gpu_bytes():
    free, _ = torch.cuda.mem_get_info()
    return free


@pytest.mark.parametrize("layout", ["single", "packed_equal", "packed_varlen"])
def test_fused_causal_matmul_logsoftmax_large_128k(layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    dtype = torch.float16
    # Release cached blocks from previous parametrized cases so the free-memory
    # check below sees the true amount available, not what's held in the pool.
    torch.cuda.empty_cache()

    total = 128 * 1024
    nheads = 32
    head_dim = 128
    block_size = 64

    if layout == "single":
        seqlens = torch.tensor([total], dtype=torch.long, device=device)
    elif layout == "packed_equal":
        seqlens = torch.full((16,), total // 16, dtype=torch.long, device=device)
    else:  # packed_varlen
        torch.manual_seed(1)
        parts = [64 * 1024]
        remaining = total - parts[0]
        while remaining > 0:
            s = min(remaining, int(torch.randint(512, 4096, (1,)).item()))
            parts.append(s)
            remaining -= s
        seqlens = torch.tensor(parts, dtype=torch.long, device=device)

    max_skb = ceildiv(seqlens.max().item(), block_size)
    needed = int(nheads * total * max_skb * 2 * 2.5)
    if _free_gpu_bytes() < needed:
        pytest.skip(f"needs ~{needed / 1e9:.1f} GB free GPU memory")

    q, kb, cu_q, cu_kb, q_offs, max_sq, max_skb_ = make_inputs(
        seqlens, seqlens, nheads, head_dim, block_size, dtype, device
    )
    sm_scale = head_dim ** -0.5

    out = fused_causal_matmul_logsoftmax(
        q, kb, cu_q, cu_kb, max_sq, max_skb_, block_size, sm_scale
    )
    grad = torch.randn_like(out)
    grad[out <= torch.finfo(dtype).min] = 0.0
    out.backward(grad)

    assert q.grad is not None and kb.grad is not None
