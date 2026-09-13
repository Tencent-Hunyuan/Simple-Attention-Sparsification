from typing import Optional
import torch
import triton
import triton.language as tl
from fla.utils import autocast_custom_bwd, autocast_custom_fwd
from sas.utils import maybe_contiguous


@triton.jit
def _fused_causal_matmul_logsoftmax_fwd_kernel(
    gOut,           # [nheads, total_q_len, max_seqlen_kb], output logsoftmax
    gLSE,           # [nheads, total_q_len], log-sum-exp per row
    gQ,             # [total_q_len, nheads, head_dim]
    gKB,            # [total_kb_len, nheads, head_dim]
    gCumSeqLensQ,   # [bsz + 1]
    gCumSeqLensKB,  # [bsz + 1]
    gQPosOffset,    # [bsz], per-sequence offset for right-aligned causal mask
    sm_scale,       # float
    stride_q_t, stride_q_h, stride_q_d,
    stride_kb_t, stride_kb_h, stride_kb_d,
    stride_out_h, stride_out_t, stride_out_k,
    stride_lse_h, stride_lse_t,
    kBlkSize: tl.constexpr,
    kHeadDim: tl.constexpr,
    kTileQ: tl.constexpr,
    kTileKB: tl.constexpr,
):
    iQTile, iHead, iSeq = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    bos_q = tl.load(gCumSeqLensQ + iSeq).to(tl.int32)
    eos_q = tl.load(gCumSeqLensQ + iSeq + 1).to(tl.int32)
    bos_k = tl.load(gCumSeqLensKB + iSeq).to(tl.int32)
    eos_k = tl.load(gCumSeqLensKB + iSeq + 1).to(tl.int32)
    seqlen_q = eos_q - bos_q
    seqlen_k = eos_k - bos_k
    q_offset = tl.load(gQPosOffset + iSeq).to(tl.int32)

    # int64 copies for pointer arithmetic only (prevent int32 offset overflow on
    # gOut [nheads, total_q, max_seqlen_kb] where iHead*stride_out_h or
    # bos_q*stride_out_t can exceed INT32_MAX at long sequence length).
    bos_q_i64 = bos_q.to(tl.int64)
    bos_k_i64 = bos_k.to(tl.int64)
    iHead_i64 = iHead.to(tl.int64)

    bot_q = iQTile * kTileQ # TODO: optimization, compiler hint: multiple of kTileQ
    if bot_q >= seqlen_q:
        return

    offs_m = bot_q + tl.arange(0, kTileQ)
    mask_m = offs_m < seqlen_q
    offs_d = tl.arange(0, kHeadDim)

    # Load Q tile: [kTileQ, kHeadDim] using strides
    q_ptrs = gQ + (bos_q_i64 + offs_m[:, None]) * stride_q_t + iHead_i64 * stride_q_h + offs_d[None, :] * stride_q_d
    q_val = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # ---- Pass 1: compute logits, write to output, track row_max + sum_exp ----
    row_max = tl.full([kTileQ], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([kTileQ], dtype=tl.float32)

    # TODO: (optimization) early exit: as we know the last query position is q_offset + bot_q + kTileQ - 1, we can skip some kb tiles
    # TODO: (optimization) off-band-> on-band two stage computation; only on-band stage requires causal mask
    for bok in range(0, seqlen_k, kTileKB):
        offs_k = bok + tl.arange(0, kTileKB)
        mask_k = offs_k < seqlen_k # TODO: (optimization) it seems that this is not needed, this has been guaranteed by causal mask.

        # Load KB tile: [kTileKB, kHeadDim] using strides
        kb_ptrs = gKB + (bos_k_i64 + offs_k[:, None]) * stride_kb_t + iHead_i64 * stride_kb_h + offs_d[None, :] * stride_kb_d
        kb_val = tl.load(kb_ptrs, mask=mask_k[:, None], other=0.0)

        # Compute logits: [kTileQ, kTileKB]
        # TODO:  (optimization) sm_scale -> sm_scale * log2e, and then use tl.exp2, for better efficiency
        # https://github.com/triton-lang/triton/issues/2893#issuecomment-1909910123
        logits = tl.dot(q_val, tl.trans(kb_val)) * sm_scale

        # Right-aligned causal mask: kb_id >= (q_offset + q_pos) // block_size -> mask out
        causal_limit = (q_offset + offs_m) // kBlkSize  # [kTileQ], per-row causal limit
        causal_mask = offs_k[None, :] < causal_limit[:, None]  # [kTileQ, kTileKB]
        valid_mask = mask_m[:, None] & mask_k[None, :] & causal_mask
        logits = tl.where(valid_mask, logits, float('-inf'))

        # Store logits to output (will be overwritten with logsoftmax in pass 2)
        # Output layout: [nheads, total_q_len, max_seqlen_kb]
        out_ptrs = gOut + iHead_i64 * stride_out_h + (bos_q_i64 + offs_m[:, None]) * stride_out_t + offs_k[None, :] * stride_out_k
        tl.store(out_ptrs, logits.to(gOut.dtype.element_ty), mask=valid_mask)

        # Online row_max + sum_exp
        tile_max = tl.max(logits, axis=1)  # [kTileQ]
        new_max = tl.maximum(row_max, tile_max)
        sum_exp = sum_exp * tl.exp(row_max - new_max) + tl.sum(tl.exp(logits - new_max[:, None]), axis=1)
        row_max = new_max

    # Compute LSE: row_max + log(sum_exp)
    # Handle all-masked rows (sum_exp == 0)
    lse = tl.where(sum_exp > 0, row_max + tl.log(sum_exp), float('-inf'))

    # Store LSE: [nheads, total_q_len]
    lse_ptrs = gLSE + iHead_i64 * stride_lse_h + (bos_q_i64 + offs_m) * stride_lse_t
    tl.store(lse_ptrs, lse.to(gLSE.dtype.element_ty), mask=mask_m)

    # ---- Pass 2: reload logits, subtract lse, write logsoftmax ----
    # Output was initialized to -inf, so columns beyond seqlen_k are already correct.
    for bok in range(0, seqlen_k, kTileKB):
        offs_k = bok + tl.arange(0, kTileKB)

        out_ptrs = gOut + iHead_i64 * stride_out_h + (bos_q_i64 + offs_m[:, None]) * stride_out_t + offs_k[None, :] * stride_out_k
        load_mask = mask_m[:, None] & (offs_k[None, :] < seqlen_k)
        logits = tl.load(out_ptrs, mask=load_mask, other=float('-inf')).to(tl.float32)

        logsm = logits - lse[:, None]
        # it can happen that logits and lse are both -inf (when the entire row is masked), resulting in nan, we set those to -inf
        # Keep -inf where logits were -inf (masked positions)
        logsm = tl.where(logits > float('-inf'), logsm, float('-inf'))

        tl.store(out_ptrs, logsm.to(gOut.dtype.element_ty), mask=load_mask)


@triton.jit
def _fused_causal_matmul_logsoftmax_bwd_kernel_dq(
    gDQ,            # [total_q_len, nheads, head_dim], output
    gDLogSM,        # [nheads, total_q_len, max_seqlen_kb], grad of logsoftmax
    gLogSM,         # [nheads, total_q_len, max_seqlen_kb], logsoftmax values
    gKB,            # [total_kb_len, nheads, head_dim]
    gDelta,         # [nheads, total_q_len], precomputed sum(dlogsm, dim=-1)
    gCumSeqLensQ,
    gCumSeqLensKB,
    gQPosOffset,    # [bsz]
    sm_scale,
    stride_dq_t, stride_dq_h, stride_dq_d,
    stride_dl_h, stride_dl_t, stride_dl_k,
    stride_l_h, stride_l_t, stride_l_k,
    stride_kb_t, stride_kb_h, stride_kb_d,
    stride_delta_h, stride_delta_t,
    kBlkSize: tl.constexpr,
    kHeadDim: tl.constexpr,
    kTileQ: tl.constexpr,
    kTileKB: tl.constexpr,
):
    iQTile, iHead, iSeq = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    bos_q = tl.load(gCumSeqLensQ + iSeq).to(tl.int32)
    eos_q = tl.load(gCumSeqLensQ + iSeq + 1).to(tl.int32)
    bos_k = tl.load(gCumSeqLensKB + iSeq).to(tl.int32)
    eos_k = tl.load(gCumSeqLensKB + iSeq + 1).to(tl.int32)
    seqlen_q = eos_q - bos_q
    seqlen_k = eos_k - bos_k
    q_offset = tl.load(gQPosOffset + iSeq).to(tl.int32)

    # int64 copies for pointer arithmetic only (prevent int32 offset overflow on
    # gDLogSM / gLogSM [nheads, total_q, max_seqlen_kb]).
    bos_q_i64 = bos_q.to(tl.int64)
    bos_k_i64 = bos_k.to(tl.int64)
    iHead_i64 = iHead.to(tl.int64)

    bot_q = iQTile * kTileQ
    if bot_q >= seqlen_q:
        return

    offs_m = bot_q + tl.arange(0, kTileQ)
    mask_m = offs_m < seqlen_q
    offs_d = tl.arange(0, kHeadDim)

    # Load delta: [kTileQ] from [nheads, total_q_len]
    delta_ptrs = gDelta + iHead_i64 * stride_delta_h + (bos_q_i64 + offs_m) * stride_delta_t
    delta = tl.load(delta_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    dq_acc = tl.zeros([kTileQ, kHeadDim], dtype=tl.float32)

    # logsoftmax backward: ds_ij = (dlogsm_ij - softmax_ij * delta_i) * sm_scale
    # where softmax_ij = exp(logsm_ij), delta_i = sum_j(dlogsm_ij)
    for bok in range(0, seqlen_k, kTileKB):
        offs_k = bok + tl.arange(0, kTileKB)
        mask_k = offs_k < seqlen_k

        # Right-aligned causal mask
        causal_limit = (q_offset + offs_m) // kBlkSize
        causal_mask = offs_k[None, :] < causal_limit[:, None]
        valid_mask = mask_m[:, None] & mask_k[None, :] & causal_mask

        # Load dlogsm and logsm tiles from [nheads, total_q_len, max_seqlen_kb]
        dl_ptrs = gDLogSM + iHead_i64 * stride_dl_h + (bos_q_i64 + offs_m[:, None]) * stride_dl_t + offs_k[None, :] * stride_dl_k
        l_ptrs = gLogSM + iHead_i64 * stride_l_h + (bos_q_i64 + offs_m[:, None]) * stride_l_t + offs_k[None, :] * stride_l_k
        dlogsm = tl.load(dl_ptrs, mask=valid_mask, other=0.0).to(tl.float32)
        logsm = tl.load(l_ptrs, mask=valid_mask, other=float('-inf')).to(tl.float32)

        # ds = (dlogsm - exp(logsm) * delta) * sm_scale
        softmax = tl.exp(logsm)
        ds = (dlogsm - softmax * delta[:, None]) * sm_scale
        ds = tl.where(valid_mask, ds, 0.0)

        # Load KB: [kTileKB, kHeadDim]
        kb_ptrs = gKB + (bos_k_i64 + offs_k[:, None]) * stride_kb_t + iHead_i64 * stride_kb_h + offs_d[None, :] * stride_kb_d
        kb_val = tl.load(kb_ptrs, mask=mask_k[:, None], other=0.0)

        # dq += ds @ kb
        dq_acc += tl.dot(ds.to(kb_val.dtype), kb_val)

    # Store dq
    dq_ptrs = gDQ + (bos_q_i64 + offs_m[:, None]) * stride_dq_t + iHead_i64 * stride_dq_h + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq_acc.to(gDQ.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _fused_causal_matmul_logsoftmax_bwd_kernel_dkb(
    gDKB,           # [total_kb_len, nheads, head_dim], output
    gDLogSM,        # [nheads, total_q_len, max_seqlen_kb]
    gLogSM,         # [nheads, total_q_len, max_seqlen_kb]
    gQ,             # [total_q_len, nheads, head_dim]
    gDelta,         # [nheads, total_q_len]
    gCumSeqLensQ,
    gCumSeqLensKB,
    gQPosOffset,    # [bsz]
    sm_scale,
    stride_dkb_t, stride_dkb_h, stride_dkb_d,
    stride_dl_h, stride_dl_t, stride_dl_k,
    stride_l_h, stride_l_t, stride_l_k,
    stride_q_t, stride_q_h, stride_q_d,
    stride_delta_h, stride_delta_t,
    kBlkSize: tl.constexpr,
    kHeadDim: tl.constexpr,
    kTileQ: tl.constexpr,
    kTileKB: tl.constexpr,
):
    iKTile, iHead, iSeq = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    bos_q = tl.load(gCumSeqLensQ + iSeq).to(tl.int32)
    eos_q = tl.load(gCumSeqLensQ + iSeq + 1).to(tl.int32)
    bos_k = tl.load(gCumSeqLensKB + iSeq).to(tl.int32)
    eos_k = tl.load(gCumSeqLensKB + iSeq + 1).to(tl.int32)
    seqlen_q = eos_q - bos_q
    seqlen_k = eos_k - bos_k
    q_offset = tl.load(gQPosOffset + iSeq).to(tl.int32)

    bot_k = iKTile * kTileKB
    if bot_k >= seqlen_k:
        return

    offs_k = bot_k + tl.arange(0, kTileKB)
    mask_k = offs_k < seqlen_k
    offs_d = tl.arange(0, kHeadDim)

    dkb_acc = tl.zeros([kTileKB, kHeadDim], dtype=tl.float32)

    # For causal: only queries where (q_offset + q_pos) // block_size > kb_id can see this kb tile
    # The minimum q_pos satisfying (q_offset + q_pos) // block_size > bot_k is:
    #   q_pos >= (bot_k + 1) * block_size - q_offset
    # We clamp to 0 and round down to tile boundary
    min_q_pos = (bot_k + 1) * kBlkSize - q_offset
    # Clamp to >= 0
    start_q = tl.maximum(min_q_pos, 0)
    # Round down to tile boundary
    start_q = (start_q // kTileQ) * kTileQ

    # int64 copies for pointer arithmetic only (prevent int32 offset overflow on
    # gDLogSM / gLogSM [nheads, total_q, max_seqlen_kb]).
    bos_q_i64 = bos_q.to(tl.int64)
    bos_k_i64 = bos_k.to(tl.int64)
    iHead_i64 = iHead.to(tl.int64)

    for boq in range(start_q, seqlen_q, kTileQ):
        offs_m = boq + tl.arange(0, kTileQ)
        mask_m = offs_m < seqlen_q

        # Right-aligned causal mask
        causal_limit = (q_offset + offs_m) // kBlkSize
        causal_mask = offs_k[None, :] < causal_limit[:, None]
        valid_mask = mask_m[:, None] & mask_k[None, :] & causal_mask

        # Load dlogsm, logsm from [nheads, total_q_len, max_seqlen_kb]
        dl_ptrs = gDLogSM + iHead_i64 * stride_dl_h + (bos_q_i64 + offs_m[:, None]) * stride_dl_t + offs_k[None, :] * stride_dl_k
        l_ptrs = gLogSM + iHead_i64 * stride_l_h + (bos_q_i64 + offs_m[:, None]) * stride_l_t + offs_k[None, :] * stride_l_k
        dlogsm = tl.load(dl_ptrs, mask=valid_mask, other=0.0).to(tl.float32)
        logsm = tl.load(l_ptrs, mask=valid_mask, other=float('-inf')).to(tl.float32)

        # delta from [nheads, total_q_len]
        delta_ptrs = gDelta + iHead_i64 * stride_delta_h + (bos_q_i64 + offs_m) * stride_delta_t
        delta = tl.load(delta_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        # ds = (dlogsm - exp(logsm) * delta) * sm_scale
        softmax = tl.exp(logsm)
        ds = (dlogsm - softmax * delta[:, None]) * sm_scale
        ds = tl.where(valid_mask, ds, 0.0)

        # Load Q: [kTileQ, kHeadDim]
        q_ptrs = gQ + (bos_q_i64 + offs_m[:, None]) * stride_q_t + iHead_i64 * stride_q_h + offs_d[None, :] * stride_q_d
        q_val = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        # dkb += ds.T @ q
        dkb_acc += tl.dot(tl.trans(ds.to(q_val.dtype)), q_val)

    # Store dkb
    dkb_ptrs = gDKB + (bos_k_i64 + offs_k[:, None]) * stride_dkb_t + iHead_i64 * stride_dkb_h + offs_d[None, :] * stride_dkb_d
    tl.store(dkb_ptrs, dkb_acc.to(gDKB.dtype.element_ty), mask=mask_k[:, None])


class FusedCausalMatmulLogSofmax(torch.autograd.Function):
    """Compute logsoftmax(causal_mask(q@kb.T * sm_scale))

    Args:
        q: [total_q_len, nheads, head_dim]
        kb: [total_kb_len, nheads, head_dim]
        cu_seqlens_q: [batch_size + 1]
        cu_seqlens_kb: [batch_size + 1]
        max_seqlen_q: int
        max_seqlen_kb: int
        block_size: int
        sm_scale: float
        q_position_offset: [batch_size], per-sequence offset for right-aligned causal mask.
            For training (prefill), this should be 0.
            For inference with KV cache, this should be the context length before the new query tokens.
    Returns:
        logsm: [nheads, total_q_len, max_seqlen_kb]
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx,
        q: torch.Tensor,
        kb: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kb: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kb: int,
        block_size: int,
        sm_scale: float,
        q_position_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert cu_seqlens_q.size(0) == cu_seqlens_kb.size(0)  # bsz + 1
        assert q.size(1) == kb.size(1)  # nheads
        assert q.size(2) == kb.size(2)  # head_dim

        q, kb = maybe_contiguous(q), maybe_contiguous(kb)

        total_q_len, nheads, head_dim = q.shape
        total_kb_len = kb.size(0)
        bsz = cu_seqlens_q.size(0) - 1

        if q_position_offset is None:
            q_position_offset = torch.zeros(bsz, dtype=torch.int32, device=q.device)
        else:
            q_position_offset = q_position_offset.to(dtype=torch.int32, device=q.device)

        # Output: [nheads, total_q_len, max_seqlen_kb]
        # TODO: use static shape for better graph compatiability:
        # 1. (nhead, total_seqlen, total_kb_len) tensor, or
        # 2. (nhead, total_seqlen*total_kb_len//2) ragged tensor + (nhead, total_seqlen, 2) index
        logsm = torch.full((nheads, total_q_len, max_seqlen_kb),
                           float('-inf'), device=q.device, dtype=q.dtype)
        # LSE: [nheads, total_q_len]
        lse = torch.empty(nheads, total_q_len, device=q.device, dtype=torch.float32)

        TILE_Q = 64
        TILE_KB = max(16, min(64, triton.next_power_of_2(max_seqlen_kb)))

        grid = (triton.cdiv(max_seqlen_q, TILE_Q), nheads, bsz)

        _fused_causal_matmul_logsoftmax_fwd_kernel[grid](
            gOut=logsm,
            gLSE=lse,
            gQ=q,
            gKB=kb,
            gCumSeqLensQ=cu_seqlens_q,
            gCumSeqLensKB=cu_seqlens_kb,
            gQPosOffset=q_position_offset,
            sm_scale=sm_scale,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_kb_t=kb.stride(0), stride_kb_h=kb.stride(1), stride_kb_d=kb.stride(2),
            stride_out_h=logsm.stride(0), stride_out_t=logsm.stride(1), stride_out_k=logsm.stride(2),
            stride_lse_h=lse.stride(0), stride_lse_t=lse.stride(1),
            kBlkSize=block_size,
            kHeadDim=head_dim,
            kTileQ=TILE_Q,
            kTileKB=TILE_KB,
            num_warps=4,
        )

        ctx.save_for_backward(q, kb, logsm, lse, cu_seqlens_q, cu_seqlens_kb, q_position_offset)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kb = max_seqlen_kb
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale

        return logsm

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dlogsm: torch.Tensor):
        q, kb, logsm, lse, cu_seqlens_q, cu_seqlens_kb, q_position_offset = ctx.saved_tensors

        dlogsm = maybe_contiguous(dlogsm)

        total_q_len, nheads, head_dim = q.shape
        bsz = cu_seqlens_q.size(0) - 1

        # Precompute delta = sum(dlogsm, dim=-1): [nheads, total_q_len]
        delta = dlogsm.sum(dim=-1).float()

        dq = torch.empty_like(q)
        dkb = torch.empty_like(kb)

        TILE_Q = 64
        TILE_KB = max(16, min(64, triton.next_power_of_2(ctx.max_seqlen_kb)))

        grid_dq = (triton.cdiv(ctx.max_seqlen_q, TILE_Q), nheads, bsz)
        _fused_causal_matmul_logsoftmax_bwd_kernel_dq[grid_dq](
            gDQ=dq,
            gDLogSM=dlogsm,
            gLogSM=logsm,
            gKB=kb,
            gDelta=delta,
            gCumSeqLensQ=cu_seqlens_q,
            gCumSeqLensKB=cu_seqlens_kb,
            gQPosOffset=q_position_offset,
            sm_scale=ctx.sm_scale,
            stride_dq_t=dq.stride(0), stride_dq_h=dq.stride(1), stride_dq_d=dq.stride(2),
            stride_dl_h=dlogsm.stride(0), stride_dl_t=dlogsm.stride(1), stride_dl_k=dlogsm.stride(2),
            stride_l_h=logsm.stride(0), stride_l_t=logsm.stride(1), stride_l_k=logsm.stride(2),
            stride_kb_t=kb.stride(0), stride_kb_h=kb.stride(1), stride_kb_d=kb.stride(2),
            stride_delta_h=delta.stride(0), stride_delta_t=delta.stride(1),
            kBlkSize=ctx.block_size,
            kHeadDim=head_dim,
            kTileQ=TILE_Q,
            kTileKB=TILE_KB,
            num_warps=4,
        )

        grid_dkb = (triton.cdiv(ctx.max_seqlen_kb, TILE_KB), nheads, bsz)
        _fused_causal_matmul_logsoftmax_bwd_kernel_dkb[grid_dkb](
            gDKB=dkb,
            gDLogSM=dlogsm,
            gLogSM=logsm,
            gQ=q,
            gDelta=delta,
            gCumSeqLensQ=cu_seqlens_q,
            gCumSeqLensKB=cu_seqlens_kb,
            gQPosOffset=q_position_offset,
            sm_scale=ctx.sm_scale,
            stride_dkb_t=dkb.stride(0), stride_dkb_h=dkb.stride(1), stride_dkb_d=dkb.stride(2),
            stride_dl_h=dlogsm.stride(0), stride_dl_t=dlogsm.stride(1), stride_dl_k=dlogsm.stride(2),
            stride_l_h=logsm.stride(0), stride_l_t=logsm.stride(1), stride_l_k=logsm.stride(2),
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_delta_h=delta.stride(0), stride_delta_t=delta.stride(1),
            kBlkSize=ctx.block_size,
            kHeadDim=head_dim,
            kTileQ=TILE_Q,
            kTileKB=TILE_KB,
            num_warps=4,
        )

        return dq.to(q), dkb.to(kb), None, None, None, None, None, None, None


def fused_causal_matmul_logsoftmax(
    q: torch.Tensor,
    kb: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kb: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kb: int,
    block_size: int,
    sm_scale: float,
    q_position_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional interface for FusedCausalMatmulLogSofmax.

    Computes logsoftmax(causal_mask(q @ kb.T * sm_scale)) for variable-length sequences.
    Supports right-aligned causal masking for inference with KV caching. Note that, self-block
    (i.e., the block that a query belongs to) is masked out (filled with -inf) to avoid info leak.

    Args:
        q: [total_q_len, nheads, head_dim]
        kb: [total_kb_len, nheads, head_dim]
        cu_seqlens_q: [batch_size + 1]
        cu_seqlens_kb: [batch_size + 1]
        max_seqlen_q: int
        max_seqlen_kb: int
        block_size: int
        sm_scale: float
        q_position_offset: Optional[batch_size] int tensor. Per-sequence position offset
            for right-aligned causal masking. The causal limit for query at local position
            q_pos is (q_position_offset + q_pos) // block_size. For training (prefill),
            pass None (defaults to 0). For inference with KV cache, pass the number of
            previously processed tokens per sequence.

    Returns:
        logsm: [nheads, total_q_len, max_seqlen_kb]
    """
    return FusedCausalMatmulLogSofmax.apply(
        q, kb, cu_seqlens_q, cu_seqlens_kb,
        max_seqlen_q, max_seqlen_kb, block_size, sm_scale,
        q_position_offset,
    )



#### reference implementation for correctness check ####
def fused_causal_matmul_logsoftmax_ref(
        q: torch.Tensor,
        kb: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kb: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kb: int,
        block_size: int,
        sm_scale: float,
        q_position_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation: logsoftmax(causal_mask(q @ kb.T * sm_scale)).

    Supports right-aligned causal masking via q_position_offset.

    Args:
        q: [total_q_len, nheads, head_dim]
        kb: [total_kb_len, nheads, head_dim]
        cu_seqlens_q: [batch_size + 1]
        cu_seqlens_kb: [batch_size + 1]
        max_seqlen_q: int
        max_seqlen_kb: int
        block_size: int
        sm_scale: float
        q_position_offset: Optional[batch_size] int tensor. Per-sequence position offset
            for right-aligned causal masking. Defaults to 0 (training/prefill).

    Returns:
        logsm: [nheads, total_q_len, max_seqlen_kb]
    """
    bsz = cu_seqlens_q.size(0) - 1
    nheads = q.size(1)
    device = q.device
    dtype = q.dtype

    # Output: [nheads, total_q_len, max_seqlen_kb]
    total_q_len = q.size(0)
    logsm = torch.full((nheads, total_q_len, max_seqlen_kb), float('-inf'),
                       device=device, dtype=dtype)

    for ib in range(bsz):
        q_start, q_end = cu_seqlens_q[ib].item(), cu_seqlens_q[ib + 1].item()
        k_start, k_end = cu_seqlens_kb[ib].item(), cu_seqlens_kb[ib + 1].item()
        seqlen_q = q_end - q_start
        seqlen_kb = k_end - k_start

        # Per-sequence offset for right-aligned causal mask
        q_off = 0 if q_position_offset is None else q_position_offset[ib].item()

        # [nheads, seqlen_q, seqlen_kb]
        score = q[q_start:q_end].permute(1, 0, 2) @ kb[k_start:k_end].permute(1, 2, 0)

        # Right-aligned causal mask: kb_id >= (q_offset + q_pos) // block_size -> -inf
        q_global_pos = q_off + torch.arange(seqlen_q, device=device)  # [seqlen_q]
        qb_id = (q_global_pos // block_size).unsqueeze(1)  # [seqlen_q, 1]
        kb_id = torch.arange(seqlen_kb, device=device).unsqueeze(0)  # [1, seqlen_kb]
        score = score.masked_fill((kb_id >= qb_id), float('-inf'))

        # logsoftmax
        score_scaled = score * sm_scale
        logsm_batch = torch.log_softmax(score_scaled, dim=-1)
        # Handle nan from all-masked rows
        logsm_batch = torch.nan_to_num(logsm_batch, nan=float('-inf'), neginf=float('-inf'))

        # NOTE: the self-block (kb_id == (q_pos + q_off) // block_size) is masked out by
        # the causal mask above. This is intentional — the self-block is always selected
        # and handled by the dense local attention kernel. Force-selecting it here is not
        # needed; the downstream block_selection code should unconditionally include it in
        # the block_mask regardless of the logsoftmax scores.

        # Output layout: [nheads, total_q_len, max_seqlen_kb]
        logsm[:, q_start:q_end, :seqlen_kb] = logsm_batch.to(dtype)

    return logsm
