"""SparseX Causal Attention — Triton kernel + autograd + reference.

Self-contained implementation of block-sparse causal attention with:
- Variable-length (varlen) packed sequences only
- Separate cu_seqlens_q / cu_seqlens_k for cross-attention / KV-cache
- GQA support (nheads_q can be a multiple of nheads_kv)
- Stride-based addressing (no contiguity requirement except last dim)
- Online softmax with log-sum-exp tracking

"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import reduce
from fla.utils import autocast_custom_bwd, autocast_custom_fwd

from sas.utils import ceildiv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tile_config(device_index: int = 0):
    """Select tile sizes based on GPU architecture."""
    props = torch.cuda.get_device_properties(device_index)
    sm_major = props.major
    if sm_major >= 9:  # Hopper
        return dict(kTileQ=128, kTileKV=64, kSliceQK=256, kSliceV=256, num_warps=8)
    elif sm_major >= 8:  # Ampere / Ada
        return dict(kTileQ=128, kTileKV=32, kSliceQK=256, kSliceV=128, num_warps=4)
    else:
        return dict(kTileQ=64, kTileKV=32, kSliceQK=256, kSliceV=64, num_warps=2)


# ---------------------------------------------------------------------------
# Forward Kernel
# ---------------------------------------------------------------------------

@triton.jit
def _sparsex_fwd_kernel(
    gQ,            # [total_q, nheads_q, head_dim]
    gK,            # [total_k, nheads_kv, head_dim]
    gV,            # [total_k, nheads_kv, head_dim_v]
    gO,            # [total_q, nheads_q, head_dim_v]
    gLSE,          # [total_q, nheads_q]
    gCuSeqLensQ,   # [bsz + 1]
    gCuSeqLensK,   # [bsz + 1]
    gBlockLogSM,   # [total_q, nheads_q, max_num_blocks], float — block gate logsm
    gThreshold,    # [total_q, nheads_q], float — logsm threshold for block pruning
    sm_scale,
    nheads_q,
    grp_heads,
    head_dim_v,
    # strides
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    stride_blsm_t, stride_blsm_h, stride_blsm_k,
    stride_th_t, stride_th_h,
    # constexprs
    kBlkSize: tl.constexpr,
    kHeadDim: tl.constexpr,
    kTileQ: tl.constexpr,
    kTileKV: tl.constexpr,
    kSliceV: tl.constexpr,
):
    iVSlice, iQTile, iBatchHead = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    iSeq = iBatchHead // nheads_q
    iHeadQ = iBatchHead % nheads_q
    iHeadKV = iHeadQ // grp_heads

    # Varlen: load per-sequence boundaries
    bos_q = tl.load(gCuSeqLensQ + iSeq).to(tl.int32)
    eos_q = tl.load(gCuSeqLensQ + iSeq + 1).to(tl.int32)
    bos_k = tl.load(gCuSeqLensK + iSeq).to(tl.int32)
    eos_k = tl.load(gCuSeqLensK + iSeq + 1).to(tl.int32)
    cur_seqlen_q = eos_q - bos_q
    cur_seqlen_k = eos_k - bos_k

    # int64 copies for pointer arithmetic only (avoid int32 offset overflow on
    # large tensors, e.g. block_logsm [nheads_q, total_q, max_blocks] where
    # iHeadQ * stride(0) can exceed INT32_MAX). Arithmetic/control-flow below
    # keeps the int32 versions unchanged.
    bos_q_i64 = bos_q.to(tl.int64)
    bos_k_i64 = bos_k.to(tl.int64)
    iHeadQ_i64 = iHeadQ.to(tl.int64)
    iHeadKV_i64 = iHeadKV.to(tl.int64)

    LOG2E: tl.constexpr = 1.44269504

    # Compute aligned boq in-kernel (no host tile table needed)
    q_k_offset = cur_seqlen_k - cur_seqlen_q
    offset_in_blk = q_k_offset % kBlkSize
    boq = iQTile * kTileQ - offset_in_blk
    eoq = boq + kTileQ
    if eoq <= 0 or boq >= cur_seqlen_q:  # no intersection with [0, cur_seqlen_q)
        return
    self_blk_id = (q_k_offset + boq) // kBlkSize  # sequence-global block id
    boq = max(boq, 0)
    eoq = min(min(eoq, cur_seqlen_q), (self_blk_id + 1) * kBlkSize - q_k_offset)

    # ---- Load Q tile: [kTileQ, kHeadDim] ----
    offs_q = boq + tl.arange(0, kTileQ)
    offs_d = tl.arange(0, kHeadDim)
    mask_q = offs_q[:, None] < eoq
    q_ptrs = gQ + (bos_q_i64 + offs_q[:, None]) * stride_q_t + iHeadQ_i64 * stride_q_h + offs_d[None, :] * stride_q_d
    sQ = tl.load(q_ptrs, mask=mask_q, other=0.0)  # [kTileQ, kHeadDim]

    # ---- Base pointer for on-demand block_logsm loading; load threshold once ----
    blsm_base = gBlockLogSM + bos_q_i64 * stride_blsm_t + iHeadQ_i64 * stride_blsm_h
    th_ptrs = gThreshold + (bos_q_i64 + offs_q) * stride_th_t + iHeadQ_i64 * stride_th_h
    sThreshold = tl.load(th_ptrs, mask=(offs_q < eoq), other=float('inf'))  # [kTileQ]

    # ---- Accumulators ----
    offs_v = iVSlice * kSliceV + tl.arange(0, kSliceV)
    sO = tl.zeros([kTileQ, kSliceV], dtype=tl.float32)
    sRowMax = tl.full([kTileQ], float('-inf'), dtype=tl.float32)
    sSumExp = tl.zeros([kTileQ], dtype=tl.float32)

    tl.static_assert(kBlkSize % kTileKV == 0, "kBlkSize must be divisible by kTileKV")
    tl.static_assert(kBlkSize % kTileQ == 0, "kBlkSize must be divisible by kTileQ")

    # ---- Phase 1: Self-block (latest block, causal masking) ----
    # Map Q positions to K positions for unequal-length support (right-aligned)
    for bok in range(self_blk_id * kBlkSize, q_k_offset + eoq, kTileKV):
        offs_k = bok + tl.arange(0, kTileKV)
        mask_k = offs_k < cur_seqlen_k
        # Load K transposed: [kHeadDim, kTileKV]
        k_ptrs = gK + (bos_k_i64 + offs_k[None, :]) * stride_k_t + iHeadKV_i64 * stride_k_h + offs_d[:, None] * stride_k_d
        sK = tl.load(k_ptrs, mask=mask_k[None, :], other=0.0)
        # Load V: [kTileKV, kSliceV]
        v_ptrs = gV + (bos_k_i64 + offs_k[:, None]) * stride_v_t + iHeadKV_i64 * stride_v_h + offs_v[None, :] * stride_v_d
        sV = tl.load(v_ptrs, mask=mask_k[:, None] & (offs_v[None, :] < head_dim_v), other=0.0)
        # Attention logits
        sP = tl.dot(sQ, sK) * sm_scale * LOG2E  # [kTileQ, kTileKV]
        # Causal mask: q_global >= k_in_q_coords AND k within bounds AND q within tile
        sP = tl.where(
            ((q_k_offset + boq + tl.arange(0, kTileQ)[:, None]) >= offs_k[None, :]) & mask_k[None, :] & (offs_q[:, None] < eoq),
            sP, float('-inf')
        )
        # Online softmax
        sRowMax, sRowMaxPrev = tl.maximum(sRowMax, tl.max(sP, 1)), sRowMax
        sRescale = tl.math.exp2(sRowMaxPrev - sRowMax)
        sP = tl.math.exp2(sP - sRowMax[:, None])
        sSumExp = sSumExp * sRescale + tl.sum(sP, 1)
        sO = sO * sRescale[:, None] + tl.dot(sP.to(sQ.dtype), sV)

    # ---- Phase 2: Previous blocks (no causal mask, block-sparse) ----
    for blk_id in range(self_blk_id - 1, -1, -1):
        # Load block logsm and compute mask in-kernel: active where logsm >= threshold
        blsm_col_ptrs = blsm_base + offs_q * stride_blsm_t + blk_id * stride_blsm_k
        sLogSM = tl.load(blsm_col_ptrs, mask=(offs_q < eoq), other=float('-inf'))
        sIsActive = (sLogSM >= sThreshold)
        # Tile-level early skip
        if tl.sum(sIsActive.to(tl.int32)) > 0:
            # sLogSM already loaded above for mask computation
            # Iterate over KV tiles within this block
            for bok in range(blk_id * kBlkSize, (blk_id + 1) * kBlkSize, kTileKV):
                offs_k = bok + tl.arange(0, kTileKV)
                # Load K transposed
                k_ptrs = gK + (bos_k_i64 + offs_k[None, :]) * stride_k_t + iHeadKV_i64 * stride_k_h + offs_d[:, None] * stride_k_d
                sK = tl.load(k_ptrs)
                # Load V
                v_ptrs = gV + (bos_k_i64 + offs_k[:, None]) * stride_v_t + iHeadKV_i64 * stride_v_h + offs_v[None, :] * stride_v_d
                sV = tl.load(v_ptrs, mask=(offs_v[None, :] < head_dim_v), other=0.0)
                # Attention
                sP = tl.dot(sQ, sK) * sm_scale * LOG2E
                # Add block gate: logsm[q, blk_id] * LOG2E for each query row
                sP = sP + (sLogSM * LOG2E)[:, None]
                sP = tl.where(sIsActive[:, None], sP, float('-inf'))
                # Online softmax
                sRowMax, sRowMaxPrev = tl.maximum(sRowMax, tl.max(sP, 1)), sRowMax
                sRescale = tl.math.exp2(sRowMaxPrev - sRowMax)
                sP = tl.math.exp2(sP - sRowMax[:, None])
                sSumExp = sSumExp * sRescale + tl.sum(sP, 1)
                sO = sO * sRescale[:, None] + tl.dot(sP.to(sQ.dtype), sV)

    # ---- Finalize ----
    sO = sO / sSumExp[:, None]
    sRowMax += tl.math.log2(sSumExp)

    # ---- Store O ----
    o_ptrs = gO + (bos_q_i64 + offs_q[:, None]) * stride_o_t + iHeadQ_i64 * stride_o_h + offs_v[None, :] * stride_o_d
    tl.store(o_ptrs, sO.to(gO.dtype.element_ty),
             mask=(offs_q[:, None] < eoq) & (offs_v[None, :] < head_dim_v))

    # ---- Store LSE (only from first V slice to avoid duplication) ----
    if iVSlice == 0:
        lse_ptrs = gLSE + (bos_q_i64 + offs_q) * stride_lse_t + iHeadQ_i64 * stride_lse_h
        tl.store(lse_ptrs, sRowMax.to(gLSE.dtype.element_ty), mask=offs_q < eoq)


# ---------------------------------------------------------------------------
# Backward Preprocess Kernel: delta = rowsum(O * dO)
# ---------------------------------------------------------------------------

@triton.jit
def _sparsex_bwd_preprocess(
    gO, gDO, gDelta,
    stride_o_t, stride_o_h, stride_o_d,
    stride_do_t, stride_do_h, stride_do_d,
    kHeadDimV: tl.constexpr,
    kHeadDimVPad: tl.constexpr,
):
    """Compute delta[t, h] = sum_d(O[t,h,d] * dO[t,h,d]) for each (token, head)."""
    iPos = tl.program_id(0)  # flat index over (total_q * nheads_q)
    idx = tl.arange(0, kHeadDimVPad)
    mask = idx < kHeadDimV

    sO = tl.load(gO + iPos * kHeadDimV + idx, mask=mask, other=0.0)
    sDO = tl.load(gDO + iPos * kHeadDimV + idx, mask=mask, other=0.0).to(tl.float32)
    sDelta = tl.sum(sO * sDO)
    tl.store(gDelta + iPos, sDelta.to(gDelta.dtype.element_ty))


# ---------------------------------------------------------------------------
# Backward Kernel: dQ (+ dGate)
# ---------------------------------------------------------------------------

@triton.jit
def _sparsex_bwd_dq_kernel(
    gQ, gK, gV, gDO, gDQ, gDG,
    gLSE, gDelta,
    gCuSeqLensQ, gCuSeqLensK,
    gBlockLogSM, gThreshold,
    sm_scale,
    nheads_q,
    grp_heads,
    head_dim_v,
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_do_t, stride_do_h, stride_do_d,
    stride_dq_t, stride_dq_h, stride_dq_d,
    stride_lse_t, stride_lse_h,
    stride_delta_t, stride_delta_h,
    stride_blsm_t, stride_blsm_h, stride_blsm_k,
    stride_th_t, stride_th_h,
    stride_dg_t, stride_dg_h, stride_dg_k,
    kBlkSize: tl.constexpr,
    kHeadDim: tl.constexpr,
    kTileQ: tl.constexpr,
    kTileKV: tl.constexpr,
    kSliceV: tl.constexpr,
):
    iVSlice, iQTile, iBatchHead = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    iSeq = iBatchHead // nheads_q
    iHeadQ = iBatchHead % nheads_q
    iHeadKV = iHeadQ // grp_heads

    bos_q = tl.load(gCuSeqLensQ + iSeq).to(tl.int32)
    eos_q = tl.load(gCuSeqLensQ + iSeq + 1).to(tl.int32)
    bos_k = tl.load(gCuSeqLensK + iSeq).to(tl.int32)
    eos_k = tl.load(gCuSeqLensK + iSeq + 1).to(tl.int32)
    cur_seqlen_q = eos_q - bos_q
    cur_seqlen_k = eos_k - bos_k
    q_k_offset = cur_seqlen_k - cur_seqlen_q

    # int64 copies for pointer arithmetic only (prevent int32 offset overflow).
    bos_q_i64 = bos_q.to(tl.int64)
    bos_k_i64 = bos_k.to(tl.int64)
    iHeadQ_i64 = iHeadQ.to(tl.int64)
    iHeadKV_i64 = iHeadKV.to(tl.int64)

    LOG2E: tl.constexpr = 1.44269504
    # Compute aligned boq in-kernel
    offset_in_blk = q_k_offset % kBlkSize
    boq = iQTile * kTileQ - offset_in_blk
    eoq = boq + kTileQ
    if eoq <= 0 or boq >= cur_seqlen_q:  # no intersection with [0, cur_seqlen_q)
        return
    self_blk_id = (q_k_offset + boq) // kBlkSize  # sequence-global block id
    boq = max(boq, 0)
    eoq = min(min(eoq, cur_seqlen_q), (self_blk_id + 1) * kBlkSize - q_k_offset)

    offs_q = boq + tl.arange(0, kTileQ)
    offs_d = tl.arange(0, kHeadDim)
    offs_v = iVSlice * kSliceV + tl.arange(0, kSliceV)
    mask_q = offs_q < eoq  # use eoq instead of cur_seqlen_q for aligned tiling

    # Load Q
    q_ptrs = gQ + (bos_q_i64 + offs_q[:, None]) * stride_q_t + iHeadQ_i64 * stride_q_h + offs_d[None, :] * stride_q_d
    sQ = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

    # Load dO
    do_ptrs = gDO + (bos_q_i64 + offs_q[:, None]) * stride_do_t + iHeadQ_i64 * stride_do_h + offs_v[None, :] * stride_do_d
    sDO = tl.load(do_ptrs, mask=mask_q[:, None] & (offs_v[None, :] < head_dim_v), other=0.0)

    # Load LSE, Delta
    lse_ptrs = gLSE + (bos_q_i64 + offs_q) * stride_lse_t + iHeadQ_i64 * stride_lse_h
    sLSE = tl.load(lse_ptrs, mask=mask_q, other=0.0)
    delta_ptrs = gDelta + (bos_q_i64 + offs_q) * stride_delta_t + iHeadQ_i64 * stride_delta_h
    sDelta = tl.load(delta_ptrs, mask=mask_q, other=0.0)

    # Base pointer for on-demand block_logsm loading; load threshold once
    blsm_base = gBlockLogSM + bos_q_i64 * stride_blsm_t + iHeadQ_i64 * stride_blsm_h
    th_ptrs = gThreshold + (bos_q_i64 + offs_q) * stride_th_t + iHeadQ_i64 * stride_th_h
    sThresholdQ = tl.load(th_ptrs, mask=mask_q, other=float('inf'))  # [kTileQ]

    sDQ = tl.zeros([kTileQ, kHeadDim], dtype=tl.float32)

    tl.static_assert(kBlkSize % kTileQ == 0, "kBlkSize must be divisible by kTileQ")

    # ---- Phase 1: Self-block (causal) ----
    for bok in range(self_blk_id * kBlkSize, q_k_offset + eoq, kTileKV):
        offs_k = bok + tl.arange(0, kTileKV)
        mask_k = offs_k < cur_seqlen_k
        # Load K transposed
        k_ptrs = gK + (bos_k_i64 + offs_k[None, :]) * stride_k_t + iHeadKV_i64 * stride_k_h + offs_d[:, None] * stride_k_d
        sK = tl.load(k_ptrs, mask=mask_k[None, :], other=0.0)
        # Load V
        v_ptrs = gV + (bos_k_i64 + offs_k[:, None]) * stride_v_t + iHeadKV_i64 * stride_v_h + offs_v[None, :] * stride_v_d
        sV = tl.load(v_ptrs, mask=mask_k[:, None] & (offs_v[None, :] < head_dim_v), other=0.0)
        # Recompute attention (same logic as forward Phase 1, single self-block)
        sP = tl.dot(sQ, sK) * sm_scale * LOG2E
        # Causal mask: q_global >= k AND k within bounds AND q within tile
        sP = tl.where(
            ((q_k_offset + boq + tl.arange(0, kTileQ)[:, None]) >= offs_k[None, :]) & mask_k[None, :] & mask_q[:, None],
            sP, float('-inf')
        )
        sP = tl.math.exp2(sP - sLSE[:, None])
        # dS = P * (dO @ V^T - delta)
        sDP = tl.dot(sDO, tl.trans(sV))
        sDS = sP * (sDP - sDelta[:, None])
        # dQ += dS @ K^T
        sDQ += tl.dot(sDS.to(sK.dtype), tl.trans(sK))

    # ---- Phase 2: Previous blocks ----
    for blk_id in range(self_blk_id - 1, -1, -1):
        # Load block logsm and compute mask in-kernel
        blsm_col_ptrs = blsm_base + offs_q * stride_blsm_t + blk_id * stride_blsm_k
        sLogSM = tl.load(blsm_col_ptrs, mask=mask_q, other=float('-inf'))
        sIsActive = (sLogSM >= sThresholdQ) & mask_q
        if tl.sum(sIsActive.to(tl.int32)) > 0:
            # sLogSM already loaded above for mask computation
            sDGate = tl.zeros([kTileQ], dtype=tl.float32)
            for bok in range(blk_id * kBlkSize, (blk_id + 1) * kBlkSize, kTileKV):
                offs_k = bok + tl.arange(0, kTileKV)
                # Load K transposed
                k_ptrs = gK + (bos_k_i64 + offs_k[None, :]) * stride_k_t + iHeadKV_i64 * stride_k_h + offs_d[:, None] * stride_k_d
                sK = tl.load(k_ptrs)
                v_ptrs = gV + (bos_k_i64 + offs_k[:, None]) * stride_v_t + iHeadKV_i64 * stride_v_h + offs_v[None, :] * stride_v_d
                sV = tl.load(v_ptrs, mask=(offs_v[None, :] < head_dim_v), other=0.0)
                sP = tl.dot(sQ, sK) * sm_scale * LOG2E
                # Add block gate logsm (sLogSM loaded before the KV loop)
                sP = sP + (sLogSM * LOG2E)[:, None]
                sP = tl.where(sIsActive[:, None], sP, float('-inf'))
                sP = tl.math.exp2(sP - sLSE[:, None])
                sDP = tl.dot(sDO, tl.trans(sV))
                sDS = sP * (sDP - sDelta[:, None])
                sDGate += tl.sum(sDS, axis=1)  # dg = sum of dS over kv positions in block
                sDQ += tl.dot(sDS.to(sK.dtype), tl.trans(sK))
            # Store dGate for this block
            dg_ptrs = gDG + (bos_q_i64 + offs_q) * stride_dg_t + iHeadQ_i64 * stride_dg_h + blk_id * stride_dg_k
            tl.store(dg_ptrs, sDGate.to(gDG.dtype.element_ty), mask=sIsActive)

    sDQ *= sm_scale
    # Store dQ
    dq_ptrs = gDQ + (bos_q_i64 + offs_q[:, None]) * stride_dq_t + iHeadQ_i64 * stride_dq_h + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, sDQ.to(gDQ.dtype.element_ty), mask=mask_q[:, None])


# ---------------------------------------------------------------------------
# Backward Kernel: dK, dV
# ---------------------------------------------------------------------------

@triton.jit
def _sparsex_bwd_dkv_kernel(
    gQ, gK, gV, gDO, gDK, gDV,
    gLSE, gDelta,
    gCuSeqLensQ, gCuSeqLensK,
    gBlockLogSM, gThreshold,
    sm_scale,
    nheads_q,
    grp_heads,
    head_dim_v,
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_do_t, stride_do_h, stride_do_d,
    stride_dk_t, stride_dk_h, stride_dk_d,
    stride_dv_t, stride_dv_h, stride_dv_d,
    stride_lse_t, stride_lse_h,
    stride_delta_t, stride_delta_h,
    stride_blsm_t, stride_blsm_h, stride_blsm_k,
    stride_th_t, stride_th_h,
    kBlkSize: tl.constexpr,
    kHeadDim: tl.constexpr,
    kTileQ: tl.constexpr,   # inner tile (Q is inner loop for dKV)
    kTileKV: tl.constexpr,  # outer tile
    kSliceV: tl.constexpr,
):
    iVSlice, iKTile, iBatchHead = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    iSeq = iBatchHead // nheads_q
    iHeadQ = iBatchHead % nheads_q
    iHeadKV = iHeadQ // grp_heads

    bos_q = tl.load(gCuSeqLensQ + iSeq).to(tl.int32)
    eos_q = tl.load(gCuSeqLensQ + iSeq + 1).to(tl.int32)
    bos_k = tl.load(gCuSeqLensK + iSeq).to(tl.int32)
    eos_k = tl.load(gCuSeqLensK + iSeq + 1).to(tl.int32)
    cur_seqlen_q = eos_q - bos_q
    cur_seqlen_k = eos_k - bos_k
    q_k_offset = cur_seqlen_k - cur_seqlen_q

    # int64 copies for pointer arithmetic only (prevent int32 offset overflow).
    bos_q_i64 = bos_q.to(tl.int64)
    bos_k_i64 = bos_k.to(tl.int64)
    iHeadQ_i64 = iHeadQ.to(tl.int64)
    iHeadKV_i64 = iHeadKV.to(tl.int64)

    LOG2E: tl.constexpr = 1.44269504

    # This kernel tiles over K/V (outer loop) and iterates over Q (inner loop)
    bok = iKTile * kTileKV
    if bok >= cur_seqlen_k:
        return
    offs_k = bok + tl.arange(0, kTileKV)
    offs_d = tl.arange(0, kHeadDim)
    offs_v = iVSlice * kSliceV + tl.arange(0, kSliceV)
    mask_k = offs_k < cur_seqlen_k

    # Block ID in K-coordinate space
    blk_id_kv = bok // kBlkSize

    # Load K transposed: [kHeadDim, kTileKV]
    k_ptrs = gK + (bos_k_i64 + offs_k[None, :]) * stride_k_t + iHeadKV_i64 * stride_k_h + offs_d[:, None] * stride_k_d
    sK = tl.load(k_ptrs, mask=mask_k[None, :], other=0.0)

    # Load V: [kTileKV, kSliceV]
    v_ptrs = gV + (bos_k_i64 + offs_k[:, None]) * stride_v_t + iHeadKV_i64 * stride_v_h + offs_v[None, :] * stride_v_d
    sV = tl.load(v_ptrs, mask=mask_k[:, None] & (offs_v[None, :] < head_dim_v), other=0.0)

    sDK = tl.zeros([kTileKV, kHeadDim], dtype=tl.float32)
    sDV = tl.zeros([kTileKV, kSliceV], dtype=tl.float32)

    tl.static_assert(kBlkSize % kTileKV == 0, "kBlkSize must be multiple of kTileKV")
    tl.static_assert(kBlkSize % kTileQ == 0, "kBlkSize must be multiple of kTileQ")

    # ---- Phase 1: Q tiles in same K-block as this K tile (causal) ----
    # This K tile covers [bok, bok + kTileKV) in K-coordinates.
    # In Q-coordinates, that maps to [bok - q_k_offset, bok + kTileKV - q_k_offset).
    # But we must also clamp to the aligned K-block boundary:
    # Only Q positions whose self-block == blk_id_kv should be in Phase 1.
    q_self_start = max(bok - q_k_offset, 0)
    q_self_end = min(cur_seqlen_q, (blk_id_kv + 1) * kBlkSize - q_k_offset)
    for boq in range(q_self_start, q_self_end, kTileQ):
        offs_qr = boq + tl.arange(0, kTileQ)
        mask_qr = offs_qr < q_self_end
        # Load Q
        q_ptrs = gQ + (bos_q_i64 + offs_qr[:, None]) * stride_q_t + iHeadQ_i64 * stride_q_h + offs_d[None, :] * stride_q_d
        sQr = tl.load(q_ptrs, mask=mask_qr[:, None], other=0.0)
        # Load dO
        do_ptrs = gDO + (bos_q_i64 + offs_qr[:, None]) * stride_do_t + iHeadQ_i64 * stride_do_h + offs_v[None, :] * stride_do_d
        sDOr = tl.load(do_ptrs, mask=mask_qr[:, None] & (offs_v[None, :] < head_dim_v), other=0.0)
        # Load LSE, Delta
        lse_ptrs = gLSE + (bos_q_i64 + offs_qr) * stride_lse_t + iHeadQ_i64 * stride_lse_h
        sLSEr = tl.load(lse_ptrs, mask=mask_qr, other=0.0)
        delta_ptrs = gDelta + (bos_q_i64 + offs_qr) * stride_delta_t + iHeadQ_i64 * stride_delta_h
        sDeltar = tl.load(delta_ptrs, mask=mask_qr, other=0.0)

        # Recompute P — causal: (Q_pos + q_k_offset) >= K_pos in K-coordinates
        sP = tl.dot(sQr, sK) * sm_scale * LOG2E
        sP = tl.where(
            ((offs_qr[:, None] + q_k_offset) >= offs_k[None, :]) & mask_qr[:, None] & mask_k[None, :],
            sP, float('-inf')
        )
        sP = tl.math.exp2(sP - sLSEr[:, None])
        # dV += P^T @ dO
        sDV += tl.dot(tl.trans(sP.to(sDOr.dtype)), sDOr)
        # dS = P * (dO @ V^T - delta)
        sDP = tl.dot(sDOr, tl.trans(sV))
        sDS = sP * (sDP - sDeltar[:, None])
        # dK += dS^T @ Q
        sDK += tl.dot(sDS.to(sK.dtype).trans(), sQr)

    # ---- Phase 2: Future Q tiles that select this K block via block mask ----
    # Q tiles beyond the self-block: their self-block in K-coords is > blk_id_kv,
    # so in Q-coords, start from ((blk_id_kv + 1) * kBlkSize - q_k_offset)
    q_start = max((blk_id_kv + 1) * kBlkSize - q_k_offset, 0)
    for boq in range(q_start, cur_seqlen_q, kTileQ):
        offs_qr = boq + tl.arange(0, kTileQ)
        mask_qr = offs_qr < cur_seqlen_q

        # Load block logsm for blk_id_kv and threshold, compute mask in-kernel
        blsm_col_ptrs = gBlockLogSM + (bos_q_i64 + offs_qr) * stride_blsm_t + iHeadQ_i64 * stride_blsm_h + blk_id_kv * stride_blsm_k
        sLogSMr = tl.load(blsm_col_ptrs, mask=mask_qr, other=float('-inf'))
        th_ptrs_r = gThreshold + (bos_q_i64 + offs_qr) * stride_th_t + iHeadQ_i64 * stride_th_h
        sThreshold_r = tl.load(th_ptrs_r, mask=mask_qr, other=float('inf'))
        is_active_vec = (sLogSMr >= sThreshold_r)

        if tl.sum(is_active_vec.to(tl.int32)) > 0:
            # Load Q, dO, LSE, Delta
            q_ptrs = gQ + (bos_q_i64 + offs_qr[:, None]) * stride_q_t + iHeadQ_i64 * stride_q_h + offs_d[None, :] * stride_q_d
            sQr = tl.load(q_ptrs, mask=mask_qr[:, None], other=0.0)
            do_ptrs = gDO + (bos_q_i64 + offs_qr[:, None]) * stride_do_t + iHeadQ_i64 * stride_do_h + offs_v[None, :] * stride_do_d
            sDOr = tl.load(do_ptrs, mask=mask_qr[:, None] & (offs_v[None, :] < head_dim_v), other=0.0)
            lse_ptrs = gLSE + (bos_q_i64 + offs_qr) * stride_lse_t + iHeadQ_i64 * stride_lse_h
            sLSEr = tl.load(lse_ptrs, mask=mask_qr, other=0.0)
            delta_ptrs = gDelta + (bos_q_i64 + offs_qr) * stride_delta_t + iHeadQ_i64 * stride_delta_h
            sDeltar = tl.load(delta_ptrs, mask=mask_qr, other=0.0)

            # Recompute P with block gate (sLogSMr already loaded above)
            sP = tl.dot(sQr, sK) * sm_scale * LOG2E
            sP = sP + (sLogSMr * LOG2E)[:, None]
            # Mask: only active queries attend
            sP = tl.where(
                is_active_vec[:, None] & mask_qr[:, None] & mask_k[None, :],
                sP, float('-inf')
            )
            sP = tl.math.exp2(sP - sLSEr[:, None])
            sDV += tl.dot(tl.trans(sP.to(sDOr.dtype)), sDOr)
            sDP = tl.dot(sDOr, tl.trans(sV))
            sDS = sP * (sDP - sDeltar[:, None])
            sDK += tl.dot(sDS.to(sK.dtype).trans(), sQr)

    sDK *= sm_scale

    # Store dK, dV — stored at expanded (nheads_q) positions for GQA reduce later
    dk_ptrs = gDK + (bos_k_i64 + offs_k[:, None]) * stride_dk_t + iHeadQ_i64 * stride_dk_h + offs_d[None, :] * stride_dk_d
    tl.store(dk_ptrs, sDK.to(gDK.dtype.element_ty), mask=mask_k[:, None])
    dv_ptrs = gDV + (bos_k_i64 + offs_k[:, None]) * stride_dv_t + iHeadQ_i64 * stride_dv_h + offs_v[None, :] * stride_dv_d
    tl.store(dv_ptrs, sDV.to(gDV.dtype.element_ty), mask=mask_k[:, None] & (offs_v[None, :] < head_dim_v))


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class SparseXCausalAttentionFunction(torch.autograd.Function):
    """Block-sparse causal attention with differentiable soft block gating.

    Each query attends to its own (causal) block plus the previous blocks whose
    gate log-softmax passes a per-query threshold. The gate value is added to the
    logits, so gradients flow back into ``block_logsm`` and the router is
    differentiable.

    Position encoding is applied by the caller — pass Q/K already rotated.
    """
    @staticmethod
    @autocast_custom_fwd
    def forward(ctx,
        q:torch.Tensor, k:torch.Tensor, v:torch.Tensor,
        cu_seqlens_q:torch.LongTensor, cu_seqlens_k:torch.LongTensor,
        max_seqlen_q:int, max_seqlen_k:int, sm_scale:float,
        block_logsm:torch.Tensor, logsm_threshold:torch.Tensor, block_size:int, # block masking inputs
    ):
        """
        Args:
            q (torch.Tensor): [total_seqlen_q, nheads, head_dim]
            k (torch.Tensor): [total_seqlen_k, nheads, head_dim]
            v (torch.Tensor): [total_seqlen_k, nheads, head_dim]
            cu_seqlens_q (torch.LongTensor): [bsz + 1]
            cu_seqlens_k (torch.LongTensor): [bsz + 1], can be different from cu_seqlens_q for
                cross attention or inference with kv cache
            max_seqlen_q (int): scalar maximum sequence length for queries
            max_seqlen_k (int): scalar maximum sequence length for keys/values
            sm_scale (float): scalar softmax scaling factor
            block_logsm (torch.Tensor): [nheads, total_seqlen_q, *] pre-computed log softmax for each block
            logsm_threshold (torch.Tensor): [nheads, total_seqlen_q] log softmax threshold for block pruning
            block_size (int): block size for block-sparse attention

        Returns:
            (attn_out, lse) (torch.Tensor, torch.Tensor):
                - attn_out: [total_seqlen_q, nheads, head_dim], attention output
                - lse: [nheads, total_seqlen_q], log-sum-exp for each query and head
        """
        # block_logsm: [nheads, total_q, max_num_blocks] — use directly, no permute needed
        # logsm_threshold: [nheads, total_q] — use directly, no permute needed
        # Kernel uses stride-based addressing, so we just pass the correct strides.

        total_q, nheads_q, head_dim = q.shape
        nheads_kv = k.shape[1]
        head_dim_v = v.shape[2]
        grp_heads = nheads_q // nheads_kv

        # Output
        o = torch.empty(total_q, nheads_q, head_dim_v, dtype=v.dtype, device=q.device)
        lse = torch.empty(total_q, nheads_q, dtype=torch.float32, device=q.device)

        # Tile config
        cfg = _get_tile_config(q.device.index)
        kTileQ = min(block_size, cfg['kTileQ'])
        assert block_size % kTileQ == 0, f"block_size ({block_size}) must be divisible by kTileQ ({kTileQ})"
        kTileKV = min(block_size, cfg['kTileKV'], max(16, triton.next_power_of_2(max_seqlen_k)))
        assert block_size % kTileKV == 0, f"block_size ({block_size}) must be divisible by kTileKV ({kTileKV})"
        kSliceV = min(cfg['kSliceV'], max(16, triton.next_power_of_2(head_dim_v)))
        num_warps = cfg['num_warps']

        # Tile dispatch — grid over-allocated based on max_seqlen, early-exit in kernel
        bsz = cu_seqlens_q.size(0) - 1
        max_tiles_q = triton.cdiv(max_seqlen_q + block_size, kTileQ)  # +block_size for offset_in_blk
        num_slices_v = triton.cdiv(head_dim_v, kSliceV)

        # block_logsm is [nheads, total_q, *]: stride(0)=head, stride(1)=token, stride(2)=block
        # kernel expects: stride_blsm_t=token, stride_blsm_h=head, stride_blsm_k=block
        grid = (num_slices_v, max_tiles_q, nheads_q * bsz)
        _sparsex_fwd_kernel[grid](
            gQ=q, gK=k, gV=v, gO=o, gLSE=lse,
            gCuSeqLensQ=cu_seqlens_q, gCuSeqLensK=cu_seqlens_k,
            gBlockLogSM=block_logsm, gThreshold=logsm_threshold,
            sm_scale=sm_scale,
            nheads_q=nheads_q,
            grp_heads=grp_heads,
            head_dim_v=head_dim_v,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_t=k.stride(0), stride_k_h=k.stride(1), stride_k_d=k.stride(2),
            stride_v_t=v.stride(0), stride_v_h=v.stride(1), stride_v_d=v.stride(2),
            stride_o_t=o.stride(0), stride_o_h=o.stride(1), stride_o_d=o.stride(2),
            stride_lse_t=lse.stride(0), stride_lse_h=lse.stride(1),
            stride_blsm_t=block_logsm.stride(1), stride_blsm_h=block_logsm.stride(0), stride_blsm_k=block_logsm.stride(2),
            stride_th_t=logsm_threshold.stride(1), stride_th_h=logsm_threshold.stride(0),
            kBlkSize=block_size,
            kHeadDim=head_dim,
            kTileQ=kTileQ,
            kTileKV=kTileKV,
            kSliceV=kSliceV,
            num_warps=num_warps,
        )

        ctx.save_for_backward(q, k, v, o, lse, logsm_threshold, block_logsm,
                              cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.block_size = block_size
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        # lse: [total_q, nheads] -> [nheads, total_q]
        return o.to(q.dtype), lse.permute(1, 0).contiguous()

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, dlse=None):
        q, k, v, o, lse, logsm_threshold, block_logsm, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        # Ensure do is contiguous — autograd may pass expanded/broadcasted tensors
        do = do.contiguous()

        block_size = ctx.block_size
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale

        total_q, nheads_q, head_dim = q.shape
        total_k = k.shape[0]
        nheads_kv = k.shape[1]
        head_dim_v = v.shape[2]
        grp_heads = nheads_q // nheads_kv

        # block_logsm: [nheads, total_q, max_num_blocks], logsm_threshold: [nheads, total_q]
        # Use directly with swapped strides (no permute needed).

        # Tile config
        cfg = _get_tile_config(q.device.index)
        kTileQ = min(block_size, cfg['kTileQ'])
        assert block_size % kTileQ == 0, f"block_size ({block_size}) must be divisible by kTileQ ({kTileQ})"
        kTileKV = min(block_size, cfg['kTileKV'])
        assert block_size % kTileKV == 0, f"block_size ({block_size}) must be divisible by kTileKV ({kTileKV})"
        kSliceV = min(cfg['kSliceV'], max(16, triton.next_power_of_2(head_dim_v)))
        num_warps = cfg['num_warps']

        # Preprocess: delta = rowsum(O * dO)
        delta = torch.empty(total_q, nheads_q, dtype=torch.float32, device=q.device)
        _sparsex_bwd_preprocess[(total_q * nheads_q,)](
            gO=o, gDO=do, gDelta=delta,
            stride_o_t=o.stride(0), stride_o_h=o.stride(1), stride_o_d=o.stride(2),
            stride_do_t=do.stride(0), stride_do_h=do.stride(1), stride_do_d=do.stride(2),
            kHeadDimV=head_dim_v,
            kHeadDimVPad=triton.next_power_of_2(head_dim_v),
        )

        # Allocate gradients
        dq = torch.empty(total_q, nheads_q, head_dim, dtype=q.dtype, device=q.device)
        # dk/dv expanded to nheads_q for GQA, will be reduced after
        dk_dtype = k.dtype if nheads_kv == nheads_q else torch.float32
        dv_dtype = v.dtype if nheads_kv == nheads_q else torch.float32
        dk = torch.empty(total_k, nheads_q, head_dim, dtype=dk_dtype, device=q.device)
        dv = torch.empty(total_k, nheads_q, head_dim_v, dtype=dv_dtype, device=q.device)
        dg = torch.zeros(total_q, nheads_q, block_logsm.shape[-1], dtype=block_logsm.dtype, device=q.device)

        num_slices_v = triton.cdiv(head_dim_v, kSliceV)
        bsz = cu_seqlens_q.size(0) - 1

        # ---- dQ kernel ----
        max_tiles_q = triton.cdiv(max_seqlen_q + block_size, kTileQ)  # +block_size for offset_in_blk

        grid_dq = (num_slices_v, max_tiles_q, nheads_q * bsz)
        _sparsex_bwd_dq_kernel[grid_dq](
            gQ=q, gK=k, gV=v, gDO=do, gDQ=dq, gDG=dg,
            gLSE=lse, gDelta=delta,
            gCuSeqLensQ=cu_seqlens_q, gCuSeqLensK=cu_seqlens_k,
            gBlockLogSM=block_logsm, gThreshold=logsm_threshold,
            sm_scale=sm_scale,
            nheads_q=nheads_q,
            grp_heads=grp_heads,
            head_dim_v=head_dim_v,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_t=k.stride(0), stride_k_h=k.stride(1), stride_k_d=k.stride(2),
            stride_v_t=v.stride(0), stride_v_h=v.stride(1), stride_v_d=v.stride(2),
            stride_do_t=do.stride(0), stride_do_h=do.stride(1), stride_do_d=do.stride(2),
            stride_dq_t=dq.stride(0), stride_dq_h=dq.stride(1), stride_dq_d=dq.stride(2),
            stride_lse_t=lse.stride(0), stride_lse_h=lse.stride(1),
            stride_delta_t=delta.stride(0), stride_delta_h=delta.stride(1),
            stride_blsm_t=block_logsm.stride(1), stride_blsm_h=block_logsm.stride(0), stride_blsm_k=block_logsm.stride(2),
            stride_th_t=logsm_threshold.stride(1), stride_th_h=logsm_threshold.stride(0),
            stride_dg_t=dg.stride(0), stride_dg_h=dg.stride(1), stride_dg_k=dg.stride(2),
            kBlkSize=block_size,
            kHeadDim=head_dim,
            kTileQ=kTileQ,
            kTileKV=kTileKV,
            kSliceV=kSliceV,
            num_warps=num_warps,
        )

        # ---- dKV kernel ----
        # dKV: outer loop over K tiles, inner loop over Q tiles
        # Note: dKV swaps roles — KV is outer (large tile), Q is inner (small tile)
        kTileKV_dkv = min(block_size, cfg['kTileQ'])  # outer KV tile uses the large cfg tile
        assert block_size % kTileKV_dkv == 0, f"block_size ({block_size}) must be divisible by kTileKV_dkv ({kTileKV_dkv})"
        kTileQ_dkv = min(block_size, cfg['kTileKV'])  # inner Q tile uses the small cfg tile
        assert block_size % kTileQ_dkv == 0, f"block_size ({block_size}) must be divisible by kTileQ_dkv ({kTileQ_dkv})"
        max_tiles_k = triton.cdiv(max_seqlen_k, kTileKV_dkv)

        grid_dkv = (num_slices_v, max_tiles_k, nheads_q * bsz)
        _sparsex_bwd_dkv_kernel[grid_dkv](
            gQ=q, gK=k, gV=v, gDO=do, gDK=dk, gDV=dv,
            gLSE=lse, gDelta=delta,
            gCuSeqLensQ=cu_seqlens_q, gCuSeqLensK=cu_seqlens_k,
            gBlockLogSM=block_logsm, gThreshold=logsm_threshold,
            sm_scale=sm_scale,
            nheads_q=nheads_q,
            grp_heads=grp_heads,
            head_dim_v=head_dim_v,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_t=k.stride(0), stride_k_h=k.stride(1), stride_k_d=k.stride(2),
            stride_v_t=v.stride(0), stride_v_h=v.stride(1), stride_v_d=v.stride(2),
            stride_do_t=do.stride(0), stride_do_h=do.stride(1), stride_do_d=do.stride(2),
            stride_dk_t=dk.stride(0), stride_dk_h=dk.stride(1), stride_dk_d=dk.stride(2),
            stride_dv_t=dv.stride(0), stride_dv_h=dv.stride(1), stride_dv_d=dv.stride(2),
            stride_lse_t=lse.stride(0), stride_lse_h=lse.stride(1),
            stride_delta_t=delta.stride(0), stride_delta_h=delta.stride(1),
            stride_blsm_t=block_logsm.stride(1), stride_blsm_h=block_logsm.stride(0), stride_blsm_k=block_logsm.stride(2),
            stride_th_t=logsm_threshold.stride(1), stride_th_h=logsm_threshold.stride(0),
            kBlkSize=block_size,
            kHeadDim=head_dim,
            kTileQ=kTileQ_dkv,     # Q is inner loop for dKV
            kTileKV=kTileKV_dkv,   # KV is outer loop for dKV
            kSliceV=kSliceV,
            num_warps=num_warps,
        )

        # GQA: reduce dk, dv over head groups
        if nheads_kv != nheads_q:
            dk = reduce(dk, 't (h g) d -> t h d', g=grp_heads, reduction='sum')
            dv = reduce(dv, 't (h g) d -> t h d', g=grp_heads, reduction='sum')

        # dg: [total_q, nheads, *] -> [nheads, total_q, *] to match block_logsm layout
        dg_hn = dg.permute(1, 0, 2).contiguous()
        # return order matches forward args: q, k, v, cu_q, cu_k, max_q, max_k, scale, block_logsm, logsm_threshold, blk_size
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None, dg_hn.to(block_logsm), None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sparsex_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    block_logsm: torch.Tensor,
    logsm_threshold: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse causal attention with differentiable soft block gating.

    Variable-length packed format only.  Position encoding is the caller's
    responsibility — pass Q/K already rotated.

    Args:
        q: [total_q, nheads_q, head_dim]
        k: [total_k, nheads_kv, head_dim]
        v: [total_k, nheads_kv, head_dim_v]
        cu_seqlens_q: [bsz + 1] cumulative Q sequence lengths
        cu_seqlens_k: [bsz + 1] cumulative K/V sequence lengths (can differ from Q)
        max_seqlen_q: maximum Q sequence length
        max_seqlen_k: maximum K/V sequence length
        block_size: block size for sparse attention
        block_logsm: [nheads_q, total_q, max_num_blocks] pre-computed log softmax for each block
        logsm_threshold: [nheads_q, total_q] log softmax threshold for block pruning
        sm_scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        (output, lse):
            - output: [total_q, nheads_q, head_dim_v]
            - lse: [nheads_q, total_q] log-sum-exp
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3, \
        "q/k/v must be [total_seqlen, nheads, head_dim] (varlen packed format)"

    return SparseXCausalAttentionFunction.apply(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        sm_scale,
        block_logsm, logsm_threshold,
        block_size,
    )


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def sparsex_attn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    block_logsm: torch.Tensor,
    logsm_threshold: torch.Tensor,
    sm_scale: Optional[float] = None,
    mode='sparse',
) -> torch.Tensor:
    """Pure-PyTorch reference implementation for sparsex_attn.

    Same interface as ``sparsex_attn``.  Uses full attention with a per-query
    token mask (no gather), looping over each query token for clarity.
    K is un-rotated; rotation is applied internally (matching the kernel).

    Args:
        q: [total_q, nheads_q, head_dim]
        k: [total_k, nheads_kv, head_dim]  (un-rotated when using RoPE)
        v: [total_k, nheads_kv, head_dim_v]
        cu_seqlens_q: [bsz + 1] cumulative Q sequence lengths
        cu_seqlens_k: [bsz + 1] cumulative K/V sequence lengths (can differ from Q)
        max_seqlen_q: maximum Q sequence length
        max_seqlen_k: maximum K/V sequence length
        block_size: block size for sparse attention
        block_logsm: [nheads_q, total_q, max_num_blocks] pre-computed log softmax
        logsm_threshold: [nheads_q, total_q] log softmax threshold for block pruning
        sm_scale: attention scale factor (default: 1/sqrt(head_dim))
        mode (str): 'dense' or 'sparse'. For dense mode, soft gate (exp(block_logsm)) is applied;
            for sparse mode, hard mask (exp(mask(block_logsm))) is applied.

    Returns:
        output: [total_q, nheads_q, head_dim_v]
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    total_q, nheads_q, head_dim = q.shape
    nheads_kv = k.shape[1]
    head_dim_v = v.shape[2]

    # Derive block mask from threshold: [nheads, total_q, max_num_blocks]
    block_mask = block_logsm >= logsm_threshold.unsqueeze(-1)

    # GQA expansion
    if nheads_q != nheads_kv:
        assert nheads_q % nheads_kv == 0
        g = nheads_q // nheads_kv
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    seqlens_q = torch.diff(cu_seqlens_q)
    seqlens_k = torch.diff(cu_seqlens_k)
    bsz = seqlens_q.shape[0]

    output = torch.zeros(total_q, nheads_q, head_dim_v, dtype=q.dtype, device=q.device)

    for i in range(bsz):
        sq_start = cu_seqlens_q[i].item()
        sk_start = cu_seqlens_k[i].item()
        cur_seqlen_q = seqlens_q[i].item()
        cur_seqlen_k = seqlens_k[i].item()
        q_k_offset = cur_seqlen_k - cur_seqlen_q

        cur_num_blocks = ceildiv(cur_seqlen_k, block_size)
        cur_logsm = block_logsm[:, sq_start:sq_start + cur_seqlen_q, :cur_num_blocks]  # [nheads, seqlen_q, cur_num_blocks]
        cur_mask = cur_logsm >= logsm_threshold[:, sq_start:sq_start + cur_seqlen_q].unsqueeze(-1) # [nheads, seqlen_q, cur_num_blocks]
        
        # repeat and trim block_mask and block_logsm to [nheads, seqlen_q, seqlen_k], fill the self-block with causal
        cur_logsm = cur_logsm.repeat_interleave(block_size, dim=-1)[:, :, :cur_seqlen_k]  # [nheads, seqlen_q, seqlen_k]
        cur_mask = cur_mask.repeat_interleave(block_size, dim=-1)[:, :, :cur_seqlen_k]  # [nheads, seqlen_q, seqlen_k]
        # Up to present, for the self-block, logsm is still -inf, and the corresponding mask is False, i.e., fully masked out
        # Next, fill casual mask at the self-block, and fill the cur_logsm at visible positions within the self-block with 0 (means gate 1.0), 
        self_block_mask = (torch.arange(q_k_offset, q_k_offset+cur_seqlen_q, device=q.device) // block_size)[:, None] \
            == (torch.arange(cur_seqlen_k, device=q.device) // block_size)[None, :] # [seqlen_q, seqlen_k]
        causal_mask = torch.arange(q_k_offset, q_k_offset+cur_seqlen_q, device=q.device)[:, None] \
            >= torch.arange(cur_seqlen_k, device=q.device)[None, :]  # [seqlen_q, seqlen_k]
        self_block_mask = self_block_mask & causal_mask  # only keep the diagonal and lower triangle in the self-block
        cur_logsm.masked_fill_(self_block_mask, 0.0)  # fill self-block with 0.0 logsm (1.0 gate)
        cur_mask = cur_mask | self_block_mask.unsqueeze(0)  # [nheads, seqlen_q, seqlen_k], True means attend, False means masked out

        if mode == 'sparse':
            neginf_mask = torch.where(cur_mask, 0.0, float('-inf'))  # [nheads, seqlen_q, seqlen_k], True -> 0.0 -> gate 1.0, False -> -inf -> gate 0.0
        else: # dense mode with soft gate
            neginf_mask = torch.zeros_like(cur_logsm) # no -inf masking, all blocks are attended to with different gates according to block_logsm

        # slice current sequence and permute to [nheads, seqlen, head_dim] for easier indexing
        cur_q = q[sq_start:sq_start + cur_seqlen_q].permute(1, 0, 2)  # [nheads, seqlen_q, head_dim]
        cur_k = k[sk_start:sk_start + cur_seqlen_k].permute(1, 0, 2)  # [nheads, seqlen_k, head_dim]
        cur_v = v[sk_start:sk_start + cur_seqlen_k].permute(1, 0, 2)  # [nheads, seqlen_k, head_dim_v]
        # repeat for per-query masking and RoPE with contiguous positions
        cur_q = cur_q.view(nheads_q, cur_seqlen_q, 1, head_dim)  # [nheads, seqlen_q, 1, head_dim]
        cur_k = cur_k.view(nheads_q, 1, cur_seqlen_k, head_dim).expand(
            -1, cur_seqlen_q, -1, -1)  # [nheads, seqlen_q, seqlen_k, head_dim]
        cur_v = cur_v.view(nheads_q, 1, cur_seqlen_k, head_dim_v).expand(
            -1, cur_seqlen_q, -1, -1)  # [nheads, seqlen_q, seqlen_k, head_dim_v]

        # QK matmul in original dtype (matches kernel's fp16 TensorCore behavior),
        # then cast to fp32 for gate addition + softmax (matches kernel's fp32 accumulation).
        attn_logits = cur_q.to(q.dtype) @ cur_k.to(q.dtype).transpose(-2, -1) * sm_scale
        attn_logits = attn_logits.float().squeeze(-2) + cur_logsm.float() + neginf_mask.float()
        attn_weights = F.softmax(attn_logits, dim=-1).to(cur_v.dtype).view(
            nheads_q, cur_seqlen_q, 1, cur_seqlen_k)  # [nheads, seqlen_q, 1, seqlen_k]
        o = attn_weights @ cur_v  # [nheads, seqlen_q, 1, seqlen_k]@[nheads, seqlen_q, seqlen_k, head_dim_v]->[nheads, seqlen_q, 1, head_dim_v]
        o = o.squeeze(-2).permute(1, 0, 2) # [seqlen_q, nheads_q, head_dim_v]
        output[sq_start:sq_start + cur_seqlen_q] = o

    return output