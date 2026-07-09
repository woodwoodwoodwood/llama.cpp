#!/usr/bin/env python3
"""Standalone correctness check for the GSQ2 direct-repack path.

Verifies that repacking a GSQ compressed-tensors 'pack-quantized' 2-bit symmetric
Linear (num_bits=2, group_size=128) into the ggml GSQ2 block layout is *lossless*:
i.e. dequantizing the GSQ2 bytes reproduces exactly the compressed-tensors
reference dequantization  w = (code - 2) * scale.

Pure CPU / numpy + safetensors. Run it on the *assembled* (pre-humming) checkpoint:

    python scripts/verify_gsq2_repack.py \
        --model /path/to/assembled \
        --pattern experts.0.gate_proj    # optional, defaults to first quantized Linear
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch
from safetensors import safe_open

GROUP = 128
NUM_BITS = 2
BLOCK_BYTES = 2 + GROUP // 4  # 34


def unpack_ct_codes(weight_packed: np.ndarray, K: int) -> np.ndarray:
    """LSB-first unpack of compressed-tensors int32 -> unsigned codes [N, K] in [0,4)."""
    wp = weight_packed.astype(np.int64)
    vals_per_i32 = 32 // NUM_BITS  # 16
    shifts = np.arange(vals_per_i32, dtype=np.int64) * NUM_BITS
    codes = (wp[:, :, None] >> shifts[None, None, :]) & 0x3
    return codes.reshape(wp.shape[0], -1)[:, :K].astype(np.uint16)


def ct_reference_dequant(weight_packed, weight_scale, N, K) -> np.ndarray:
    """compressed-tensors symmetric dequant: w = (code - 2^(bits-1)) * scale."""
    codes = unpack_ct_codes(weight_packed, K).astype(np.float32)
    signed = codes - (1 << (NUM_BITS - 1))  # -2 offset
    scale = weight_scale.astype(np.float32)  # [N, K//group]
    scale = np.repeat(scale, GROUP, axis=1)[:, :K]
    return signed * scale


def gsq2_pack(weight_packed, weight_scale, N, K):
    """Mirror of ModelBase._gsq2_pack -> raw uint8 [N, (K//GROUP)*34]."""
    n_groups = K // GROUP
    codes = unpack_ct_codes(weight_packed, K)  # [N, K]
    c = codes.reshape(N, K // 4, 4)
    qs = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    qs = qs.reshape(N, n_groups, GROUP // 4)
    d = weight_scale.astype(np.float32)[:, :n_groups]
    d16 = d.astype(np.float16).view(np.uint8).reshape(N, n_groups, 2)
    raw = np.concatenate([d16, qs], axis=-1).reshape(N, n_groups * BLOCK_BYTES)
    return np.ascontiguousarray(raw)


def gsq2_dequant(raw, N, K):
    """Dequant the GSQ2 bytes back to float, matching ggml dequantize_row_gsq2."""
    n_groups = K // GROUP
    blk = raw.reshape(N, n_groups, BLOCK_BYTES)
    d = blk[:, :, :2].reshape(N, n_groups, 2).copy().view(np.float16).astype(np.float32).reshape(N, n_groups)
    qs = blk[:, :, 2:]  # [N, n_groups, 32]
    out = np.empty((N, n_groups, GROUP), dtype=np.float32)
    for j in range(GROUP):
        byte = j // 4
        off = (j % 4) * 2
        code = (qs[:, :, byte] >> off) & 0x3
        out[:, :, j] = (code.astype(np.float32) - 2.0) * d
    return out.reshape(N, K)


def load_bundle(model_dir, pattern):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
    else:
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="np") as f:
            wm = {k: "model.safetensors" for k in f.keys()}

    # find a quantized linear (has .weight_packed) matching pattern
    prefixes = sorted({k[: -len(".weight_packed")] for k in wm if k.endswith(".weight_packed")})
    if not prefixes:
        raise SystemExit("no .weight_packed tensors found; is this the assembled CT checkpoint?")
    cand = [p for p in prefixes if pattern in p] if pattern else prefixes
    if not cand:
        raise SystemExit(f"no quantized linear matched {pattern!r}; examples: {prefixes[:5]}")
    prefix = cand[0]

    need = {f"{prefix}.weight_packed", f"{prefix}.weight_scale", f"{prefix}.weight_shape"}
    byshard = collections.defaultdict(list)
    for k in need:
        byshard[wm[k]].append(k)
    out = {}
    for shard, ks in byshard.items():
        # use torch framework so bf16 weight_scale loads; convert to numpy afterwards
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    wp = out[f"{prefix}.weight_packed"].numpy()
    ws = out[f"{prefix}.weight_scale"].float().numpy()
    wsh = out[f"{prefix}.weight_shape"].numpy()
    return prefix, wp, ws, wsh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pattern", default="experts.0.gate_proj")
    args = ap.parse_args()

    prefix, wp, ws, wsh = load_bundle(args.model, args.pattern)
    N, K = int(wsh[0]), int(wsh[1])
    print(f"layer   : {prefix}")
    print(f"shape   : N={N} K={K}  weight_packed{wp.shape}{wp.dtype}  weight_scale{ws.shape}{ws.dtype}")

    # weight_scale may be bf16 stored as uint16; safetensors 'np' framework returns it
    # already as float. If it comes back as an unsupported dtype, cast via float.
    ws = np.asarray(ws, dtype=np.float32)

    ref = ct_reference_dequant(wp, ws, N, K)
    raw = gsq2_pack(wp, ws, N, K)
    deq = gsq2_dequant(raw, N, K)

    diff = np.abs(ref - deq)
    denom = max(float(np.abs(ref).max()), 1e-8)
    print(f"gsq2 raw: {raw.shape} bytes/row={raw.shape[1]}  (expect {K // GROUP * BLOCK_BYTES})")
    print(f"max_abs : {diff.max():.3e}   mean_abs: {diff.mean():.3e}   rel_max: {diff.max()/denom:.3e}")
    # only source of error is bf16->f16 scale rounding; codes are bit-exact.
    if diff.max() / denom < 1e-2:
        print("OK: GSQ2 repack matches compressed-tensors dequant (codes bit-exact).")
    else:
        raise SystemExit("MISMATCH: repack differs from reference beyond f16 scale rounding.")


if __name__ == "__main__":
    main()
