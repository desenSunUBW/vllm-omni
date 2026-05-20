# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark HunyuanImage-3.0-style MixFusion with a truncated DiT.

This benchmark does not load the 80B checkpoint.  It builds a small stack of
randomly initialized Hunyuan-shaped DiT blocks and compares three execution
paths for mixed-resolution image token sequences:

* independent: run each request separately
* padding: pad to the largest sequence length and run a dense batch
* mixfusion: split image tokens into GCD chunks, run token-wise ops on chunks,
  recover full sequences only around attention, then scatter back to chunks

The default dimensions follow HunyuanImage-3.0's DiT shape closely enough to
measure kernel/utilization trends.  Use smaller dimensions for quick smoke
tests on limited GPUs or CPU.

Example:
    python benchmarks/diffusion/hunyuan_image3_mixfusion_4layer_bench.py \
        --image-sizes 1024x1024,512x512 --layers 4 --iters 20

Smoke test:
    python benchmarks/diffusion/hunyuan_image3_mixfusion_4layer_bench.py \
        --image-sizes 256x256,128x128 --hidden-size 512 \
        --intermediate-size 1408 --heads 8 --layers 2 --iters 3 --warmup 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RequestLayout:
    seq_len: int
    chunk_start: int
    chunk_count: int


@dataclass(frozen=True)
class MixFusionPlan:
    chunk_size: int
    layouts: tuple[RequestLayout, ...]

    @property
    def total_chunks(self) -> int:
        return sum(layout.chunk_count for layout in self.layouts)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class HunyuanLikeBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, heads: int) -> None:
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by heads={heads}.")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.norm1 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = RMSNorm(hidden_size)
        self.gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.heads, self.head_dim)
        k = k.view(bsz, seq_len, self.heads, self.head_dim)
        v = v.view(bsz, seq_len, self.heads, self.head_dim)
        return q, k, v

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).contiguous().view(q.shape[0], -1, self.hidden_size)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        q, k, v = self._project_qkv(self.norm1(x))
        x = residual + self.out_proj(self._attention(q, k, v))
        x = x + self._mlp(self.norm2(x))
        return x

    def forward_mixfusion(self, chunks: torch.Tensor, plan: MixFusionPlan) -> torch.Tensor:
        residual = chunks
        q, k, v = self._project_qkv(self.norm1(chunks))

        attn_chunks = torch.empty_like(chunks)
        for layout in plan.layouts:
            start = layout.chunk_start
            stop = start + layout.chunk_count
            q_full = q[start:stop].reshape(1, layout.seq_len, self.heads, self.head_dim)
            k_full = k[start:stop].reshape(1, layout.seq_len, self.heads, self.head_dim)
            v_full = v[start:stop].reshape(1, layout.seq_len, self.heads, self.head_dim)
            attn_full = self._attention(q_full, k_full, v_full)
            attn_chunks[start:stop] = attn_full.reshape(layout.chunk_count, plan.chunk_size, self.hidden_size)

        chunks = residual + self.out_proj(attn_chunks)
        chunks = chunks + self._mlp(self.norm2(chunks))
        return chunks


class HunyuanLikeStack(nn.Module):
    def __init__(self, layers: int, hidden_size: int, intermediate_size: int, heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            HunyuanLikeBlock(hidden_size=hidden_size, intermediate_size=intermediate_size, heads=heads)
            for _ in range(layers)
        )

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward_full(x)
        return x

    def forward_mixfusion(self, chunks: torch.Tensor, plan: MixFusionPlan) -> torch.Tensor:
        for layer in self.layers:
            chunks = layer.forward_mixfusion(chunks, plan)
        return chunks


def parse_image_sizes(raw: str, downsample: int, patch_size: int) -> list[tuple[int, int, int]]:
    sizes = []
    divisor = downsample * patch_size
    for item in raw.split(","):
        h_raw, w_raw = item.lower().split("x", maxsplit=1)
        height = int(h_raw)
        width = int(w_raw)
        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(f"Image size {height}x{width} must be divisible by {divisor}.")
        token_h = height // divisor
        token_w = width // divisor
        sizes.append((token_h, token_w, token_h * token_w))
    return sizes


def build_plan(seq_lens: list[int]) -> MixFusionPlan:
    chunk_size = math.gcd(*seq_lens)
    layouts = []
    chunk_start = 0
    for seq_len in seq_lens:
        chunk_count = seq_len // chunk_size
        layouts.append(RequestLayout(seq_len=seq_len, chunk_start=chunk_start, chunk_count=chunk_count))
        chunk_start += chunk_count
    return MixFusionPlan(chunk_size=chunk_size, layouts=tuple(layouts))


def make_sequences(
    seq_lens: list[int],
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    return [torch.randn(1, seq_len, hidden_size, dtype=dtype, device=device) for seq_len in seq_lens]


def make_padded(sequences: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(seq.shape[1] for seq in sequences)
    padded = sequences[0].new_zeros((len(sequences), max_len, sequences[0].shape[-1]))
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.shape[1]] = seq[0]
    return padded


def make_mixfusion_chunks(sequences: list[torch.Tensor], plan: MixFusionPlan) -> torch.Tensor:
    chunks = []
    for seq, layout in zip(sequences, plan.layouts, strict=True):
        chunks.append(seq.reshape(layout.chunk_count, plan.chunk_size, seq.shape[-1]))
    return torch.cat(chunks, dim=0)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn: Callable[[], object], device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def attention_work(seq_lens: list[int]) -> dict[str, int]:
    max_len = max(seq_lens)
    return {
        "independent": sum(seq_len * seq_len for seq_len in seq_lens),
        "padding": len(seq_lens) * max_len * max_len,
        "mixfusion": sum(seq_len * seq_len for seq_len in seq_lens),
    }


def token_work(seq_lens: list[int]) -> dict[str, int]:
    max_len = max(seq_lens)
    return {
        "independent": sum(seq_lens),
        "padding": len(seq_lens) * max_len,
        "mixfusion": sum(seq_lens),
    }


def resolve_dtype(raw: str) -> torch.dtype:
    if raw == "float16":
        return torch.float16
    if raw == "bfloat16":
        return torch.bfloat16
    if raw == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HunyuanImage-3.0-style 4-layer MixFusion.")
    parser.add_argument("--image-sizes", default="1024x1024,512x512")
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    token_shapes = parse_image_sizes(args.image_sizes, args.downsample, args.patch_size)
    seq_lens = [seq_len for _, _, seq_len in token_shapes]
    plan = build_plan(seq_lens)

    model = HunyuanLikeStack(
        layers=args.layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        heads=args.heads,
    ).to(device=device, dtype=dtype)
    model.eval()

    sequences = make_sequences(seq_lens, args.hidden_size, dtype, device)
    padded = make_padded(sequences)
    chunks = make_mixfusion_chunks(sequences, plan)

    with torch.inference_mode():
        independent_ms = time_ms(
            lambda: [model.forward_full(seq) for seq in sequences],
            device,
            args.warmup,
            args.iters,
        )
        padding_ms = time_ms(lambda: model.forward_full(padded), device, args.warmup, args.iters)
        mixfusion_ms = time_ms(lambda: model.forward_mixfusion(chunks, plan), device, args.warmup, args.iters)

    attn = attention_work(seq_lens)
    tokens = token_work(seq_lens)
    result = {
        "config": {
            "image_sizes": args.image_sizes,
            "token_shapes": [(h, w) for h, w, _ in token_shapes],
            "seq_lens": seq_lens,
            "chunk_size": plan.chunk_size,
            "total_chunks": plan.total_chunks,
            "layers": args.layers,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "heads": args.heads,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "time_ms": {
            "independent": independent_ms,
            "padding": padding_ms,
            "mixfusion": mixfusion_ms,
        },
        "speedup": {
            "mixfusion_vs_independent": independent_ms / mixfusion_ms,
            "mixfusion_vs_padding": padding_ms / mixfusion_ms,
        },
        "relative_work": {
            "token_vs_padding": {
                "independent": tokens["independent"] / tokens["padding"],
                "mixfusion": tokens["mixfusion"] / tokens["padding"],
            },
            "attention_vs_padding": {
                "independent": attn["independent"] / attn["padding"],
                "mixfusion": attn["mixfusion"] / attn["padding"],
            },
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
