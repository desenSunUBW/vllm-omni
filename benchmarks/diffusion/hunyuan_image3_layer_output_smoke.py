# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test HunyuanImage-3.0 batched-prompt output count with few DiT layers.

This script is intentionally a correctness smoke, not a quality benchmark. It
loads the real HunyuanImage-3.0 pipeline, truncates the DiT stack to a small
number of layers on every diffusion worker, then checks that one batched
diffusion request with N prompts returns N images.

Example:
    python benchmarks/diffusion/hunyuan_image3_layer_output_smoke.py \
        --model tencent/HunyuanImage-3.0-Instruct \
        --image-sizes 512x512,512x768 \
        --num-layers 1 --steps 1 --tensor-parallel-size 2 \
        --output-dir /tmp/hunyuan_layer_smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt


class HunyuanLayerSmokeWorkerExtension:
    """Worker RPC extension used by the smoke script.

    Multiproc diffusion workers own the loaded pipeline, so the driver process
    cannot directly mutate the model. This method is called through
    ``DiffusionExecutor.collective_rpc`` and runs on every worker rank.
    """

    def truncate_hunyuan_layers(self, num_layers: int) -> dict[str, Any]:
        pipeline = self.model_runner.pipeline
        if pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")

        model = getattr(pipeline, "model", None)
        layers = getattr(model, "layers", None)
        if layers is None:
            nested_model = getattr(model, "model", None)
            layers = getattr(nested_model, "layers", None)

        if layers is None:
            raise RuntimeError("Could not find Hunyuan DiT layers on the loaded pipeline.")
        if not isinstance(layers, nn.ModuleList):
            raise TypeError(f"Expected Hunyuan layers to be nn.ModuleList, got {type(layers)!r}.")

        original_layers = len(layers)
        if num_layers < 1 or num_layers > original_layers:
            raise ValueError(f"--num-layers must be in [1, {original_layers}], got {num_layers}.")

        truncated = nn.ModuleList(list(layers[:num_layers]))
        if hasattr(model, "layers"):
            model.layers = truncated
        else:
            model.model.layers = truncated

        return {
            "original_layers": original_layers,
            "active_layers": num_layers,
            "pipeline": type(pipeline).__name__,
            "model": type(model).__name__,
        }


def parse_image_sizes(raw: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in raw.split(","):
        h_raw, w_raw = item.lower().split("x", maxsplit=1)
        sizes.append((int(h_raw), int(w_raw)))
    if not sizes:
        raise ValueError("At least one image size is required.")
    return sizes


def parse_prompts(raw: str | None, count: int) -> list[str]:
    if raw is not None:
        prompts = [item.strip() for item in raw.split("||") if item.strip()]
        if len(prompts) != count:
            raise ValueError(f"Expected {count} prompts separated by '||', got {len(prompts)}.")
        return prompts

    defaults = [
        "A small glass observatory on Mars at sunrise",
        "A watercolor painting of a quiet mountain lake",
        "A transparent mechanical keyboard on a studio desk",
        "A cozy reading room with warm sunlight and plants",
        "A futuristic train station under soft rain",
        "A ceramic teapot shaped like a cloud",
        "A minimal product photo of a silver camera",
        "A fantasy library with floating lanterns",
    ]
    prompts = defaults[:count]
    while len(prompts) < count:
        prompts.append(f"A high quality image sample {len(prompts)}")
    return prompts


def build_prompt_dicts(prompts: list[str], sizes: list[tuple[int, int]]) -> list[OmniTextPrompt]:
    return [
        {
            "prompt": prompt,
            "height": height,
            "width": width,
            "modalities": ["image"],
        }
        for prompt, (height, width) in zip(prompts, sizes, strict=True)
    ]


def make_config(args: argparse.Namespace) -> OmniDiffusionConfig:
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        sequence_parallel_size=args.ulysses_degree * args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
    )
    config = OmniDiffusionConfig(
        model=args.model,
        model_class_name=args.model_class_name,
        trust_remote_code=True,
        dtype=getattr(torch, args.dtype),
        distributed_executor_backend=args.distributed_executor_backend,
        enforce_eager=args.enforce_eager,
        parallel_config=parallel_config,
        quantization_config=args.quantization,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        output_type=args.output_type,
        max_num_seqs=1,
        worker_extension_cls=(
            "benchmarks.diffusion.hunyuan_image3_layer_output_smoke.HunyuanLayerSmokeWorkerExtension"
        ),
    )
    config.enrich_config()
    return config


def make_sampling_params(args: argparse.Namespace, first_size: tuple[int, int]) -> OmniDiffusionSamplingParams:
    height, width = first_size
    return OmniDiffusionSamplingParams(
        height=height,
        width=width,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_provided=True,
        num_outputs_per_prompt=1,
        extra_args={
            "enable_mixfusion": args.enable_mixfusion,
            "mixfusion_min_chunk_tokens": args.mixfusion_min_chunk_tokens,
            "mixfusion_max_chunks": args.mixfusion_max_chunks,
            "use_system_prompt": args.use_system_prompt,
            "system_prompt": args.system_prompt,
        },
    )


def count_images(outputs: list[Any]) -> int:
    image_count = 0
    for output in outputs:
        images = getattr(output, "images", None) or []
        image_count += len(images)
    return image_count


def save_outputs(outputs: list[Any], output_dir: str | None) -> None:
    if output_dir is None:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_idx = 0
    for output in outputs:
        images = getattr(output, "images", None) or []
        for image in images:
            image.save(out_dir / f"batched_{image_idx}.png")
            image_idx += 1


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    sizes = parse_image_sizes(args.image_sizes)
    prompts = parse_prompts(args.prompts, len(sizes))
    prompt_dicts = build_prompt_dicts(prompts, sizes)

    config = make_config(args)
    engine = DiffusionEngine(config)
    try:
        truncate_info = engine.executor.collective_rpc(
            "truncate_hunyuan_layers",
            args=(args.num_layers,),
            timeout=args.rpc_timeout,
        )

        request_id = f"hunyuan-layer-smoke-{uuid.uuid4()}"
        request = OmniDiffusionRequest(
            prompts=prompt_dicts,
            sampling_params=make_sampling_params(args, sizes[0]),
            request_ids=[f"{request_id}-{idx}" for idx in range(len(prompt_dicts))],
            request_id=request_id,
        )

        start = time.perf_counter()
        outputs = await engine.step(request)
        elapsed = time.perf_counter() - start
        actual_images = count_images(outputs)
        save_outputs(outputs, args.output_dir)

        result = {
            "status": "passed" if actual_images == len(prompt_dicts) else "failed",
            "expected_images": len(prompt_dicts),
            "actual_images": actual_images,
            "num_request_outputs": len(outputs),
            "elapsed_s": elapsed,
            "image_sizes": sizes,
            "num_layers": args.num_layers,
            "steps": args.steps,
            "enable_mixfusion": args.enable_mixfusion,
            "truncate_info": truncate_info,
        }
        print(json.dumps(result, indent=2, default=str))
        if actual_images != len(prompt_dicts):
            raise AssertionError(f"Expected {len(prompt_dicts)} images, got {actual_images}.")
        return result
    finally:
        engine.executor.shutdown()


def validate_args(args: argparse.Namespace) -> None:
    sizes = parse_image_sizes(args.image_sizes)
    if len(sizes) < 1:
        raise ValueError("--image-sizes must contain at least one size.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.num_layers < 1:
        raise ValueError("--num-layers must be >= 1.")
    if args.cfg_parallel_size != 1:
        raise ValueError("This smoke test requires --cfg-parallel-size 1.")
    if args.ulysses_degree != 1 or args.ring_degree != 1:
        raise ValueError("This smoke test requires sequence parallel disabled: --ulysses-degree 1 --ring-degree 1.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-class-name", default="HunyuanImage3Pipeline")
    parser.add_argument("--image-sizes", default="512x512,512x768")
    parser.add_argument("--prompts", default=None, help="Prompts separated by '||'. Must match --image-sizes count.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae-use-slicing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vae-use-tiling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-diffusion-pipeline-profiler", action="store_true")
    parser.add_argument("--enable-mixfusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixfusion-min-chunk-tokens", type=int, default=256)
    parser.add_argument("--mixfusion-max-chunks", type=int, default=128)
    parser.add_argument("--use-system-prompt", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--output-type", default="pil", choices=["pil", "latent"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rpc-timeout", type=float, default=120.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
