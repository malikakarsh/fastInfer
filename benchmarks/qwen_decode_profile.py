"""Profile a short original-model decode run to locate major costs."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.nn.attention import SDPBackend, sdpa_kernel

from qwen_decode_compare import MODEL, REVISION, ROOT
from qwen_linear_check import require_finite


def component_labels(model):
    """Select model regions; the target projection is nested inside MLP."""
    result = []
    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            label = "component/attention"
        elif name.endswith("mlp"):
            label = "component/mlp"
        elif name.endswith(("input_layernorm", "post_attention_layernorm")) or name == "model.norm":
            label = "component/normalization"
        elif name == "lm_head":
            label = "component/vocabulary_projection"
        elif name == "model.layers.0.mlp.up_proj":
            label = "component/target_up_proj"
        else:
            continue
        result.append((module, label))
    return result


@contextmanager
def annotate_components(model):
    """Label module execution without synchronizing at module boundaries."""
    handles = []
    opened = defaultdict(list)

    def before(module, inputs, label):
        region = record_function(label)
        region.__enter__()
        opened[module].append(region)

    def after(module, inputs, output):
        if opened[module]:
            opened[module].pop().__exit__(None, None, None)

    try:
        for module, label in component_labels(model):
            handles.append(module.register_forward_pre_hook(
                lambda module, inputs, label=label: before(module, inputs, label)
            ))
            handles.append(module.register_forward_hook(after, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for regions in opened.values():
            while regions:
                regions.pop().__exit__(None, None, None)


def prepare_decode(model, encoded):
    from transformers import DynamicCache

    prefill = model(**encoded, past_key_values=DynamicCache(),
                    use_cache=True, return_dict=True)
    require_finite(prefill.logits, "prefill.logits")
    token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return prefill.past_key_values, token, encoded["attention_mask"]


def decode_steps(model, cache, token, mask, steps):
    for _ in range(steps):
        mask = torch.cat((mask, torch.ones_like(token)), dim=1)
        output = model(input_ids=token, attention_mask=mask,
                       past_key_values=cache, use_cache=True, return_dict=True)
        cache = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return output.logits


def event_summary(event):
    return {
        "name": event.key, "calls": event.count,
        "self_cpu_us": event.self_cpu_time_total,
        "total_cpu_us": event.cpu_time_total,
        "self_device_us": event.self_device_time_total,
        "total_device_us": event.device_time_total,
        "device_type": str(event.device_type),
    }


def component_summaries(averages):
    """CPU annotations attribute nested GPU work; device annotations stay separate."""
    return [event_summary(event) for event in averages
            if event.key.startswith("component/")
            and event.device_type == torch.autograd.DeviceType.CPU]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", action="store_true", help="also save a Chrome trace")
    args = parser.parse_args()
    if args.steps < 1 or not torch.cuda.is_available():
        parser.error("positive steps and a CUDA GPU required")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; original model, FP16, math SDPA", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", output_attentions=False,
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    ).to("cuda").eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for _ in range(2):
            logits = decode_steps(model, *prepare_decode(model, encoded), args.steps)
            require_finite(logits, "warmup.logits")
            del logits
        cache, token, mask = prepare_decode(model, encoded)
        torch.cuda.synchronize()
        print(f"Profiling {args.steps} cached decode steps (prefill excluded)...", flush=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_stack=False) as prof:
            with annotate_components(model):
                logits = decode_steps(model, cache, token, mask, args.steps)
            torch.cuda.synchronize()
        require_finite(logits, "profiled_decode.logits")

    averages = list(prof.key_averages())
    regions = component_summaries(averages)
    device_annotations = [event_summary(event) for event in averages
                          if event.key.startswith("component/")
                          and event.device_type == torch.autograd.DeviceType.CUDA]
    raw_device_events = [event for event in prof.events()
                         if event.device_type == torch.autograd.DeviceType.CUDA]
    gpu_events = len(raw_device_events)
    non_component_events = [event for event in raw_device_events
                            if not event.name.startswith("component/")]
    device_us = sum(event.self_device_time_total for event in non_component_events)
    cpu_top = sorted((event_summary(event) for event in averages
                      if not event.key.startswith("component/")),
                     key=lambda row: row["self_cpu_us"], reverse=True)[:15]
    device_top = sorted((event_summary(event) for event in averages
                         if event.device_type == torch.autograd.DeviceType.CPU
                         and not event.key.startswith("component/")),
                        key=lambda row: row["self_device_us"], reverse=True)[:15]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or ROOT / "results/raw" / f"qwen_decode_profile_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_original_decode_profile", "model": MODEL, "revision": REVISION,
        "dtype": "float16", "attention": "sdpa_math", "cache": "DynamicCache",
        "allow_fp16_reduced_precision_reduction": True,
        "prompt": args.prompt, "steps": args.steps, "warmup_trajectories": 2,
        "gpu_activity_events": gpu_events,
        "gpu_non_component_activity_events": len(non_component_events),
        "summed_non_component_device_activity_us": device_us,
        "device_component_annotations": device_annotations,
        "component_regions": regions, "top_self_cpu_operations": cpu_top,
        "top_self_device_operations": device_top,
        "scope": "Diagnostic original-model decode profile. Prefill excluded. Profiling and "
                 "annotation overhead prevent comparison with unprofiled benchmark latency. "
                 "CPU time is host operation duration, not necessarily active CPU computation. "
                 "Component totals include nested work; target_up_proj overlaps MLP. "
                 "GPU component annotations are stored separately from CPU component records. "
                 "Only non-component device events enter the summed activity duration; "
                 "this is not wall time and may overlap.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
    }
    if args.trace:
        trace_path = destination.with_suffix(".trace.json")
        prof.export_chrome_trace(str(trace_path))
        report["trace_path"] = str(trace_path)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GPU activity events captured: {gpu_events}", flush=True)
    if not gpu_events:
        print("GPU activity unavailable: CPU timings cannot establish GPU bottlenecks.", flush=True)
    print("\nModel regions: totals include nested operations; target up_proj is inside MLP.")
    print("Region                          Calls    Host total ms   GPU work ms")
    for row in sorted(regions, key=lambda row: row["total_device_us"], reverse=True):
        print(f"{row['name']:31} {row['calls']:5} {row['total_cpu_us']/1000:15.3f} "
              f"{row['total_device_us']/1000:15.3f}")
    print("\nTop operations by their own GPU time (excluding child operations):")
    for row in device_top[:10]:
        print(f"  {row['name']}: {row['self_device_us']/1000:.3f} ms | {row['calls']} calls")
    print("\nTop operations by their own CPU time (excluding child operations):")
    for row in cpu_top[:10]:
        print(f"  {row['name']}: {row['self_cpu_us']/1000:.3f} ms | {row['calls']} calls")
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
