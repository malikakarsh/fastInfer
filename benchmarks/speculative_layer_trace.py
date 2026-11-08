"""Locate the first module-level difference in a saved Qwen2 ranking flip.

Accuracy only, for user execution. No generation, draft model, or performance
measurement. Compares matching causal rows in sequential versus block forwards,
checks hooks against uninstrumented logits, and probes the first independent
per-token module with equal input rows but different outputs.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from speculative_divergence import describe_logits, recover_first_block, replay


def cpu_double(tensor):
    # Separate operations are intentional: MPS may attempt dtype conversion on
    # the source device for a combined .to(device="cpu", dtype=torch.float64).
    return tensor.detach().cpu().double()


def difference(actual, expected):
    a, b = cpu_double(actual), cpu_double(expected)
    if a.shape != b.shape:
        return {"shape_matches": False, "actual_shape": list(a.shape), "expected_shape": list(b.shape)}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"shape_matches": True, "finite": False, "equal": False}
    error = (a - b).abs()
    result = {"shape_matches": True, "finite": True,
              "equal": bool(torch.equal(a, b)),
              "different_values": int((a != b).sum()),
              "max_abs": float(error.max()), "mean_abs": float(error.mean()),
              "rms_error": float(error.square().mean().sqrt())}
    if a.ndim == 2:
        result["different_values_per_token"] = (a != b).sum(-1).tolist()
    return result


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        return next((item for item in output if isinstance(item, torch.Tensor)), None)
    return None


def independent_module(module):
    return isinstance(module, (torch.nn.Linear, torch.nn.SiLU)) or module.__class__.__name__ == "Qwen2RMSNorm"


class Trace:
    def __init__(self, rows):
        self.rows = rows
        self.active = False
        self.values = {}
        self.shapes = {}
        self.inputs = {}
        self.order = []

    def record(self, key, tensor):
        if not self.active or tensor is None or tensor.ndim != 3 or tensor.shape[0] != 1:
            return
        if key not in self.values:
            self.order.append(key)
            self.values[key] = []
            self.shapes[key] = []
        # Retain only the common causal rows. Block's future rows are excluded
        # from comparisons, but retained separately for independent-op probes.
        self.values[key].append(tensor[0, :self.rows].detach().cpu().clone())
        self.shapes[key].append({"shape": list(tensor.shape), "stride": list(tensor.stride()),
                                 "dtype": str(tensor.dtype)})

    def before(self, name, module, args, kwargs):
        tensor = first_tensor(args)
        if tensor is None:
            tensor = kwargs.get("hidden_states")
        self.record(name + ".input", tensor)
        if self.active and tensor is not None and independent_module(module):
            self.inputs.setdefault(name, []).append(tensor.detach().cpu().clone())
        if self.active and name.endswith("self_attn"):
            embeddings = kwargs.get("position_embeddings")
            if embeddings is not None:
                self.record(name + ".rotary_cos", embeddings[0])
                self.record(name + ".rotary_sin", embeddings[1])

    def after(self, name, output):
        self.record(name + ".output", first_tensor(output))

    def joined(self, key):
        result = torch.cat(self.values[key], dim=0)
        if result.shape[0] != self.rows:
            raise ValueError(f"wrong causal row count for {key}: {result.shape[0]}")
        return result

    @contextmanager
    def attach(self, model):
        handles = []
        names = {"embed_tokens", "input_layernorm", "post_attention_layernorm", "self_attn",
                 "q_proj", "k_proj", "v_proj", "o_proj", "mlp", "gate_proj", "up_proj",
                 "down_proj", "act_fn", "norm", "lm_head"}
        try:
            for name, module in model.named_modules():
                if name.split(".")[-1] not in names and not (name.startswith("model.layers.") and name.split(".")[-1].isdigit()):
                    continue
                def before(module, args, kwargs, name=name):
                    self.before(name, module, args, kwargs)
                def after(module, args, kwargs, output, name=name):
                    self.after(name, output)
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after, with_kwargs=True))
            yield
        finally:
            self.active = False
            for handle in handles:
                handle.remove()


def cache_snapshot(cache, length):
    result = {}
    for index, layer in enumerate(cache.layers):
        for kind in ("keys", "values"):
            value = getattr(layer, kind, None)
            if value is None:
                raise ValueError("expected ordinary Qwen2 cache tensors")
            result[f"layer{index}.{kind}"] = value[..., :length, :].detach().cpu().clone()
    return result


@torch.inference_mode()
def capture(model, prompt, common, block, sequential):
    trace = Trace(len(common))
    cache = DynamicCache(config=model.config)
    with trace.attach(model):
        model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        prefill_cache = cache_snapshot(cache, prompt.shape[1])
        trace.active = True
        if sequential:
            for token in common:
                output = model(input_ids=prompt.new_tensor([[token]]), past_key_values=cache,
                               use_cache=True, logits_to_keep=1, return_dict=True)
            vector = output.logits[0, -1].detach().cpu().clone()
        else:
            output = model(input_ids=prompt.new_tensor([block]), past_key_values=cache,
                           use_cache=True, logits_to_keep=0, return_dict=True)
            vector = output.logits[0, len(common) - 1].detach().cpu().clone()
        shared_cache = cache_snapshot(cache, prompt.shape[1] + len(common))
    return trace, vector, prefill_cache, shared_cache


@torch.inference_mode()
def probe(module, full_input, common_rows, device):
    """Replay identical input rows in different shapes, keeping weights fixed."""
    x = full_input.to(device)
    full = module(x)[:, :common_rows].detach().cpu()
    single = torch.cat([module(x[:, i:i + 1].contiguous()).detach().cpu()
                        for i in range(common_rows)], dim=1)
    full_repeat = module(x)[:, :common_rows].detach().cpu()
    single_repeat = torch.cat([module(x[:, i:i + 1].contiguous()).detach().cpu()
                               for i in range(common_rows)], dim=1)
    x64 = cpu_double(full_input[:, :common_rows])
    if isinstance(module, torch.nn.Linear):
        # Move before conversion: MPS does not support FP64 tensors.
        w64 = cpu_double(module.weight)
        bias64 = cpu_double(module.bias) if module.bias is not None else None
        reference = F.linear(x64, w64, bias64)
        reference32 = reference
        # The FP32 control uses the SAME stored inputs/weights, not a newly
        # downloaded or differently trained checkpoint.
        w32 = module.weight.detach().float()
        bias32 = module.bias.detach().float() if module.bias is not None else None
        x32 = x.float()
        full32 = F.linear(x32, w32, bias32)[:, :common_rows].cpu()
        single32 = torch.cat([F.linear(x32[:, i:i + 1].contiguous(), w32, bias32).cpu()
                              for i in range(common_rows)], dim=1)
    elif isinstance(module, torch.nn.SiLU):
        reference = F.silu(x64)
        reference32 = reference
        full32 = F.silu(x.float())[:, :common_rows].cpu()
        single32 = torch.cat([F.silu(x[:, i:i + 1].float().contiguous()).cpu()
                              for i in range(common_rows)], dim=1)
    else:
        # High-precision RMSNorm equation retaining the output-dtype cast before
        # learned-weight multiplication, as in the original Qwen2 module.
        normalized = x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + module.variance_epsilon)
        weight64 = cpu_double(module.weight)
        reference = normalized.to(full_input.dtype).double() * weight64
        reference32 = normalized * weight64
        x32 = x.float()
        def normalize(value):
            return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + module.variance_epsilon) * module.weight.detach().float()
        full32 = normalize(x32)[:, :common_rows].cpu()
        single32 = torch.cat([normalize(x32[:, i:i + 1].contiguous()).cpu()
                              for i in range(common_rows)], dim=1)
    rounded = reference.to(full.dtype)
    mismatch_details = []
    for coordinates in (full != single).nonzero()[:8].tolist():
        coordinate = tuple(coordinates)
        block_value = float(full[coordinate])
        single_value = float(single[coordinate])
        midpoint = (block_value + single_value) / 2
        exact_reference = float(reference[coordinate])
        mismatch_details.append({"coordinates": coordinates,
                                 "block_value": block_value, "single_value": single_value,
                                 "cpu_fp64_reference": exact_reference,
                                 "rounded_cpu_fp64": float(rounded[coordinate]),
                                 "midpoint_between_outputs": midpoint,
                                 "reference_minus_midpoint": exact_reference - midpoint,
                                 "fp32_block_value": float(full32[coordinate]),
                                 "fp32_single_value": float(single32[coordinate])})
    return {"module_type": module.__class__.__name__, "full_input_shape": list(x.shape),
            "single_input_shape": [1, 1, x.shape[-1]],
            "shape_difference_reproduced": not torch.equal(full, single),
            "full_vs_single": difference(full, single),
            "full_repeat": difference(full_repeat, full), "single_repeat": difference(single_repeat, single),
            "full_vs_cpu_fp64": difference(full, reference),
            "single_vs_cpu_fp64": difference(single, reference),
            "full_vs_rounded_cpu_fp64": difference(full, rounded),
            "single_vs_rounded_cpu_fp64": difference(single, rounded),
            "fp32_full_vs_single": difference(full32, single32),
            "fp32_full_vs_cpu_fp64": difference(full32, reference32),
            "fp32_single_vs_cpu_fp64": difference(single32, reference32),
            "first_differing_values": mismatch_details,
            "note": "CPU FP64 is a numerical reference, not proof of correct rounding. "
                    "A shape difference alone does not establish a backend bug. Inputs "
                    "are restored contiguously; original trace strides are recorded separately."}


@torch.inference_mode()
def intervene(model, prompt, common, block, name, source, destination, sequential, cache_factory=None):
    """Replace only differing coordinates, with exact source input/output guards.

    Self-to-self is the control for hook/cloning effects. Cross-path swaps test
    whether the first local difference influences the downstream ranking while
    preserving the original execution shape and every other module output value.
    """
    module = dict(model.named_modules())[name]
    source_rows = source.joined(name + ".output")
    replacement_rows = destination.joined(name + ".output")
    mask = source_rows != replacement_rows
    cache = cache_factory() if cache_factory is not None else DynamicCache(config=model.config)
    active = False
    calls = 0
    changed = 0

    def hook(module, args, kwargs, output):
        nonlocal calls, changed
        if not active:
            return None
        inputs = first_tensor(args)
        if inputs is None:
            inputs = kwargs.get("hidden_states")
        expected_input = source.inputs[name][calls if sequential else 0]
        if inputs is None or not torch.equal(inputs.detach().cpu(), expected_input):
            raise RuntimeError("intervention source inputs changed; refusing to patch a different computation")
        start, stop = (calls, calls + 1) if sequential else (0, len(common))
        selected = output[0, :stop - start].detach().cpu()
        if not torch.equal(selected, source_rows[start:stop]):
            raise RuntimeError("intervention source outputs changed; repeat stability must be investigated")
        local_mask = mask[start:stop].to(output.device)
        replacement = replacement_rows[start:stop].to(output.device)
        patched = output.clone()
        patched[0, :stop - start] = torch.where(local_mask, replacement, output[0, :stop - start])
        changed += int(mask[start:stop].sum())
        calls += 1
        return patched

    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        if cache_factory is None:
            model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        active = True
        if sequential:
            for token in common:
                output = model(input_ids=prompt.new_tensor([[token]]), past_key_values=cache,
                               use_cache=True, logits_to_keep=1, return_dict=True)
            vector = output.logits[0, -1]
        else:
            output = model(input_ids=prompt.new_tensor([block]), past_key_values=cache,
                           use_cache=True, logits_to_keep=0, return_dict=True)
            vector = output.logits[0, len(common) - 1]
        expected_calls = len(common) if sequential else 1
        if calls != expected_calls:
            raise RuntimeError("unexpected number of intervention calls")
        return vector.detach().cpu().clone(), changed
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case-id", type=int, default=151)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--causal-check", action="store_true",
                        help="also swap only the first local output differences, with self-swap controls")
    parser.add_argument("--probe-module", help="probe a specific independent module with identical input rows")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    saved = json.loads(args.report.read_text())
    case = next((c for c in saved["cases"] if c["id"] == args.case_id), None)
    if case is None:
        parser.error("case not found")
    index, baseline, candidate, block = recover_first_block(case, saved["settings"]["gamma"])
    device = torch.device(args.device or saved["environment"]["device"])
    dtype = getattr(torch, args.dtype or saved["environment"]["dtype"].split(".")[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or args.report.with_name(f"speculative_layer_trace_{args.case_id}_{stamp}.json")
    if destination.exists():
        parser.error("output exists; choose a new filename")
    report = {"status": "running", "source_report": str(args.report.resolve()),
              "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ("speculative_layer_trace.py", "speculative_divergence.py")},
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "device": str(device), "dtype": str(dtype), "attention": args.attention,
                              "mps_flags": {key: os.environ.get(key) for key in (
                                  "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK")}},
              "target_snapshot": saved["target_snapshot"], "case_id": args.case_id,
              "common_history": baseline[:index], "block": block,
              "scope": "Module-level causal-row trace and one independent-op replay. "
                       "No generation, draft model, rollback, timing or automatic bug verdict."}

    def save():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(saved["target_snapshot"], local_files_only=True)
        text = case["prompt"] if saved["settings"]["prompt_format"] == "raw" else f"Question: {case['prompt']}\nAnswer:"
        prompt = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if prompt.shape[1] != case["input_tokens"]:
            raise ValueError("prompt reconstruction differs from saved report")
        print(f"Loading target only on {device} as {dtype}; layer diagnostic, no benchmark", flush=True)
        model = AutoModelForCausalLM.from_pretrained(saved["target_snapshot"], local_files_only=True,
                    dtype=dtype, attn_implementation=args.attention).to(device).eval()
        # Establish uninstrumented logits first so hook effects are detectable.
        controls = {}
        for path in ("sequential_cached", "verification_block"):
            vector, _ = replay(model, prompt, baseline[:index], block, index - 1, path)
            controls[path] = vector.detach().cpu().clone()
        seq, seq_logits, seq_prefill, seq_cache = capture(model, prompt, baseline[:index], block, True)
        bat, bat_logits, bat_prefill, bat_cache = capture(model, prompt, baseline[:index], block, False)
        report["hook_controls"] = {"sequential": difference(seq_logits, controls["sequential_cached"]),
                                   "block": difference(bat_logits, controls["verification_block"])}
        report["hooks_preserve_logits"] = all(c.get("equal", False) for c in report["hook_controls"].values())
        report["logits"] = {"sequential": describe_logits(seq_logits, tokenizer, (baseline[index], candidate[index]))[0],
                            "block": describe_logits(bat_logits, tokenizer, (baseline[index], candidate[index]))[0]}
        report["prefill_cache_comparisons"] = {key: difference(bat_prefill[key], value) for key, value in seq_prefill.items()}
        report["shared_cache_comparisons"] = {key: difference(bat_cache[key], value) for key, value in seq_cache.items()}
        report["stages"] = []
        for key in seq.order:
            if key not in bat.values:
                raise ValueError(f"block trace missing {key}")
            report["stages"].append({"stage": key, "comparison": difference(bat.joined(key), seq.joined(key)),
                                      "sequential_calls": seq.shapes[key], "block_calls": bat.shapes[key]})
        differing = [stage for stage in report["stages"] if not stage["comparison"].get("equal", False)]
        report["first_differing_stage"] = differing[0] if differing else None
        print("Hooks preserve logits:", report["hooks_preserve_logits"], flush=True)
        print("First differing stage:", json.dumps(report["first_differing_stage"], indent=2), flush=True)
        modules = dict(model.named_modules())
        selected = None
        for stage in report["stages"]:
            key = stage["stage"]
            if not key.endswith(".output") or stage["comparison"].get("equal", False):
                continue
            name = key[:-len(".output")]
            input_key = name + ".input"
            if name in bat.inputs and input_key in seq.values and torch.equal(seq.joined(input_key), bat.joined(input_key)):
                selected = name
                break
        report["first_independent_module_with_equal_inputs"] = selected
        if args.probe_module:
            requested = args.probe_module
            input_key, output_key = requested + ".input", requested + ".output"
            if requested not in bat.inputs or input_key not in seq.values or output_key not in seq.values:
                raise ValueError("requested probe must name a traced Linear, RMSNorm or SiLU module")
            if not torch.equal(seq.joined(input_key), bat.joined(input_key)):
                raise ValueError("requested module's causal input rows differ; cannot isolate shape on identical inputs")
            selected = requested
        report["selected_probe_module"] = selected
        save()
        if selected is not None and report["hooks_preserve_logits"]:
            print("Replaying independent operation:", selected, flush=True)
            report["operator_probe"] = probe(modules[selected], bat.inputs[selected][0], index, device)
            print(json.dumps(report["operator_probe"], indent=2), flush=True)
            save()
            if args.causal_check:
                report["causal_check"] = {"module": selected, "runs": {}}
                variants = (("sequential_self_control", seq, seq, True),
                            ("block_self_control", bat, bat, False),
                            ("sequential_with_block_values", seq, bat, True),
                            ("block_with_sequential_values", bat, seq, False))
                for label, source, destination_trace, sequential in variants:
                    vector, changed = intervene(model, prompt, baseline[:index], block,
                                                selected, source, destination_trace, sequential)
                    description, _ = describe_logits(vector, tokenizer, (baseline[index], candidate[index]))
                    control = seq_logits if sequential else bat_logits
                    report["causal_check"]["runs"][label] = {
                        "changed_coordinates": changed, "scores": description,
                        "versus_original_path": difference(vector, control)}
                    print(f"{label}: changed {changed} values, winner={description['argmax_text']!r}, "
                          f"baseline-minus-speculative={description['baseline_minus_speculative_score']:.8f}", flush=True)
                    save()
                runs = report["causal_check"]["runs"]
                report["causal_check"]["self_controls_preserve_logits"] = all(
                    runs[label]["versus_original_path"].get("equal", False)
                    for label in ("sequential_self_control", "block_self_control"))
                report["causal_check"]["note"] = (
                    "Only interpret swaps if self controls preserve logits. A changed winner establishes "
                    "a causal contribution under that intervention; an unchanged winner means the first "
                    "difference alone was insufficient. Later operators can introduce additional drift. "
                    "Neither result alone proves or excludes a backend correctness bug.")
        report["status"] = "complete" if report["hooks_preserve_logits"] else "hook_control_failed"
        if args.causal_check and "causal_check" not in report:
            report["causal_check"] = {"skipped": True, "reason": "no eligible operation or hook controls failed"}
        elif args.causal_check and not report["causal_check"]["self_controls_preserve_logits"]:
            report["status"] = "causal_control_failed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
