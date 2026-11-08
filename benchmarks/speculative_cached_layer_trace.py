"""Layer trace for a captured later-round verification, with identical KV state.

Target-only accuracy diagnostic. Loads a locally generated tensor-only replay
artifact after hash verification; no draft, prompt regeneration or benchmarking.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastinfer.speculative_cached import _forward
from speculative_divergence import describe_logits
from speculative_failure_trace import restore_cache, pack_cache, cache_equal
from speculative_layer_trace import Trace, difference, probe, intervene


def validate_artifact(data):
    block, row, prefix = data["verification_input_ids"], data["row"], data["prefix_token_ids"]
    if (not isinstance(block, list) or not block or not isinstance(row, int)
            or not 0 <= row < len(block) or not isinstance(prefix, list) or not prefix):
        raise ValueError("invalid verification block, selected row or prefix")
    if any(not isinstance(token, int) or token < 0 for token in prefix + block):
        raise ValueError("token IDs must be nonnegative integers")
    if not data["cache"]:
        raise ValueError("replay artifact has no cache")
    dtype = data["cache"][0]["keys"].dtype
    for entry in data["cache"]:
        for kind in ("keys", "values"):
            tensor = entry[kind]
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2] != len(prefix)
                    or tensor.dtype != dtype or not tensor.is_floating_point()):
                raise ValueError("cache layers must share dtype and match the captured prefix length")
            if len(entry[kind + "_stride"]) != 4:
                raise ValueError("cache strides must have four dimensions")
    return dtype


def candidate_modules(sequential, block):
    result = []
    for key in sequential.order:
        if not key.endswith(".output") or key not in block.values:
            continue
        name = key[:-len(".output")]
        before = name + ".input"
        if (name in block.inputs and before in sequential.values and before in block.values
                and torch.equal(sequential.joined(before), block.joined(before))
                and not torch.equal(sequential.joined(key), block.joined(key))):
            result.append(name)
    return result


@torch.inference_mode()
def replay_artifact(model, data, device, sequential, trace=None):
    cache = restore_cache(model, data["cache"], device)
    if not cache_equal(pack_cache(cache), data["cache"]):
        raise RuntimeError("restored cache differs from artifact before the forward pass")
    count = data["row"] + 1
    common = data["verification_input_ids"][:count]

    def forward():
        if sequential:
            for token in common:
                logits = _forward(model, torch.tensor([[token]], dtype=torch.long, device=device), cache, keep=1)
            return logits[0, -1]
        inputs = torch.tensor([data["verification_input_ids"]], dtype=torch.long, device=device)
        return _forward(model, inputs, cache)[0, data["row"]]

    if trace is None:
        vector = forward().detach().cpu().clone()
    else:
        with trace.attach(model):
            trace.active = True
            vector = forward().detach().cpu().clone()
    if not cache_equal(pack_cache(cache, len(data["prefix_token_ids"])), data["cache"]):
        raise RuntimeError("forward pass modified retained prior-cache values")
    shared_cache = pack_cache(cache, len(data["prefix_token_ids"]) + count)
    return vector, shared_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="saved speculative_failure_trace JSON")
    parser.add_argument("--case-id", type=int, default=148)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--causal-check", action="store_true",
                        help="screen equal-input candidate operations with guarded cross-swaps")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--probe-module", help="restrict probes to this equal-input independent module")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_candidates < 1:
        parser.error("max-candidates must be positive")
    saved = json.loads(args.report.read_text())
    case = next((case for case in saved["cases"] if case["id"] == args.case_id), None)
    if case is None or not case.get("replay_artifact"):
        parser.error("selected case must have a captured replay artifact")
    artifact_path = Path(case["replay_artifact"])
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    if not artifact_path.exists():
        artifact_path = args.report.resolve().parent / Path(case["replay_artifact"]).name
    actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_sha != case["replay_artifact_sha256"]:
        parser.error("artifact SHA256 differs from trace report")
    data = torch.load(artifact_path, map_location="cpu", weights_only=True)
    dtype = validate_artifact(data)
    device = torch.device(args.device or saved["environment"]["device"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or args.report.with_name(f"speculative_cached_layer_trace_{args.case_id}_{stamp}.json")
    if destination.exists():
        parser.error("output exists; choose a new filename")
    report = {"status": "running", "case_id": args.case_id,
              "source_report": str(args.report.resolve()),
              "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "artifact": str(artifact_path), "artifact_sha256": actual_sha,
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ("speculative_cached_layer_trace.py", "speculative_layer_trace.py",
                                              "speculative_failure_trace.py", "speculative_divergence.py")},
              "target_snapshot": data["target_snapshot"],
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "device": str(device), "dtype": str(dtype), "attention": "sdpa"},
              "prefix_tokens": len(data["prefix_token_ids"]),
              "verification_input_ids": data["verification_input_ids"], "selected_row": data["row"],
              "scope": "Target-only accuracy replay from identical captured KV values/strides. "
                       "No draft, generation, prefill, cache-history rebuild or performance measurement. "
                       "Bounded causal screening identifies contributors, not an automatic backend bug."}
    report["source_sha256"]["speculative_cached.py"] = hashlib.sha256((ROOT / "speculative_cached.py").read_bytes()).hexdigest()

    def save():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(data["target_snapshot"], local_files_only=True)
        print(f"Loading target only on {device} as {dtype}; captured case {args.case_id}, no benchmark", flush=True)
        model = AutoModelForCausalLM.from_pretrained(data["target_snapshot"], local_files_only=True,
                    dtype=dtype, attn_implementation="sdpa").to(device).eval()
        if model.config.model_type != "qwen2" or len(data["cache"]) != len(model.model.layers):
            raise ValueError("captured cache must match the original Qwen2 target layers")
        watched = (data["baseline_token"], data["speculative_token"])
        controls = {}
        for name, sequential in (("sequential", True), ("block", False)):
            vectors = [replay_artifact(model, data, device, sequential)[0] for _ in range(2)]
            controls[name] = vectors[0]
            report[name + "_repeat"] = difference(vectors[1], vectors[0])
        count = data["row"] + 1
        seq, bat = Trace(count), Trace(count)
        seq_logits, seq_cache = replay_artifact(model, data, device, True, seq)
        bat_logits, bat_cache = replay_artifact(model, data, device, False, bat)
        report["hook_controls"] = {"sequential": difference(seq_logits, controls["sequential"]),
                                   "block": difference(bat_logits, controls["block"])}
        report["block_matches_captured_verification"] = difference(controls["block"], data["actual_verification_logits"])
        report["sequential_vs_original_baseline"] = difference(controls["sequential"], data["baseline_logits"])
        report["logits"] = {name: describe_logits(vector, tokenizer, watched)[0] for name, vector in controls.items()}
        report["controls_passed"] = (
            all(report[name + "_repeat"].get("equal", False) for name in controls)
            and all(value.get("equal", False) for value in report["hook_controls"].values())
            and report["block_matches_captured_verification"].get("equal", False))
        report["winner_difference_reproduced"] = int(seq_logits.argmax()) != int(bat_logits.argmax())
        for name in controls:
            scores = report["logits"][name]
            print(f"{name}: winner={scores['argmax_text']!r}, "
                  f"baseline-minus-speculative={scores['baseline_minus_speculative_score']:.8f}", flush=True)
        print("Controls passed:", report["controls_passed"], flush=True)
        report["shared_cache_comparisons"] = {
            f"layer{index}.{kind}": difference(b[kind], a[kind])
            for index, (a, b) in enumerate(zip(seq_cache, bat_cache)) for kind in ("keys", "values")}
        report["stages"] = []
        for key in seq.order:
            if key not in bat.values:
                raise ValueError(f"missing block trace stage {key}")
            report["stages"].append({"stage": key, "comparison": difference(bat.joined(key), seq.joined(key)),
                                      "sequential_calls": seq.shapes[key], "block_calls": bat.shapes[key]})
        differing = [stage for stage in report["stages"] if not stage["comparison"].get("equal", False)]
        report["first_differing_stage"] = differing[0] if differing else None
        candidates = candidate_modules(seq, bat)
        report["equal_input_candidate_modules"] = candidates
        print("First differing stage:", json.dumps(report["first_differing_stage"], indent=2), flush=True)
        print("Equal-input candidates:", candidates, flush=True)
        if args.probe_module:
            if args.probe_module not in candidates:
                raise ValueError("requested module must have equal causal inputs and different outputs")
            candidates = [args.probe_module]
        report["screening_limit"] = args.max_candidates
        report["screening_truncated"] = len(candidates) > args.max_candidates
        selected = candidates[0] if candidates else None
        save()
        if report["controls_passed"] and report["winner_difference_reproduced"]:
            common = data["verification_input_ids"][:count]
            block = data["verification_input_ids"]
            dummy_prompt = torch.tensor([common], dtype=torch.long, device=device)
            factory = lambda: restore_cache(model, data["cache"], device)
            if args.causal_check:
                report["causal_screen"] = []
                for name in candidates[:args.max_candidates]:
                    screening = {"module": name, "runs": {}}
                    report["causal_screen"].append(screening)
                    for label, source, replacement, sequential in (
                        ("sequential_self_control", seq, seq, True),
                        ("block_self_control", bat, bat, False),
                        ("sequential_with_block_values", seq, bat, True),
                        ("block_with_sequential_values", bat, seq, False)):
                        vector, changed = intervene(model, dummy_prompt, common, block, name,
                            source, replacement, sequential, cache_factory=factory)
                        scores, _ = describe_logits(vector, tokenizer, watched)
                        screening["runs"][label] = {"changed_coordinates": changed, "scores": scores,
                            "versus_original_path": difference(vector, controls["sequential" if sequential else "block"])}
                    runs = screening["runs"]
                    screening["self_controls_passed"] = all(runs[key]["versus_original_path"].get("equal", False)
                        for key in ("sequential_self_control", "block_self_control"))
                    screening["winner_changed"] = any(
                        runs[key]["scores"]["argmax_id"] != report["logits"][path]["argmax_id"]
                        for key, path in (("sequential_with_block_values", "sequential"), ("block_with_sequential_values", "block")))
                    print(f"{name}: self-controls={screening['self_controls_passed']}, "
                          f"sequential swap winner={runs['sequential_with_block_values']['scores']['argmax_text']!r}, "
                          f"block swap winner={runs['block_with_sequential_values']['scores']['argmax_text']!r}", flush=True)
                    save()
                    if not screening["self_controls_passed"]:
                        report["status"] = "causal_control_failed"
                        return
                    if screening["winner_changed"]:
                        selected = name
                        report["causal_contributor_found"] = name
                        break
            if selected:
                report["selected_probe_module"] = selected
                print("Independent operator probe:", selected, flush=True)
                report["operator_probe"] = probe(dict(model.named_modules())[selected], bat.inputs[selected][0], count, device)
                print(json.dumps(report["operator_probe"], indent=2), flush=True)
            report["status"] = "complete"
        else:
            report["status"] = "control_failed" if not report["controls_passed"] else "winner_difference_not_reproduced"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
