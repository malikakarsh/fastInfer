"""Four-way cache-history/execution replay, with preselected passing controls.

Accuracy diagnostic only. --plan-only reads saved JSON without loading models.
Otherwise the user runs selected baseline/speculative generations and replays.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "benchmarks"))

import torch
import transformers
from transformers import AutoTokenizer, DynamicCache

from fastinfer.speculative_cached import cached_decode, _forward
from speculative_failure_trace import Recorder, DivergenceFound, pack_cache, restore_cache, cache_equal, assert_prefix
from speculative_divergence import describe_logits, compare_logits
from speculative_mt_bench import load_target, package_version


def first_difference(case):
    validation = case.get("validation", {})
    a = validation.get("baseline", {}).get("generated", [])
    b = validation.get("speculative", {}).get("generated", [])
    return next((index for index, (x, y) in enumerate(zip(a, b)) if x != y), None)


def spread_indices(size, count):
    count = min(size, count)
    if count == 0:
        return []
    if count == 1:
        return [0]
    return [i * (size - 1) // (count - 1) for i in range(count)]


def make_plan(saved, failure_count, control_count, all_cases=False):
    """Freeze selection from saved outcomes, never from new probe results."""
    identifiers = [case["id"] for case in saved["cases"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("source report contains duplicate case IDs")
    if all_cases:
        failure_count = control_count = len(saved["cases"])
    failures, controls, excluded = [], [], []
    for case in saved["cases"]:
        baseline = case.get("validation", {}).get("baseline", {}).get("generated", [])
        if not case.get("baseline_repeat_matches", False):
            excluded.append({"id": case["id"], "reason": "saved baseline unstable or unavailable"})
        elif case.get("correctness_passed") and len(baseline) > 1:
            controls.append(case)
        elif case.get("correctness_passed"):
            excluded.append({"id": case["id"], "reason": "passing output too short for a post-prefill decision"})
        else:
            index = first_difference(case)
            if index is None or index == 0:
                excluded.append({"id": case["id"], "reason": "length-only, missing output, or prefill mismatch"})
            else:
                failures.append((index, case))
    failures.sort(key=lambda pair: (pair[0], pair[1]["id"]))
    chosen = [failures[i] for i in spread_indices(len(failures), failure_count)]
    if not chosen:
        raise ValueError("no eligible saved failures for early/middle/late selection")
    entries = [{"id": case["id"], "category": case["category"], "group": "saved_failure",
                "decision_token_1_based": index + 1,
                "selection": "evenly spaced ranks of first mismatch positions"} for index, case in chosen]
    available = list(controls)
    for i in range(min(control_count, len(available))):
        index, failure = chosen[i % len(chosen)]
        eligible = [c for c in available if len(c["validation"]["baseline"]["generated"]) > index]
        pool = eligible or available
        control = min(pool, key=lambda c: (
            c["category"] != failure["category"],
            abs(c["input_tokens"] - failure["input_tokens"]), c["id"]))
        available.remove(control)
        point = min(index, len(control["validation"]["baseline"]["generated"]) - 1)
        entries.append({"id": control["id"], "category": control["category"], "group": "saved_pass_control",
                        "decision_token_1_based": point + 1, "matched_failure_id": failure["id"],
                        "same_decision_position": point == index,
                        "selection": "available passing prompt; prefer matching position, category, prompt length"})
    if all_cases and {entry["id"] for entry in entries} != set(identifiers):
        missing = sorted(set(identifiers) - {entry["id"] for entry in entries})
        raise ValueError(f"cannot replay every source case; unsupported IDs: {missing}. See source outcomes.")
    return {"entries": entries, "excluded": excluded, "all_cases": all_cases,
            "source_case_count": len(identifiers), "selected_case_count": len(entries),
            "note": ("All source prompts, one selected decision per prompt; not an all-token audit or predictor evaluation."
                     if all_cases else "Selected cases are a diagnostic sample, not a failure-rate estimate or predictor evaluation.")}


class DecisionRecorder(Recorder):
    """Capture first mismatch OR a predetermined matching control decision."""
    def __init__(self, prompt, expected, decision_index):
        super().__init__(prompt, expected)
        self.decision_index = decision_index

    def __call__(self, event, payload):
        super().__call__(event, payload)
        if event == "emit" and payload["index"] == self.decision_index:
            self.mismatch = {"generated_index": payload["index"],
                             "generated_token_1_based": payload["index"] + 1,
                             "baseline_token": self.expected[payload["index"]],
                             "speculative_token": payload["token"], "row": payload["row"],
                             "current_call": self.current,
                             "actual_vector": payload["logits"].detach().cpu().clone()}
            raise DivergenceFound()


def winner_pattern(winners):
    shape = (winners["ordinary_serial"] != winners["ordinary_block"] or
             winners["speculative_serial"] != winners["speculative_block"])
    history = (winners["ordinary_serial"] != winners["speculative_serial"] or
               winners["ordinary_block"] != winners["speculative_block"])
    if shape and history:
        return "both_factors_or_interaction"
    if shape:
        return "current_execution_winner_effect_only"
    if history:
        return "cache_history_winner_effect_only"
    return "no_winner_effect_observed"


def cache_distance(a, b):
    if len(a) != len(b):
        raise ValueError("cache layer counts differ")
    result = []
    for index, (left, right) in enumerate(zip(a, b)):
        row = {"layer": index}
        for kind in ("keys", "values"):
            x, y = left[kind], right[kind]
            if x.shape != y.shape:
                raise ValueError("cache shapes differ")
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise ValueError("nonfinite cache values")
            error = (x.float() - y.float()).abs()
            row[kind] = {"elements": x.numel(), "different_values": int((x != y).sum()),
                         "max_abs": float(error.max()) if x.numel() else 0.}
        result.append(row)
    return result


def structural_checks(recorder):
    forwards = [r for r in recorder.records if r["event"] == "target_before"]
    crops = [r for r in recorder.records if r["event"] == "crop"]
    return {
        "lineage_passed": all(r.get("lineage_passed", False) for r in recorder.records
                              if r["event"] in ("target_before", "draft_before")),
        "positions_observed_and_correct": bool(forwards) and all(
            r.get("actual_positions") == r["expected_positions"] for r in forwards),
        "retained_cache_values_preserved": all(all(r["prefix_values_preserved"].get(p, False)
            for p in ("target", "draft")) for r in crops),
    }


@torch.inference_mode()
def four_way(model, tokenizer, recorder, baseline_vector, repeats):
    capture = recorder.mismatch
    call = capture["current_call"]
    if call["phase"] != "verification":
        return {"status": "unsupported_decision_phase"}
    device = next(model.parameters()).device
    prefix, inputs, row = call["prefix_token_ids"], call["input_ids"], capture["row"]
    assert_prefix(prefix + inputs[:row + 1], recorder.prompt + recorder.generated[:-1], "four-way context")
    if len(prefix) < len(recorder.prompt):
        raise ValueError("verification prefix shorter than prompt")
    ordinary = DynamicCache(config=model.config)
    _forward(model, torch.tensor([recorder.prompt], device=device), ordinary, keep=1)
    for token in prefix[len(recorder.prompt):]:
        _forward(model, torch.tensor([[token]], device=device), ordinary, keep=1)
    snapshots = {"ordinary": pack_cache(ordinary), "speculative": recorder.snapshot}
    # Keep this oracle-only measurement separate from candidate runtime features.
    distances = cache_distance(snapshots["ordinary"], snapshots["speculative"])
    watched_a = capture["baseline_token"]
    watched_b = capture["speculative_token"]
    if watched_a == watched_b:
        alternative = capture["actual_vector"].float().clone()
        alternative[watched_a] = float("-inf")
        watched_b = int(alternative.argmax())
    watched = (watched_a, watched_b)
    cells, vectors, restore_checks = {}, {}, []
    for history in ("ordinary", "speculative"):
        for execution in ("serial", "block"):
            name = f"{history}_{execution}"
            cells[name], vectors[name] = [], []
            for _ in range(repeats):
                cache = restore_cache(model, snapshots[history], device)
                restore_checks.append(cache_equal(pack_cache(cache), snapshots[history]))
                if execution == "serial":
                    for token in inputs[:row + 1]:
                        logits = _forward(model, torch.tensor([[token]], device=device), cache, keep=1)
                    vector = logits[0, -1]
                else:
                    vector = _forward(model, torch.tensor([inputs], device=device), cache)[0, row]
                description, values = describe_logits(vector, tokenizer, watched)
                cells[name].append(description)
                vectors[name].append(values)
            print(f"  {name}: winner={cells[name][0]['argmax_text']!r}, "
                  f"top-two gap={cells[name][0]['top_two_gap']:.8f}", flush=True)
    reference = baseline_vector.float()
    actual = capture["actual_vector"].float()
    checks = {
        "all_restored_cache_values_match": all(restore_checks),
        "ordinary_serial_reproduces_baseline_logits": torch.equal(vectors["ordinary_serial"][0], reference),
        "speculative_block_reproduces_actual_logits": torch.equal(vectors["speculative_block"][0], actual),
        "all_repeats_exact": all(torch.equal(v, values[0]) for values in vectors.values() for v in values[1:]),
        **structural_checks(recorder),
    }
    winners = {name: entries[0]["argmax_id"] for name, entries in cells.items()}
    return {"status": "complete" if all(checks.values()) else "controls_failed",
            "controls": checks, "cells": cells,
            "logit_comparisons": {
                "shape_with_ordinary_cache": compare_logits(vectors["ordinary_block"][0], vectors["ordinary_serial"][0]),
                "shape_with_speculative_cache": compare_logits(vectors["speculative_block"][0], vectors["speculative_serial"][0]),
                "history_with_serial_execution": compare_logits(vectors["speculative_serial"][0], vectors["ordinary_serial"][0]),
                "history_with_block_execution": compare_logits(vectors["speculative_block"][0], vectors["ordinary_block"][0]),
            },
            "winner_pattern": winner_pattern(winners) if all(checks.values()) else "unclassified_controls_failed",
            "oracle_only_cache_distance": distances,
            "watched_ids": list(watched),
            "note": "A winner pattern describes this decision only. Cache construction changes prior arithmetic; "
                    "cache differences alone are not bugs or deployable predictor features. No FP32 repair is tested."}


def run_with_recorder(target, draft, prompt, recorder, options, gamma):
    handle = target.model.rotary_emb.register_forward_pre_hook(recorder.rotary_hook, with_kwargs=True)
    try:
        return cached_decode(target, prompt, draft=draft, gamma=gamma, observer=recorder, **options)
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--failure-count", type=int)
    parser.add_argument("--control-count", type=int)
    parser.add_argument("--all-cases", action="store_true", help="include every prompt in the source report")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.all_cases and (args.failure_count is not None or args.control_count is not None):
        parser.error("--all-cases cannot be combined with failure/control counts")
    args.failure_count = 3 if args.failure_count is None else args.failure_count
    args.control_count = 3 if args.control_count is None else args.control_count
    if args.failure_count < 1 or args.control_count < 0 or args.repeats < 2:
        parser.error("positive failure count, nonnegative control count, and at least two repeats required")
    saved = json.loads(args.report.read_text())
    try:
        plan = make_plan(saved, args.failure_count, args.control_count, all_cases=args.all_cases)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Selected {plan['selected_case_count']}/{plan['source_case_count']} source prompts", flush=True)
    for entry in plan["entries"]:
        print(f"{entry['id']} {entry['category']}: {entry['group']}, "
              f"planned generated token {entry['decision_token_1_based']}", flush=True)
    if args.plan_only:
        print("Plan only: no models loaded or inference executed.")
        if args.output:
            if args.output.exists():
                parser.error("output exists")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"source_report": str(args.report.resolve()),
                "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(), "plan": plan}, indent=2) + "\n")
        return
    device = torch.device(saved["environment"]["device"])
    dtype = getattr(torch, saved["environment"]["dtype"].split(".")[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or args.report.with_name(f"speculative_history_cases_{stamp}.json")
    if destination.exists():
        parser.error("output exists")
    quantization = saved["settings"].get("target_quantization", "none")
    report = {"status": "running", "source_report": str(args.report.resolve()),
              "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(), "plan": plan,
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                  "bitsandbytes": package_version("bitsandbytes"), "accelerate": package_version("accelerate"),
                  "python": platform.python_version(), "platform": platform.platform(),
                  "device": str(device), "dtype": str(dtype), "attention": "sdpa"},
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                  "benchmarks/speculative_history_cases.py", "benchmarks/speculative_failure_trace.py",
                  "benchmarks/speculative_divergence.py", "benchmarks/speculative_mt_bench.py", "speculative_cached.py")},
              "scope": "Selected accuracy replays only; no timing or predictor performance. Synthetic cache perturbations "
                       "are not used. Passing controls are selected before new inference. No precision repair is tested.",
              "cases": []}

    def save():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)

    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(saved["target_snapshot"], local_files_only=True)
        draft_tokenizer = AutoTokenizer.from_pretrained(saved["draft_snapshot"], local_files_only=True)
        if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
            raise ValueError("target/draft token-ID maps differ")
        print(f"Loading {quantization} target and unquantized draft on {device}; accuracy only", flush=True)
        target, report["target_quantization"] = load_target(saved["target_snapshot"], device, dtype, quantization)
        draft, _ = load_target(saved["draft_snapshot"], device, dtype, "none")
        eos = target.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        options = {"max_new_tokens": saved["settings"]["max_new_tokens"], "eos_token_ids": eos}
        for number, entry in enumerate(plan["entries"], 1):
            print(f"Case {number}/{len(plan['entries'])}: prompt {entry['id']}", flush=True)
            original = next(c for c in saved["cases"] if c["id"] == entry["id"])
            case = {**entry, "status": "running"}
            report["cases"].append(case)
            text = original["prompt"] if saved["settings"]["prompt_format"] == "raw" else f"Question: {original['prompt']}\nAnswer:"
            prompt = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            if prompt.shape[1] != original["input_tokens"]:
                raise ValueError("prompt token length differs from source report")
            baseline = Recorder(prompt[0].tolist())
            output, _ = run_with_recorder(target, None, prompt, baseline, options, saved["settings"]["gamma"])
            repeat, _ = cached_decode(target, prompt, **options)
            case["baseline_repeat_matches"] = repeat.generated == output.generated
            case["baseline_matches_saved"] = output.generated == original["validation"]["baseline"]["generated"]
            if not case["baseline_repeat_matches"] or not case["baseline_matches_saved"]:
                case["status"] = "baseline_changed_or_unstable"
                print(f"{entry['id']}: baseline changed/unstable; attribution skipped", flush=True)
                save()
                continue
            recorder = DecisionRecorder(prompt[0].tolist(), output.generated, entry["decision_token_1_based"] - 1)
            try:
                run_with_recorder(target, draft, prompt, recorder, options, saved["settings"]["gamma"])
            except DivergenceFound:
                pass
            if recorder.mismatch is None:
                case["status"] = "planned_decision_not_reached"
                save()
                continue
            capture = recorder.mismatch
            index, call = capture["generated_index"], capture["current_call"]
            case["observed_decision_token_1_based"] = index + 1
            case["earlier_divergence_than_planned"] = index + 1 < entry["decision_token_1_based"]
            case["tokens_agree_at_observed_decision"] = capture["baseline_token"] == capture["speculative_token"]
            case["verification_context"] = {
                "phase": call["phase"], "prefix_token_ids": call["prefix_token_ids"],
                "input_token_ids": call["input_ids"], "row": capture["row"],
                "baseline_token": capture["baseline_token"], "speculative_token": capture["speculative_token"],
                "emission_source": call["emissions"][-1]["source"],
            }
            case["observations_available_during_speculation"] = {
                "current_top_two_gap": describe_logits(capture["actual_vector"], tokenizer,
                    (capture["baseline_token"], capture["speculative_token"]))[0]["top_two_gap"],
                "current_block_length": len(call["input_ids"]), "verification_row": capture["row"],
                "verification_lengths_so_far": [len(r["input_ids"]) for r in recorder.records
                    if r["event"] == "target_before" and r["phase"] == "verification"],
                "cache_prefix_length": call["cache_length_before"],
            }
            print(f"{entry['id']}: observing generated token {index + 1}; "
                  f"agreement={case['tokens_agree_at_observed_decision']}", flush=True)
            case["four_way"] = four_way(target, tokenizer, recorder, baseline.vectors[index], args.repeats)
            case["status"] = case["four_way"]["status"]
            print(f"  pattern: {case['four_way'].get('winner_pattern', case['status'])}", flush=True)
            save()
        report["status"] = "complete" if all(c["status"] == "complete" for c in report["cases"]) else "needs_review"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(f"Saved {destination}", flush=True)
    if report["status"] != "complete":
        raise SystemExit("Some replay controls need review; inspect the saved report")


if __name__ == "__main__":
    main()
