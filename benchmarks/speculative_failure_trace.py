"""Trace later-round speculative failures from a saved benchmark, without timing.

For user execution only. Runs selected prompts, stops speculation at its first
divergence, saves an actual verification/cache replay artifact, and compares
execution shapes and cache histories. Does not assume a numerical/backend cause.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from fastinfer.speculative_cached import cached_decode, crop_to, _forward
from speculative_divergence import describe_logits, compare_logits


class DivergenceFound(Exception):
    pass


def assert_prefix(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: token history/row alignment invariant failed")


def pack_cache(cache, length=None):
    size = cache.get_seq_length() if length is None else length
    result = []
    for layer in cache.layers:
        entry = {}
        for kind in ("keys", "values"):
            value = getattr(layer, kind)
            selected = value[..., :size, :]
            entry[kind] = selected.detach().cpu().clone()
            entry[kind + "_stride"] = list(selected.stride())
        result.append(entry)
    return result


def cache_equal(actual, expected):
    return len(actual) == len(expected) and all(
        torch.equal(a[kind], b[kind]) for a, b in zip(actual, expected) for kind in ("keys", "values"))


@torch.inference_mode()
def restore_cache(model, packed, device):
    cache = DynamicCache(config=model.config)
    for index, entry in enumerate(packed):
        restored = []
        for kind in ("keys", "values"):
            value = entry[kind]
            # Retain the strided view layout left by cropping, as well as values.
            tensor = torch.empty_strided(value.shape, entry[kind + "_stride"], dtype=value.dtype, device=device)
            tensor.copy_(value.to(device))
            restored.append(tensor)
        cache.update(restored[0], restored[1], index)
    return cache


class Recorder:
    def __init__(self, prompt, expected=None):
        self.prompt = list(prompt)
        self.expected = expected
        self.lineage = {"target": [], "draft": []}
        self.pending = {}
        self.records = []
        self.schedule = []
        self.generated = []
        self.vectors = []
        self.current = None
        self.snapshot = None
        self.mismatch = None
        self.crop_snapshot = None

    def __call__(self, event, payload):
        if event in ("target_before", "draft_before"):
            provider = event.split("_")[0]
            prefix = self.lineage[provider]
            cached = payload["cache"].get_seq_length()
            if cached != len(prefix):
                raise RuntimeError(f"{provider} cache length disagrees with independently tracked tokens")
            ids = payload["input_ids"][0].tolist()
            history = payload["history"][0].tolist()
            if provider == "target" and payload["phase"] == "verification":
                assert_prefix(prefix, history[:-1], "target verification prefix")
                assert_prefix(ids, history[-1:] + payload["proposals"], "target verification input")
            else:
                assert_prefix(prefix + ids, history, f"{provider} forward")
            entry = {"event": event, "cache_length_before": cached, "input_ids": ids,
                     "expected_positions": list(range(cached, cached + len(ids))),
                     "phase": payload.get("phase", "draft"), "lineage_passed": True}
            self.records.append(entry)
            self.pending[provider] = (ids, entry)
            if provider == "target":
                self.current = entry
                entry["prefix_token_ids"] = list(prefix)
                entry["generated_before"] = list(self.generated)
                entry["proposals"] = payload.get("proposals", [])
                entry["emissions"] = []
                if payload["phase"] == "verification":
                    self.snapshot = pack_cache(payload["cache"])
            return
        if event in ("target_after", "draft_after"):
            provider = event.split("_")[0]
            ids, entry = self.pending[provider]
            self.lineage[provider].extend(ids)
            actual = payload["cache"].get_seq_length()
            if actual != len(self.lineage[provider]):
                raise RuntimeError(f"{provider} cache append length invariant failed")
            entry["cache_length_after"] = actual
            if provider == "target":
                self.schedule.append({"kind": "forward", "input_ids": ids,
                                      "keep": 0 if entry["phase"] == "verification" else 1,
                                      "prefix_length": entry["cache_length_before"]})
            return
        if event == "emit":
            index, token = payload["index"], payload["token"]
            if index != len(self.generated):
                raise RuntimeError("emitted token index invariant failed")
            phase, row = self.current["phase"], payload["row"]
            if phase == "prefill":
                context = self.prompt
            else:
                context = self.current["prefix_token_ids"] + self.current["input_ids"][:row + 1]
            assert_prefix(context, self.prompt + self.generated, "emitted logit causal context")
            self.current["emissions"].append({"generated_index": index, "token": token,
                                               "row": row, "source": payload.get("source", phase)})
            self.generated.append(token)
            vector = payload["logits"].detach().cpu().clone()
            if self.expected is None:
                self.vectors.append(vector)
            elif index >= len(self.expected) or token != self.expected[index]:
                self.mismatch = {"generated_index": index, "generated_token_1_based": index + 1,
                                 "baseline_token": self.expected[index] if index < len(self.expected) else None,
                                 "speculative_token": token, "row": row,
                                 "current_call": self.current, "actual_vector": vector}
                raise DivergenceFound()
            return
        if event == "crop_before":
            self.crop_snapshot = {}
            entry = {"event": "crop", "target_length": payload["target_length"],
                     "draft_length": payload["draft_length"], "prefix_values_preserved": {}}
            self.records.append(entry)
            for provider in ("target", "draft"):
                length = payload[provider + "_length"]
                self.crop_snapshot[provider] = pack_cache(payload[provider + "_cache"], length)
                self.lineage[provider] = self.lineage[provider][:length]
                assert_prefix(self.lineage[provider], payload["history"][0].tolist()[:length], f"{provider} crop lineage")
            self.schedule.append({"kind": "crop", "length": payload["target_length"]})
            return
        if event == "crop_after":
            entry = self.records[-1]
            for provider in ("target", "draft"):
                cache = payload[provider + "_cache"]
                if cache.get_seq_length() != len(self.lineage[provider]):
                    raise RuntimeError(f"{provider} crop length invariant failed")
                equal = cache_equal(pack_cache(cache), self.crop_snapshot[provider])
                entry["prefix_values_preserved"][provider] = equal
                if not equal:
                    raise RuntimeError(f"{provider} cropping changed retained cache values")
            self.crop_snapshot = None

    def rotary_hook(self, module, args, kwargs):
        if self.current is None:
            return
        positions = kwargs.get("position_ids")
        if positions is None and len(args) > 1:
            positions = args[1]
        if positions is not None:
            actual = positions[0].tolist()
            self.current["actual_positions"] = actual
            assert_prefix(actual, self.current["expected_positions"], "target rotary positions")


@torch.inference_mode()
def probes(model, tokenizer, recorder, baseline_vector, prompt_length, repeats):
    mismatch = recorder.mismatch
    call = mismatch["current_call"]
    if call["phase"] != "verification" or mismatch["baseline_token"] is None:
        return {"skipped": True, "reason": "first mismatch is outside a reconstructable verification call"}
    device = next(model.parameters()).device
    watched = (mismatch["baseline_token"], mismatch["speculative_token"])
    block = torch.tensor([call["input_ids"]], dtype=torch.long, device=device)
    prefix = call["prefix_token_ids"]
    row = mismatch["row"]
    shared = prefix + call["input_ids"][:row + 1]
    assert_prefix(shared, recorder.prompt + recorder.generated[:-1], "probe common history")
    result = {"actual_verification": describe_logits(mismatch["actual_vector"], tokenizer, watched)[0],
              "baseline_sequential": describe_logits(baseline_vector, tokenizer, watched)[0], "paths": {}}
    vectors = {}
    for path in ("same_cache_block", "same_cache_serial_inputs", "same_schedule_rebuilt_block",
                 "fresh_sequential_history_block", "fresh_prefill_history_block", "uncached_common_history"):
        result["paths"][path] = []
        vectors[path] = []
        for _ in range(repeats):
            metadata = {}
            if path.startswith("same_cache"):
                cache = restore_cache(model, recorder.snapshot, device)
                metadata["restored_cache_matches_snapshot"] = cache_equal(pack_cache(cache), recorder.snapshot)
            elif path == "same_schedule_rebuilt_block":
                cache = DynamicCache(config=model.config)
                # The final schedule entry is the divergent verification call;
                # replay preceding calls and crop operations exactly as traced.
                for step in recorder.schedule[:-1]:
                    if step["kind"] == "crop":
                        crop_to(cache, step["length"])
                    else:
                        if cache.get_seq_length() != step["prefix_length"]:
                            raise RuntimeError("schedule rebuild prefix length invariant failed")
                        _forward(model, block.new_tensor([step["input_ids"]]), cache, step["keep"])
                metadata["rebuilt_cache_matches_snapshot"] = cache_equal(pack_cache(cache), recorder.snapshot)
            elif path == "fresh_sequential_history_block":
                cache = DynamicCache(config=model.config)
                _forward(model, block.new_tensor([prefix[:prompt_length]]), cache, keep=1)
                for token in prefix[prompt_length:]:
                    _forward(model, block.new_tensor([[token]]), cache, keep=1)
            elif path == "fresh_prefill_history_block":
                cache = DynamicCache(config=model.config)
                _forward(model, block.new_tensor([prefix]), cache, keep=1)

            if path == "uncached_common_history":
                vector = model(input_ids=block.new_tensor([shared]), use_cache=False,
                               logits_to_keep=1, return_dict=True).logits[0, -1]
            elif path == "same_cache_serial_inputs":
                for token in call["input_ids"][:row + 1]:
                    logits = _forward(model, block.new_tensor([[token]]), cache, keep=1)
                vector = logits[0, -1]
            else:
                if cache.get_seq_length() != len(prefix):
                    raise RuntimeError("probe cache length invariant failed")
                vector = _forward(model, block, cache)[0, row]
            description, values = describe_logits(vector, tokenizer, watched)
            result["paths"][path].append({**metadata, **description})
            vectors[path].append(values)
        first = result["paths"][path][0]
        print(f"  {path}: winner={first['argmax_text']!r}, "
              f"baseline-minus-speculative={first['baseline_minus_speculative_score']:.8f}", flush=True)
    actual = mismatch["actual_vector"].float()
    result["versus_actual_verification"] = {path: compare_logits(values[0], actual) for path, values in vectors.items()}
    result["repeat_stability"] = {path: [compare_logits(value, values[0]) for value in values[1:]]
                                   for path, values in vectors.items()}
    result["findings"] = {
        "captured_cache_block_reproduces_actual_logits": result["versus_actual_verification"]["same_cache_block"]["different_logit_values"] == 0,
        "same_cache_serial_vs_block_winner_differs": int(vectors["same_cache_block"][0].argmax()) != int(vectors["same_cache_serial_inputs"][0].argmax()),
        "all_probe_repeats_exact": all(check["different_logit_values"] == 0
            for checks in result["repeat_stability"].values() for check in checks),
        "same_schedule_rebuild_reproduces_cache_values": all(
            entry.get("rebuilt_cache_matches_snapshot", False) for entry in result["paths"]["same_schedule_rebuilt_block"])}
    print("  Findings:", json.dumps(result["findings"]), flush=True)
    result["note"] = ("Same-cache shape controls isolate within-call shape sensitivity; fresh-history "
                      "controls also change prior arithmetic. A fresh-cache mismatch alone does not prove a cache bug. "
                      "Schedule, lineage, positions, retained-prefix checks, scores and repeats must be read together.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case-ids", type=int, nargs="+", default=[115])
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two probe repeats are required")
    if len(set(args.case_ids)) != len(args.case_ids):
        parser.error("case IDs must be unique")
    saved = json.loads(args.report.read_text())
    selected = [next((case for case in saved["cases"] if case["id"] == identifier), None) for identifier in args.case_ids]
    if any(case is None for case in selected):
        parser.error("case ID not found in report")
    if not saved.get("draft_snapshot"):
        parser.error("report must contain target and draft snapshots")
    device = torch.device(args.device or saved["environment"]["device"])
    dtype = getattr(torch, saved["environment"]["dtype"].split(".")[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or args.report.with_name(f"speculative_failure_trace_{stamp}.json")
    if destination.exists():
        parser.error("output exists; select a new filename")
    report = {"status": "running", "source_report": str(args.report.resolve()),
              "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), ROOT / "speculative_cached.py",
                                              ROOT / "benchmarks/speculative_divergence.py")},
              "environment": {"device": str(device), "dtype": str(dtype), "attention": "sdpa",
                              "torch": torch.__version__, "transformers": transformers.__version__,
                              "mps_flags": {key: os.environ.get(key) for key in (
                                  "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK")}},
              "target_snapshot": saved["target_snapshot"], "draft_snapshot": saved["draft_snapshot"],
              "scope": "Selected-prompt accuracy trace, no performance claim. Observer-enabled "
                       "generation stops at first mismatch. Cache history is tracked independently; "
                       "KV values and layouts are captured for same-state replay. Both models resident.",
              "cases": []}

    def save():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(saved["target_snapshot"], local_files_only=True)
        draft_tokenizer = AutoTokenizer.from_pretrained(saved["draft_snapshot"], local_files_only=True)
        if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
            raise ValueError("draft/target token-ID mappings differ")
        print(f"Loading original target and draft on {device} as {dtype}; selected-case diagnostic only", flush=True)
        target = AutoModelForCausalLM.from_pretrained(saved["target_snapshot"], local_files_only=True,
                    dtype=dtype, attn_implementation="sdpa").to(device).eval()
        draft = AutoModelForCausalLM.from_pretrained(saved["draft_snapshot"], local_files_only=True,
                    dtype=dtype, attn_implementation="sdpa").to(device).eval()
        eos = target.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        for original in selected:
            case = {"id": original["id"], "category": original["category"],
                    "prompt": original["prompt"], "status": "running"}
            report["cases"].append(case)
            text = original["prompt"] if saved["settings"]["prompt_format"] == "raw" else f"Question: {original['prompt']}\nAnswer:"
            prompt = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            if prompt.shape[1] != original["input_tokens"]:
                raise ValueError("prompt reconstruction length differs from original report")
            options = {"max_new_tokens": saved["settings"]["max_new_tokens"], "eos_token_ids": eos}
            baseline = Recorder(prompt[0].tolist())
            case["baseline_trace"] = baseline.records
            handle = target.model.rotary_emb.register_forward_pre_hook(baseline.rotary_hook, with_kwargs=True)
            try:
                output, _ = cached_decode(target, prompt, observer=baseline, **options)
            finally:
                handle.remove()
            repeat, _ = cached_decode(target, prompt, **options)
            case["baseline_repeat_matches"] = repeat.generated == output.generated
            case["baseline_matches_saved"] = output.generated == original["validation"]["baseline"]["generated"]
            case["baseline_generated"] = output.generated
            if not case["baseline_repeat_matches"]:
                case["status"] = "baseline_unstable"
                save()
                print(f"{case['id']}: baseline unstable; skipping attribution", flush=True)
                continue
            recorder = Recorder(prompt[0].tolist(), expected=output.generated)
            case["trace"] = recorder.records
            handle = target.model.rotary_emb.register_forward_pre_hook(recorder.rotary_hook, with_kwargs=True)
            try:
                candidate, _ = cached_decode(target, prompt, draft=draft,
                    gamma=saved["settings"]["gamma"], observer=recorder, **options)
            except DivergenceFound:
                mismatch = recorder.mismatch
                case["mismatch"] = {key: value for key, value in mismatch.items() if key != "actual_vector"}
                case["speculative_generated_until_stop"] = recorder.generated
            finally:
                handle.remove()
            if recorder.mismatch is None:
                case["status"] = "no_divergence_reproduced"
                case["speculative_generated"] = candidate.generated
                print(f"{case['id']}: no token divergence reproduced in this trace", flush=True)
                save()
                continue
            call = recorder.mismatch["current_call"]
            token_index = recorder.mismatch["generated_index"]
            forwards = [entry for entry in recorder.records if entry["event"] == "target_before"]
            crops = [entry for entry in recorder.records if entry["event"] == "crop"]
            case["structural_checks"] = {
                "forward_lineage_checks_passed": all(entry.get("lineage_passed", False)
                    for entry in recorder.records if entry["event"] in ("target_before", "draft_before")),
                "target_position_checks_observed": sum("actual_positions" in entry for entry in forwards),
                "target_calls_traced": len(forwards),
                "all_target_position_checks_available": all("actual_positions" in entry for entry in forwards),
                "crop_events_checked": len(crops),
                "retained_cache_values_preserved": all(
                    all(entry["prefix_values_preserved"].get(provider, False) for provider in ("target", "draft"))
                    for entry in crops)}
            print(f"{case['id']}: first mismatch at generated token {token_index + 1}; "
                  f"phase={call['phase']}, verification row={recorder.mismatch['row']}, "
                  f"cache prefix={call['cache_length_before']}", flush=True)
            print("  Structural checks:", json.dumps(case["structural_checks"]), flush=True)
            watched = (recorder.mismatch["baseline_token"], recorder.mismatch["speculative_token"])
            baseline_vector = baseline.vectors[token_index] if token_index < len(baseline.vectors) else None
            if watched[0] is not None:
                baseline_scores, _ = describe_logits(baseline_vector, tokenizer, watched)
                actual_scores, _ = describe_logits(recorder.mismatch["actual_vector"], tokenizer, watched)
                print(f"  baseline: winner={baseline_scores['argmax_text']!r}, "
                      f"baseline-minus-speculative={baseline_scores['baseline_minus_speculative_score']:.8f}", flush=True)
                print(f"  actual verification: winner={actual_scores['argmax_text']!r}, "
                      f"baseline-minus-speculative={actual_scores['baseline_minus_speculative_score']:.8f}", flush=True)
            if call["phase"] == "verification":
                artifact = destination.with_name(destination.stem + f"_case{case['id']}_replay.pt")
                # Plain tensors/primitives only, suitable for weights_only=True loading.
                torch.save({"cache": recorder.snapshot, "prefix_token_ids": call["prefix_token_ids"],
                            "verification_input_ids": call["input_ids"], "row": recorder.mismatch["row"],
                            "prompt_token_ids": prompt[0].tolist(), "target_snapshot": saved["target_snapshot"],
                            "baseline_logits": baseline_vector,
                            "actual_verification_logits": recorder.mismatch["actual_vector"],
                            "baseline_token": recorder.mismatch["baseline_token"],
                            "speculative_token": recorder.mismatch["speculative_token"]}, artifact)
                case["replay_artifact"] = str(artifact)
                case["replay_artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            save()
            case["probes"] = probes(target, tokenizer, recorder, baseline_vector, prompt.shape[1], args.repeats)
            case["status"] = "divergence_traced"
            save()
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
