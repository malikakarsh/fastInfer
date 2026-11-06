"""Audit installed Inductor source and ancestry for PyTorch issue #179561.

Read-only: does not install, patch, or execute downloaded code. Git ancestry
cannot detect an equivalent cherry-picked fix; inspect source and runtime tests
together. GitHub API failures are reported as unknown, not as absence of a fix.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import urllib.request

import torch


FIX = "24c273c46fab7f60c049783b89daffe10d619910"
ISSUE = "https://github.com/pytorch/pytorch/issues/179561"


def check_ancestry(build_sha):
    url = f"https://api.github.com/repos/pytorch/pytorch/compare/{FIX}...{build_sha}"
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "fastinfer-cast-fix-audit",
    })
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
        status = data.get("status")
        merge_base = data.get("merge_base_commit", {}).get("sha")
        contains = merge_base == FIX if status in {"ahead", "behind", "identical", "diverged"} else None
        return {"url": url, "github_status": status,
                "merge_base": merge_base, "contains_original_fix_commit": contains,
                "note": "False does not exclude an equivalent cherry-picked fix."}
    except Exception as exc:
        return {"url": url, "contains_original_fix_commit": None,
                "error": f"{type(exc).__name__}: {exc}"}


def inspect_lowering():
    try:
        from torch._inductor import lowering
        sources = {name: inspect.getsource(getattr(lowering, name))
                   for name in ("to_dtype", "_convert_element_type")}
        # This fingerprint identifies the implementation pattern in v2.13.0,
        # not every possible equivalent implementation in other versions.
        pattern_present = (
            "use_compute_types" in sources["to_dtype"]
            and "emulate_precision_casts" in sources["_convert_element_type"]
            and "use_compute_types" in sources["_convert_element_type"]
        )
        return {"path": inspect.getfile(lowering), "sources": sources,
                "v2_13_explicit_cast_guard_pattern_present": pattern_present,
                "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest()
                                  for name, source in sources.items()}}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip GitHub API ancestry query")
    parser.add_argument("--output", type=Path, default=Path("results/raw/pytorch_cast_fix_audit.json"))
    args = parser.parse_args()
    report = {"torch": torch.__version__, "torch_git": torch.version.git_version,
              "issue": ISSUE, "original_fix_commit": FIX,
              "installed_lowering": inspect_lowering()}
    report["ancestry"] = ({"contains_original_fix_commit": None, "skipped": True}
                          if args.offline else check_ancestry(torch.version.git_version))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
