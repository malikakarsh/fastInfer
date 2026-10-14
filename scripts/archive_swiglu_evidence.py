"""Bundle SwiGLU reports, generated code and current sources with member hashes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = ROOT / "results/raw"
    reports = sorted({*raw.glob("swiglu*.json"), *raw.glob("qwen_real_swiglu*.json"),
                      *raw.glob("qwen_swiglu_decode*.json")})
    if not any(not path.name.endswith("_console_summary.json")
               and path.name.startswith("swiglu_") for path in reports):
        parser.error("No full SwiGLU JSON reports found; run the GPU benchmarks first")
    generated = sorted(path for directory in raw.glob("swiglu_generated_*")
                       if directory.is_dir() for path in directory.iterdir()
                       if path.suffix in (".py", ".json"))
    sources = sorted({*ROOT.glob("kernels/*swiglu*.py"),
                      *ROOT.glob("benchmarks/*swiglu*.py"),
                      *ROOT.glob("scripts/*swiglu*.py"),
                      ROOT / "benchmarks/softmax_compare.py",
                      ROOT / "benchmarks/matmul_compare.py",
                      ROOT / "scripts/rmsnorm_compiled_inspect.py", ROOT / "decode_swiglu.py"})
    files = sorted(set(reports + generated + sources))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else raw
    destination = args.output or base / f"swiglu_evidence_{stamp}.zip"
    if destination.exists():
        parser.error("Output already exists; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": stamp, "scope": "Preserve full reports including any failure records. "
                "Current sources are snapshots; older report hashes can refer to older source versions.",
                "members": []}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            content = path.read_bytes()
            if path.suffix == ".json":
                json.loads(content)
            name = str(path.relative_to(ROOT))
            archive.writestr(name, content)
            manifest["members"].append({"path": name, "bytes": len(content),
                                         "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    print(f"Archived {len(reports)} JSON reports, {len(generated)} generated files, "
          f"{len(sources)} source files: {destination}")
    print("SHA256:", hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
