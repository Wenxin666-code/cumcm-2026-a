"""Verify the immutable support package and light entry points; never solve a PDE."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission"


def main() -> int:
    manifest = json.loads((PACKAGE / "MANIFEST.sha256.json").read_text(encoding="utf-8"))
    recorded = manifest["files"]
    for name, expected in recorded.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {name}")
        path = PACKAGE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed archived file: {name}")
    actual = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()}
    if actual != set(recorded) | {"MANIFEST.sha256.json"}:
        raise ValueError(f"Unexpected file set: {sorted(actual ^ (set(recorded) | {'MANIFEST.sha256.json'}))}")
    sources = sorted((PACKAGE / "code").rglob("*.py"))
    for source in sources:
        ast.parse(source.read_bytes(), filename=str(source))
    env = {key: value for key, value in os.environ.items() if not key.startswith("CUMCM_")}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    for question in range(1, 5):
        for mode in ("check", "plan"):
            command = [sys.executable, "-B", str(PACKAGE / "code" / f"问题{question}.py"), "--mode", mode]
            result = subprocess.run(command, cwd=PACKAGE, env=env, capture_output=True,
                                    text=True, encoding="utf-8", timeout=120, check=True)
            report = json.loads(result.stdout)
            if report.get("PDE_started") is not False:
                raise ValueError(f"Unexpected solver state: question {question}, mode {mode}")
            if mode == "check" and report.get("selected_results_passed") is not True:
                raise ValueError(f"Result check failed: question {question}")
            if mode == "plan" and not report.get("tasks"):
                raise ValueError(f"Empty plan: question {question}")
            print(f"Question {question}: {mode} passed")
    print(f"Archive verified: {len(recorded)} manifest entries, {len(sources)} Python sources; no solver run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
