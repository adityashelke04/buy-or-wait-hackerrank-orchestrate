"""Build code.zip for submission, and fail loudly if it approaches 50 MB."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 45                                   # guard trips well before 50 MB

FOLDERS = ["code", "tests", "evaluation", "dataset", "scripts", "docs"]
FILES = ["README.md", "requirements.txt", ".env.example", "pytest.ini",
         "output.csv", "problem_statement.md", "AGENTS.md", ".gitignore"]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}


def _keep(rel: Path) -> bool:
    if any(part in SKIP_PARTS for part in rel.parts):
        return False
    if rel.suffix in {".pyc", ".pyo"}:
        return False
    if rel.name == ".env" or (rel.name.startswith(".env.") and rel.name != ".env.example"):
        return False                            # never ship a secret
    if rel.name in {"log.txt", "llm_cache.json", "code.zip", "output_samples.csv"}:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "code.zip")
    out = parser.parse_args().out
    out.unlink(missing_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in FILES:
            if (ROOT / name).exists():
                z.write(ROOT / name, name)
        for folder in FOLDERS:
            for src in sorted((ROOT / folder).rglob("*")):
                rel = src.relative_to(ROOT)
                if src.is_file() and _keep(rel):
                    z.write(src, rel.as_posix())

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"{out} -> {size_mb:.2f} MB")
    if size_mb >= LIMIT_MB:
        raise SystemExit(f"code.zip is {size_mb:.1f} MB; the limit is 50 MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
