"""Build code.zip for submission.

Fails closed on two conditions:
  - the archive approaches the 50 MB upload limit
  - any text file in it carries something that must never be submitted: an API
    key or token, a private key, an IP address, an email address, an absolute
    path from a developer machine, or a filled-in secret assignment

Run it:  python scripts/package.py
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 45                                   # guard trips well before 50 MB

FOLDERS = ["code", "tests", "evaluation", "dataset", "scripts", "docs"]
FILES = ["README.md", "requirements.txt", ".env.example", "pytest.ini",
         "output.csv", "problem_statement.md", "AGENTS.md", ".gitignore"]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
SKIP_NAMES = {"log.txt", "llm_cache.json", "code.zip", "output_samples.csv"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".onnx", ".zip"}

# Addresses that identify nobody: git co-author trailers and documentation
# placeholders.
ALLOWED_EMAILS = {"noreply@anthropic.com"}

SENSITIVE: dict[str, re.Pattern[str]] = {
    "API key or token": re.compile(
        r"AIza[0-9A-Za-z_\-]{30,}"              # Google
        r"|sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"   # Anthropic / OpenAI
        r"|gsk_[A-Za-z0-9]{20,}"                # Groq
        r"|gh[pousr]_[A-Za-z0-9]{30,}"          # GitHub
        r"|hf_[A-Za-z0-9]{30,}"                 # Hugging Face
        r"|xox[abprs]-[A-Za-z0-9\-]{10,}"       # Slack
        r"|AKIA[0-9A-Z]{16}"),                  # AWS
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "IP address": re.compile(
        r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"),
    "email address": re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)*\.[A-Za-z]{2,}\b"),
    "absolute local path": re.compile(
        r"\b[A-Za-z]:[\\/](?:Users|Orca|projects|home)\b"
        r"|/(?:home|Users)/[A-Za-z0-9_.\-]+/", re.I),
    "filled-in secret": re.compile(
        r"(?im)^\s*[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)\s*=\s*['\"]?[^\s'\"#]{6,}"),
}


def _keep(rel: Path) -> bool:
    if any(part in SKIP_PARTS for part in rel.parts):
        return False
    if rel.suffix in {".pyc", ".pyo"}:
        return False
    if rel.name == ".env" or (rel.name.startswith(".env.") and rel.name != ".env.example"):
        return False                            # never ship a secret
    return rel.name not in SKIP_NAMES


def collect(root: Path = ROOT) -> list[tuple[Path, str]]:
    """(source file, archive name) for everything that belongs in code.zip."""
    out: list[tuple[Path, str]] = []
    for name in FILES:
        if (root / name).exists():
            out.append((root / name, name))
    for folder in FOLDERS:
        for src in sorted((root / folder).rglob("*")):
            rel = src.relative_to(root)
            if src.is_file() and _keep(rel):
                out.append((src, rel.as_posix()))
    return out


def find_sensitive(name: str, text: str) -> list[str]:
    """Human-readable findings for one file; empty when it is clean."""
    findings: list[str] = []
    for label, pattern in SENSITIVE.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            if label == "email address" and value.lower() in ALLOWED_EMAILS:
                continue
            line = text.count("\n", 0, match.start()) + 1
            shown = value if label in ("IP address", "absolute local path") else value[:6] + "..."
            findings.append(f"{name}:{line}: {label} ({shown})")
    return findings


def scan(files: list[tuple[Path, str]]) -> list[str]:
    findings: list[str] = []
    for src, name in files:
        if src.suffix.lower() in BINARY_SUFFIXES:
            continue
        text = src.read_bytes().decode("utf-8", errors="replace")
        findings.extend(find_sensitive(name, text))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "code.zip")
    out = parser.parse_args().out

    files = collect()
    findings = scan(files)
    if findings:
        raise SystemExit("refusing to build code.zip - sensitive content found:\n  "
                         + "\n  ".join(findings))

    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, name in files:
            z.write(src, name)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"{out.name}: {len(files)} files, {size_mb:.2f} MB, sensitive-content scan clean")
    if size_mb >= LIMIT_MB:
        raise SystemExit(f"code.zip is {size_mb:.1f} MB; the limit is 50 MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
