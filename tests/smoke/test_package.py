import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("package", ROOT / "scripts" / "package.py")
package = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package)


def _build(tmp_path) -> Path:
    target = tmp_path / "code.zip"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "package.py"),
                    "--out", str(target)], check=True, capture_output=True)
    return target


def test_package_builds_complete_and_under_the_size_limit(tmp_path):
    target = _build(tmp_path)
    size_mb = target.stat().st_size / (1024 * 1024)
    assert size_mb < 45, f"code.zip is {size_mb:.1f} MB, too close to the 50 MB limit"

    with zipfile.ZipFile(target) as z:
        names = set(z.namelist())
    for required in ["README.md", "code/main.py", "evaluation/usage_report.md",
                     "code/evaluation/usage_report.md", "requirements.txt",
                     ".env.example", "evaluation/score.py", "output.csv",
                     "dataset/requests.csv"]:
        assert required in names, f"{required} missing from code.zip"


def test_package_never_ships_secrets_caches_or_git(tmp_path):
    with zipfile.ZipFile(_build(tmp_path)) as z:
        names = z.namelist()
    assert not any(n.startswith(".git/") for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert not any(n.split("/")[-1] == ".env" for n in names)
    assert not any(n.split("/")[-1] in {"log.txt", "llm_cache.json"} for n in names)


def test_every_file_that_would_be_packaged_is_free_of_sensitive_content():
    assert package.scan(package.collect()) == []


# Each fixture is assembled from fragments joined at run time, so this file never
# holds a literal the scanner would (rightly) refuse to package.
_ = "".join


@pytest.mark.parametrize("text", [
    _(["key = AI", "zaSyA1234567890abcdefghijklmnopqrstuv"]),
    _(["sk-", "ant-api03-abcdefghijklmnopqrstuvwxyz"]),
    _(["gh", "p_abcdefghijklmnopqrstuvwxyz0123456789"]),
    _(["-----BEGIN RSA ", "PRIVATE KEY-----"]),
    _(["endpoint http://192.", "168.1.10:8080/api"]),
    _(["host 127.", "0.0.1"]),
    _(["contact someone", "@", "example.com"]),
    _(["cd C:", "\\", "Users\\someone\\project"]),
    _(["cd D:", "/Orca/projects/x"]),
    _(["/ho", "me/someone/project"]),
    _(["LLM_API", "_KEY=abcd1234xyz"]),
])
def test_scanner_catches_sensitive_content(text):
    assert package.find_sensitive("f", text), text


@pytest.mark.parametrize("text", [
    "LLM_API_KEY=",
    "Co-Authored-By: Claude <noreply@anthropic.com>",
    "IDR 1.234.567,89 and version 3.9.2",
    "amount 17229139.2 on 2026-09-13",
    "python code/main.py --out output.csv",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
])
def test_scanner_ignores_ordinary_content(text):
    assert package.find_sensitive("f", text) == [], text
