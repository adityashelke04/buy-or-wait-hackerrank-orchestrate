import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_package_builds_complete_and_under_the_size_limit(tmp_path):
    target = tmp_path / "code.zip"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "package.py"),
                    "--out", str(target)], check=True, capture_output=True)
    assert target.exists()
    size_mb = target.stat().st_size / (1024 * 1024)
    assert size_mb < 45, f"code.zip is {size_mb:.1f} MB, too close to the 50 MB limit"

    with zipfile.ZipFile(target) as z:
        names = set(z.namelist())
    for required in ["README.md", "code/main.py", "evaluation/usage_report.md",
                     "requirements.txt", ".env.example", "evaluation/score.py"]:
        assert required in names, f"{required} missing from code.zip"


def test_package_never_ships_secrets_caches_or_git(tmp_path):
    target = tmp_path / "code.zip"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "package.py"),
                    "--out", str(target)], check=True, capture_output=True)
    with zipfile.ZipFile(target) as z:
        names = z.namelist()
    assert not any(n.startswith(".git/") for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert not any(n.split("/")[-1] == ".env" for n in names)
    assert "log.txt" not in names
