"""The command-line entry point: argument validation and .env handling."""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bow_main", ROOT / "code" / "main.py")
bow_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bow_main)


def test_dotenv_sets_missing_variables_and_never_overrides_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nBOW_TEST_A=from-file\nBOW_TEST_B='quoted'\n"
                   "BOW_TEST_C=from-file\nBOW_TEST_EMPTY=\nnot a pair\n", encoding="utf-8")
    monkeypatch.delenv("BOW_TEST_A", raising=False)
    monkeypatch.delenv("BOW_TEST_B", raising=False)
    monkeypatch.delenv("BOW_TEST_EMPTY", raising=False)
    monkeypatch.setenv("BOW_TEST_C", "from-shell")
    bow_main.load_dotenv(env)
    assert os.environ["BOW_TEST_A"] == "from-file"
    assert os.environ["BOW_TEST_B"] == "quoted"
    assert os.environ["BOW_TEST_C"] == "from-shell"
    assert "BOW_TEST_EMPTY" not in os.environ


def test_missing_dotenv_is_not_an_error(tmp_path):
    bow_main.load_dotenv(tmp_path / "absent.env")


@pytest.mark.parametrize("argv", [
    ["--backend", "bogus"],
    ["--estimator", "bogus"],
    ["--requests-file", "does_not_exist.csv"],
])
def test_bad_arguments_are_rejected_before_any_work(argv):
    with pytest.raises(SystemExit) as exc:
        bow_main.main(argv)
    assert exc.value.code == 2


def test_cloud_backend_without_a_key_fails_with_a_clear_message(monkeypatch, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(bow_main, "load_dotenv", lambda path: None)
    with pytest.raises(SystemExit):
        bow_main.main(["--backend", "cloud"])
    assert "LLM_API_KEY is not set" in capsys.readouterr().err
