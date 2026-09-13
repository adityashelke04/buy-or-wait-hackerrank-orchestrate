"""The cloud backend, exercised against a fake client - no network, no key."""
from types import SimpleNamespace

import pytest

from buyorwait.evidence.cloud_provider import CloudProvider


class FakeClient:
    def __init__(self, content=None, raise_error=False):
        self.content, self.raise_error, self.calls = content, raise_error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_error:
            raise ConnectionError("network down")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30))


def test_returns_parsed_json():
    c = CloudProvider(client=FakeClient('{"amendments": []}'))
    assert c.complete_json("p", {}) == {"amendments": []}


def test_requests_deterministic_json_output():
    fake = FakeClient('{"amendments": []}')
    CloudProvider(client=fake).complete_json("p", {})
    assert fake.calls[0]["temperature"] == 0
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_network_failure_fails_safe_to_none():
    assert CloudProvider(client=FakeClient(raise_error=True)).complete_json("p", {}) is None


def test_non_json_reply_fails_safe_to_none():
    assert CloudProvider(client=FakeClient("sure! here you go")).complete_json("p", {}) is None


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        CloudProvider()
