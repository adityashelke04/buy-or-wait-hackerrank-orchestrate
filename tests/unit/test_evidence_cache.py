from pathlib import Path

from buyorwait.evidence.cache import Cache


def test_key_is_stable_for_identical_inputs():
    assert Cache.key("model", "prompt") == Cache.key("model", "prompt")


def test_key_differs_when_any_part_differs():
    assert Cache.key("model", "a") != Cache.key("model", "b")
    assert Cache.key("model-a", "p") != Cache.key("model-b", "p")


def test_key_is_not_confused_by_concatenation():
    """('ab','c') and ('a','bc') must not collide."""
    assert Cache.key("ab", "c") != Cache.key("a", "bc")


def test_miss_then_hit(tmp_path):
    c = Cache(tmp_path / "cache.json")
    k = Cache.key("m", "p")
    assert c.get(k) is None
    assert c.misses == 1
    c.put(k, {"amendments": []})
    assert c.get(k) == {"amendments": []}
    assert c.hits == 1


def test_survives_a_round_trip_to_disk(tmp_path):
    path = tmp_path / "cache.json"
    a = Cache(path)
    a.put(Cache.key("m", "p"), {"amendments": [{"kind": "no_change"}]})
    a.flush()

    b = Cache(path)
    assert b.get(Cache.key("m", "p")) == {"amendments": [{"kind": "no_change"}]}


def test_corrupt_cache_file_is_ignored_rather_than_crashing(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json", encoding="utf-8")
    c = Cache(path)
    assert c.get(Cache.key("m", "p")) is None


def test_cached_none_is_distinguishable_from_a_miss(tmp_path):
    """A provider that legitimately returned nothing must not be retried
    forever - the null result is itself worth caching."""
    c = Cache(tmp_path / "cache.json")
    k = Cache.key("m", "p")
    c.put(k, None)
    assert c.has(k)
    assert c.get(k) is None
