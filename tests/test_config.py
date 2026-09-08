import hashlib
from pathlib import Path

import pytest

from common.config import ConfigError, load_config_bundle, merge_config_layers, resolve_config_paths


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_bundle_deep_merges_and_freezes_nested_values(tmp_path):
    base = _write(
        tmp_path / "base.yaml",
        "net:\n  host: jetson\n  ports:\n    result: 5556\n    header: 5555\nclasses: [drone, person]\n",
    )
    override = _write(tmp_path / "site.yaml", "net:\n  ports:\n    result: 6556\n")

    bundle = load_config_bundle([base, override], required_sections=("net",))

    assert bundle.data["net"]["host"] == "jetson"
    assert bundle.data["net"]["ports"] == {"result": 6556, "header": 5555}
    assert bundle.data["classes"] == ("drone", "person")
    with pytest.raises(TypeError):
        bundle.data["net"]["host"] = "other"
    mutable = bundle.mutable_copy()
    mutable["net"]["host"] = "other"
    assert bundle.data["net"]["host"] == "jetson"


def test_digest_depends_on_resolved_values_not_yaml_formatting(tmp_path):
    first = _write(tmp_path / "first.yaml", "b: 2\na: {nested: true}\n")
    second = _write(
        tmp_path / "second.yaml",
        "# key order and comments are irrelevant\na:\n  nested: true\nb: 2\n",
    )
    assert load_config_bundle([first]).digest == load_config_bundle([second]).digest


def test_source_hashes_exact_bytes_and_loader_writes_nothing(tmp_path):
    source = _write(tmp_path / "config.yaml", "net: {}\n")
    before = {path.name for path in tmp_path.iterdir()}
    bundle = load_config_bundle([source])

    assert {path.name for path in tmp_path.iterdir()} == before
    assert bundle.sources[0].path == source.resolve()
    assert bundle.sources[0].size == len(source.read_bytes())
    assert bundle.sources[0].sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert bundle.provenance()["config_digest"] == bundle.digest


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- not\n- a mapping\n", "top level"),
        ("port: 1\nport: 2\n", "duplicate configuration key"),
        ("when: 2026-09-08\n", "unsupported YAML value type date"),
        ("value: .nan\n", "finite number"),
        ("1: value\n", "non-string mapping key"),
    ],
)
def test_loader_rejects_ambiguous_or_nonportable_yaml(tmp_path, text, message):
    source = _write(tmp_path / "invalid.yaml", text)
    with pytest.raises(ConfigError, match=message):
        load_config_bundle([source])


def test_loader_rejects_missing_file_and_required_section(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config_bundle([tmp_path / "missing.yaml"])
    source = _write(tmp_path / "config.yaml", "video: {}\n")
    with pytest.raises(ConfigError, match="'net' mapping"):
        load_config_bundle([source], required_sections=("net",))


def test_merge_replaces_sequences_and_does_not_mutate_inputs():
    base = {"nested": {"left": 1}, "items": [1, 2]}
    site = {"nested": {"right": 2}, "items": [3]}
    merged = merge_config_layers(base, site)

    assert merged == {"nested": {"left": 1, "right": 2}, "items": [3]}
    merged["nested"]["left"] = 9
    assert base["nested"]["left"] == 1


def test_resolve_config_paths_supports_cli_and_iterable_extras():
    assert resolve_config_paths("base.yaml", " a.yaml, ,b.yaml ") == (
        Path("base.yaml"), Path("a.yaml"), Path("b.yaml")
    )
    assert resolve_config_paths("base.yaml", [Path("site.yaml")]) == (
        Path("base.yaml"), Path("site.yaml")
    )
