from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_faster_whisper_is_not_a_base_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert not any(dep.startswith("faster-whisper") for dep in deps)

    voice_extra = data["project"]["optional-dependencies"]["voice"]
    assert any(dep.startswith("faster-whisper") for dep in voice_extra)


def test_manifest_includes_bundled_skills():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft skills" in manifest
    assert "graft optional-skills" in manifest


def test_protocol_adapter_extras_are_reachable_from_all():
    """Adapter extras must be pulled in by `[all]`, not just declared.

    `agui_adapter` and `acp_adapter` ship as console scripts, so their
    protocol packages have to exist in the environment that CI and
    packagers install (`.[all,dev]`). Declaring the extra on its own is
    not enough: `uv pip install -e ".[all,dev]"` silently omits it and
    every adapter import fails with ModuleNotFoundError at first use.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    all_extra = data["project"]["optional-dependencies"]["all"]

    for extra in ("acp", "agui"):
        assert f"hermes-agent[{extra}]" in all_extra
