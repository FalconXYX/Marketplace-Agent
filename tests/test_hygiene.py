"""Secret hygiene: nothing private ships, and the private files stay ignored.

Shipped files: tracked, plus untracked and not ignored, outside tests/.

| Check | Where |
|---|---|
| API key shapes: Google, Groq, OpenRouter, Discord bot token | Shipped files |
| Pickup addresses the test suite uses | Shipped files |
| The owner's address, email, secrets (env, .env, marketplace_agent.toml) | Shipped files |
| data/, .env, facebook_profile/, marketplace_agent.toml ignored | .gitignore |
| Only real settings, no secret values | .env.example, marketplace_agent.toml.example |
"""

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import TomlConfigSettingsSource

from marketplace_agent.config import Settings

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
TOML_EXAMPLE = ROOT / "marketplace_agent.toml.example"

KEY_PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "Groq API key": re.compile(r"gsk_[0-9A-Za-z]{40,}"),
    "OpenRouter API key": re.compile(r"sk-or-(?:v1-)?[0-9A-Za-z]{32,}"),
    "Discord bot token": re.compile(
        r"(?<![\w-])[MNO][\w-]{23,27}\.[\w-]{6}\.[\w-]{27,40}(?![\w-])"
    ),
}
# The sim's made-up address (sim/harness.py) and docstring examples in the guard may ship.
TEST_ADDRESSES = ("123 Example St", "77 St Clair Ave W")
PRIVATE_PATHS = (
    "data/agent.db",
    "data/photos/1/a.jpg",
    "data/facebook_profile/Default/Cookies",
    "facebook_profile/Default/Cookies",
    ".env",
    "marketplace_agent.toml",
)
SHIPPED_EXAMPLES = (".env.example", "marketplace_agent.toml.example")
MIN_PRIVATE_CHARS = 6  # shorter values match too much by chance


def _git(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("needs git")
    done = subprocess.run(
        [git, "-C", str(ROOT), *args], check=False, capture_output=True, text=True
    )
    if done.returncode not in (0, 1):  # 1: check-ignore found nothing
        pytest.skip(f"not a git checkout: {done.stderr.strip()}")
    return done.stdout


def _shipped_texts() -> dict[str, str]:
    """Every file that could be committed, outside tests/, as text. Binary files are skipped."""
    listed = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    texts = {}
    for name in sorted(set(filter(None, listed.split("\0")))):
        path = ROOT / name
        if name.startswith("tests/") or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" not in data:
            texts[name] = data.decode("utf-8", errors="replace")
    assert texts, "found no files to check"
    return texts


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def _hits(texts: dict[str, str], needles: dict[str, str]) -> list[str]:
    """'path: label' for every needle (label -> text) found, ignoring case and spacing."""
    squashed = {name: _squash(text) for name, text in texts.items()}
    return [
        f"{name}: {label}"
        for label, needle in needles.items()
        for name, text in squashed.items()
        if _squash(needle) in text
    ]


def test_key_patterns_match_real_shapes():
    alnum = "aB3" * 30
    samples = {
        "Google API key": "AIza" + alnum[:35],
        "Groq API key": "gsk_" + alnum[:52],
        "OpenRouter API key": "sk-or-v1-" + "0f" * 32,
        "Discord bot token": f"MT{alnum[:24]}.{alnum[:6]}.{alnum[:38]}",
    }
    for label, sample in samples.items():
        assert KEY_PATTERNS[label].search(f"key = '{sample}'"), label


def test_no_api_keys_in_shipped_files():
    hits = [
        f"{name}: {label}"
        for name, text in _shipped_texts().items()
        for label, pattern in KEY_PATTERNS.items()
        if pattern.search(text)
    ]
    assert not hits, f"API-key-shaped strings: {hits}"


def test_no_test_pickup_address_in_shipped_files():
    hits = _hits(_shipped_texts(), {address: address for address in TEST_ADDRESSES})
    assert not hits, f"test pickup addresses outside tests/: {hits}"


def test_no_local_private_values_in_shipped_files(monkeypatch):
    """The owner's real address and secrets, read the way the agent reads them."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(Settings.model_config, "toml_file", str(ROOT / "marketplace_agent.toml"))
    try:
        settings = Settings()
    except ValidationError as exc:
        pytest.skip(f"local settings do not load: {exc.error_count()} error(s)")
    private = {"pickup_address": settings.pickup_address, "kijiji_email": settings.kijiji_email}
    for name in Settings.model_fields:
        value = getattr(settings, name)
        if isinstance(value, SecretStr):
            private[name] = value.get_secret_value()
    needles = {
        name: value
        for name, value in private.items()
        if value and len(value.strip()) >= MIN_PRIVATE_CHARS
    }
    if not needles:
        pytest.skip("no local address or secrets configured")
    hits = _hits(_shipped_texts(), needles)
    assert not hits, f"locally configured private values in shipped files: {hits}"


def test_gitignore_covers_private_paths():
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    for pattern in ("data/", ".env", "facebook_profile/", "marketplace_agent.toml"):
        assert pattern in lines, f".gitignore lacks {pattern}"
    ignored = set(_git("check-ignore", "--no-index", *PRIVATE_PATHS, *SHIPPED_EXAMPLES).split())
    assert ignored == set(PRIVATE_PATHS)


def _env_example_lines() -> list[tuple[str, str]]:
    """(name, value) for every NAME=value line, commented out or not; inline comments dropped."""
    pairs = []
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = re.match(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$", line)
        if match:
            name, value = match.groups()
            pairs.append((name, re.split(r"\s#", value, maxsplit=1)[0].strip()))
    return pairs


def _field_for(env_name: str) -> str:
    return env_name.removeprefix("MA_").lower()


def _secret_fields() -> set[str]:
    fields = Settings.model_fields.items()
    secrets = {name for name, f in fields if SecretStr in getattr(f.annotation, "__args__", ())}
    assert secrets, "expected SecretStr settings"
    return secrets


def test_env_example_names_real_settings_and_every_secret():
    names = [name for name, _ in _env_example_lines()]
    unknown = [name for name in names if _field_for(name) not in Settings.model_fields]
    assert not unknown, f".env.example names unknown settings: {unknown}"
    missing = _secret_fields() - {_field_for(name) for name in names}
    assert not missing, f".env.example lacks secrets: {sorted(missing)}"


def test_env_example_holds_no_secret_values():
    private = _secret_fields() | {"pickup_address", "kijiji_email"}
    filled = [name for name, value in _env_example_lines() if _field_for(name) in private and value]
    assert not filled, f".env.example has values for: {filled}"


def test_a_developers_toml_never_reaches_the_tests(monkeypatch, tmp_path):
    """The README quickstart makes one in the repo root; tests must see the defaults anyway."""
    local = 'enabled_platforms = ["kijiji"]\n\n[[availability]]\nweekday = 0\nstart = "18:00"\n'
    (tmp_path / "marketplace_agent.toml").write_text(local + 'end = "21:00"\n')
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    assert settings.enabled_platforms == ["fake"]
    assert settings.availability == Settings.model_fields["availability"].get_default(
        call_default_factory=True
    )


def test_toml_example_loads_into_settings(monkeypatch, tmp_path):
    raw = tomllib.loads(TOML_EXAMPLE.read_text())
    unknown = sorted(set(raw) - set(Settings.model_fields))
    assert not unknown, f"marketplace_agent.toml.example has unknown keys: {unknown}"
    assert "pickup_address" not in raw, "the exact address belongs in .env, never an example"
    monkeypatch.chdir(tmp_path)  # no local marketplace_agent.toml or env settings mixed in
    for name in [name for name in os.environ if name.startswith("MA_")]:
        monkeypatch.delenv(name)
    values = TomlConfigSettingsSource(Settings, toml_file=TOML_EXAMPLE)()
    settings = Settings(_env_file=None, **values)
    assert settings.availability and settings.enabled_platforms
    for cap in (settings.daily_outbound_cap, settings.daily_publish_cap):
        assert {"fake", "kijiji", "facebook"} <= set(cap), "a missing platform gets a cap of 0"
