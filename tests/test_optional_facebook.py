"""Everything but Facebook works when Playwright cannot be imported (the `facebook` extra is optional)."""

import os
import subprocess
import sys
import textwrap

BLOCK = textwrap.dedent(
    """
    import sys
    for name in ("playwright", "playwright.sync_api", "playwright.async_api"):
        sys.modules[name] = None  # any import of it raises ImportError
    """
)


def _run(tmp_path, body: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MA_") and not k.endswith("_API_KEY")}
    env["MA_DATA_DIR"] = str(tmp_path / "data")
    return subprocess.run(
        [sys.executable, "-c", BLOCK + textwrap.dedent(body)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_every_module_imports_without_playwright(tmp_path):
    result = _run(
        tmp_path,
        """
        import importlib, pkgutil
        import marketplace_agent
        failed = []
        for mod in pkgutil.walk_packages(marketplace_agent.__path__, "marketplace_agent."):
            try:
                importlib.import_module(mod.name)
            except Exception as exc:
                failed.append(f"{mod.name}: {exc!r}")
        assert not failed, failed
        import playwright  # still blocked after importing everything
        """,
    )
    assert "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr, result.stderr
    assert "AssertionError" not in result.stderr, result.stderr


def test_non_facebook_commands_work_without_playwright(tmp_path):
    from PIL import Image

    Image.new("RGB", (64, 48), (120, 80, 40)).save(tmp_path / "desk.jpg")
    result = _run(
        tmp_path,
        """
        from typer.testing import CliRunner
        from marketplace_agent.cli import app

        runner = CliRunner()
        commands = [
            ["--help"],
            ["init"],
            ["add-item", "--hint", "IKEA desk $120", "--photo", "desk.jpg"],
            ["goal", "1", "--price", "120", "--floor", "95"],
            ["items"],
            ["show", "1"],
            ["pause", "1"],
            ["resume", "1"],
            ["escalations"],
            ["log", "1"],
            ["trust"],
            ["memory"],
            ["quota"],
            ["tick"],
            ["doctor"],
            ["simulate", "--scenario", "desk", "--days", "7", "--quiet"],
        ]
        for args in commands:
            out = runner.invoke(app, args)
            assert out.exception is None or isinstance(out.exception, SystemExit), (args, out.output, repr(out.exception))
            assert "Traceback" not in out.output, (args, out.output)
            if args[0] != "doctor":  # doctor exits non-zero without a model key or address
                assert out.exit_code == 0, (args, out.output)
        print("ALL OK")
        """,
    )
    assert "ALL OK" in result.stdout, result.stdout[-3000:] + result.stderr[-3000:]
