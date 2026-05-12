from pathlib import Path
from click.testing import CliRunner


def test_ledger_tail_works_on_empty(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from cli import cli
    r = CliRunner().invoke(cli, ["ledger", "tail", "5"])
    assert r.exit_code == 0
    # Empty ledger should print []
    assert r.output.strip() in ("[]", "[\n]")


def test_baseline_show_works_on_empty(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from cli import cli
    r = CliRunner().invoke(cli, ["baseline", "show"])
    assert r.exit_code == 0
    assert "(no baseline yet)" in r.output
