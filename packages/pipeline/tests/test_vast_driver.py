"""Tests for the rented-host driver, `benchmarks/vast/campaign.py`.

The driver is a stdlib script outside the package, loaded here by path. No
instance is touched and no command runs: git, ssh, scp, rsync and the vastai
CLI are replaced, and the tests look at what the driver would hand to them.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

DRIVER = Path(__file__).resolve().parents[3] / "benchmarks" / "vast" / "campaign.py"


def _load_driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vast_campaign", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_push_stages_every_scp_source_even_with_no_clip_to_carry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Found on the first shakedown (2026-09-15): with the clips list empty,
    the clips directory was never created and scp refused the push."""
    driver = _load_driver()
    monkeypatch.setattr(driver, "SCRATCH", tmp_path)
    monkeypatch.setattr(driver, "CLIPS", ())
    monkeypatch.setattr(driver, "OPTIONAL_CLIPS", ())
    monkeypatch.setattr(driver, "head_commit", lambda: "a" * 40)

    def fake_bundle(argv: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        # The ref bookkeeping around `git bundle create` is the other test's;
        # here only the bundle file has to appear where scp will look.
        assert argv[:3] == ["git", "-C", str(driver.REPO)]
        if argv[3] == "bundle":
            assert check
            Path(argv[5]).write_bytes(b"bundle")
        return subprocess.CompletedProcess(argv, 0)

    remote_commands: list[str] = []

    def fake_ssh(_instance: str, command: str, *, check: bool = True) -> int:
        remote_commands.append(command)
        return 0

    sent: list[Path] = []

    def fake_scp(_instance: str, sources: list[Path], _remote: str) -> None:
        sent.extend(sources)

    monkeypatch.setattr(driver.subprocess, "run", fake_bundle)
    monkeypatch.setattr(driver, "ssh", fake_ssh)
    monkeypatch.setattr(driver, "scp", fake_scp)

    assert driver.cmd_push(type("Args", (), {"instance": "1", "commit": None})()) == 0

    assert [source.name for source in sent] == [
        "unproject-aaaaaaaaaaaa.bundle",
        "unproject-clips",
        "setup-instance.sh",
    ]
    assert all(source.exists() for source in sent), "scp needs every source on disk"
    assert (tmp_path / "unproject-clips").is_dir()
    assert list((tmp_path / "unproject-clips").iterdir()) == []
    assert any("mv -f unproject-clips clips" in command for command in remote_commands)


def test_run_keeps_its_log_out_of_the_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Found on the CUDA shakedown (2026-09-15): the log written into the
    repository root was an untracked file, so the preflight saw a dirty tree."""
    driver = _load_driver()
    commands: list[str] = []

    def fake_ssh(_instance: str, command: str, *, check: bool = True) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(driver, "ssh", fake_ssh)
    args = type("Args", (), {"instance": "1", "machine": "C-rtx4070", "suites": "matrix,cpu"})()

    assert driver.cmd_run(args) == 0

    (command,) = commands
    assert "make bench-matrix bench-cpu MACHINE=C-rtx4070" in command
    log = driver.remote_log("C-rtx4070")
    assert f"> {log} 2>&1" in command
    assert not log.startswith(driver.REMOTE_REPO + "/")
    assert log.startswith(driver.REMOTE_WORK + "/")


def test_push_bundles_the_commit_asked_for_through_a_temporary_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`git bundle` refuses a bare commit id ("empty bundle"); with --commit
    the driver makes a ref for it and removes the ref after, and does not
    look at the tree, which may have moved on (2026-09-15: a registry line
    committed for another machine while the rented hosts stayed at 7e7ac5c)."""
    driver = _load_driver()
    monkeypatch.setattr(driver, "SCRATCH", tmp_path)
    monkeypatch.setattr(driver, "CLIPS", ())
    monkeypatch.setattr(driver, "OPTIONAL_CLIPS", ())
    monkeypatch.setattr(driver, "ssh", lambda *_a, **_k: 0)
    monkeypatch.setattr(driver, "scp", lambda *_a, **_k: None)
    monkeypatch.setattr(
        driver, "head_commit", lambda: pytest.fail("HEAD must not be consulted with --commit")
    )
    sha = "7e7ac5cf7645a64244f87b9c22d49be032bee448"
    git_calls: list[list[str]] = []

    def fake_git(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        git_calls.append(argv)
        if argv[3:5] == ["rev-parse", "--verify"]:
            return subprocess.CompletedProcess(argv, 0, stdout=sha + "\n", stderr="")
        if argv[3] == "bundle":
            Path(argv[5]).write_bytes(b"bundle")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(driver.subprocess, "run", fake_git)

    assert driver.cmd_push(type("Args", (), {"instance": "1", "commit": "7e7ac5c"})()) == 0

    ref = f"refs/campaign/{sha[:12]}"
    verbs = [call[3:] for call in git_calls]
    assert ["rev-parse", "--verify", "7e7ac5c^{commit}"] in verbs
    assert ["update-ref", ref, sha] in verbs
    assert ["bundle", "create", str(tmp_path / f"unproject-{sha[:12]}.bundle"), ref] in verbs
    assert ["update-ref", "-d", ref] in verbs


def test_destroy_skips_the_confirmation_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a terminal the CLI printed "Aborted" and left the instance
    running (2026-09-15); the driver must say yes itself."""
    driver = _load_driver()
    calls: list[tuple[str, ...]] = []

    def fake_vastai(*args: str, raw: bool = False) -> str:
        calls.append(args)
        return "destroyed"

    monkeypatch.setattr(driver, "vastai", fake_vastai)

    assert driver.cmd_destroy(type("Args", (), {"instance": "42"})()) == 0
    assert calls == [("destroy", "instance", "42", "--yes")]


def test_seed_resumes_over_the_direct_port_with_the_key_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _load_driver()
    repo = tmp_path / "repo"
    for name in driver.SEED_FILES[:2]:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(b"zip")
    monkeypatch.setattr(driver, "REPO", repo)
    monkeypatch.setattr(
        driver,
        "instance",
        lambda _id: {
            "ssh_host": "ssh1.vast.ai",
            "ssh_port": 1,
            "public_ipaddr": "1.2.3.4",
            "direct_port_start": 5000,
        },
    )
    remote: list[str] = []

    def fake_ssh(_instance: str, command: str, *, check: bool = True) -> int:
        remote.append(command)
        return 0

    transfers: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        transfers.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(driver, "ssh", fake_ssh)
    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    args = type("Args", (), {"instance": "1", "direct": True, "ssh_key": "~/.ssh/k"})()

    assert driver.cmd_seed(args) == 0

    assert "git clone" in remote[0] and "mkdir -p" in remote[0]
    assert len(transfers) == 2  # only the files present here travel
    argv = transfers[0]
    assert argv[:3] == ["rsync", "--partial", "--inplace"]
    assert "-p 5000" in argv[4] and "IdentitiesOnly=yes" in argv[4] and "/.ssh/k" in argv[4]
    assert argv[-1] == f"root@1.2.3.4:{driver.REMOTE_REPO}/datasets/_download/"
