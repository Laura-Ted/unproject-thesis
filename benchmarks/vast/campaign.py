#!/usr/bin/env python3
"""Drive one rented Linux machine through the campaign, from the operator's Mac.

ADR-0006 §4.3. A thin, explicit wrapper over the `vastai` CLI and ssh; every
step is a subcommand so a step that fails can be rerun on its own, and every
remote command is printed before it runs. Nothing here deletes anything but
the rented instance, and only under `destroy`.

    campaign.py search  --gpu RTX_4070 [--consumer-cpu] [--max-dph 0.6]
    campaign.py create  <offer-id> --machine C-rtx4070 [--disk 80]
    campaign.py wait    <instance-id>
    campaign.py probe   <instance-id>              # host load, card idle?, network speeds
    campaign.py push    <instance-id> [--commit <sha>]     # bundle at HEAD (or the sha), scripts
    campaign.py seed    <instance-id> [--direct] [--ssh-key k]  # archives + weights from here
    campaign.py setup   <instance-id> --machine C-rtx4070 [--commit <sha>] [--accept-mupots-terms]
    campaign.py run     <instance-id> --machine C-rtx4070 [--suites matrix,people,cpu]
    campaign.py pull    <instance-id> --machine C-rtx4070 [--into artifacts/campaign]
    campaign.py ssh     <instance-id> [-- command...]
    campaign.py destroy <instance-id>

Needs: an API key set with `uvx vastai set api-key <key>` (kept by the CLI in
~/.vast_api_key, never in this repository), an SSH public key registered
with the account, and a clean committed tree - the bundle is made from HEAD
and a dirty tree is refused, because the results have to name a commit.
`--commit` bundles that commit instead, so a later commit on the branch
(a lab note, a registry line) does not move the campaign's commit.

Two things learnt on 2026-09-15 that `probe` and `seed` answer: a host's
advertised bandwidth says nothing about what the origin servers and the
operator's machine actually get (0.1 to 30 MB/s were measured), and a
shared host's external load or a card busy for a tenant this container
cannot see contaminate the numbers - both are visible before a byte is
transferred. `seed` copies the official archives and the weights from the
operator's machine over the instance's direct ssh port (the proxy managed
1 MB/s); `setup` then verifies every digest as it always did, so the bytes
are the campaign's whether they came from the origin or from here.

Stdlib only, so it runs with the system Python.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Files carried from the operator's machine because a transcode is not
# bit-stable across builds. Empty since 2026-09-15: every campaign input is
# derived from an official archive by the ingest, identically everywhere.
CLIPS: tuple[str, ...] = ()
OPTIONAL_CLIPS: tuple[str, ...] = ()
# Minimal image with the CUDA runtime and Ubuntu 24.04; torch brings its own
# libraries from the lockfile, the image only has to boot with the driver.
DEFAULT_IMAGE = "nvidia/cuda:13.0.1-base-ubuntu24.04"
# Where `push` stages the bundle and the clips directory before scp.
SCRATCH = Path("/tmp")
REMOTE_DIR = "/root/unproject-push"
REMOTE_WORK = "/root/unproject-campaign"
REMOTE_REPO = f"{REMOTE_WORK}/Unproject"
CONSUMER_CPU_MARKERS = ("Ryzen", "Core(TM) i", "Core i", "Core Ultra")
# What `seed` carries, relative to the repository, when present here: the
# official archives fetch-datasets.sh would download and the pinned weights.
# Each is verified against its pinned digest by setup on the instance.
SEED_FILES: tuple[str, ...] = (
    "datasets/_download/MOT16.zip",
    "datasets/_download/MultiPersonTestSet.zip",
    "models/multihmr2.pt",
)
# Origin servers whose speed `probe` measures from the instance.
PROBE_URLS: tuple[tuple[str, str], ...] = (
    ("motchallenge.net", "https://motchallenge.net/data/MOT16.zip"),
    (
        "gvv.mpi-inf.mpg.de",
        "https://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/MultiPersonTestSet.zip",
    ),
    (
        "naverlabs",
        "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multihmr2.pt",
    ),
    ("pypi", "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.0.tar.gz"),
    (
        "github",
        "https://github.com/facebookresearch/dinov3/archive/"
        "6876159a11b4df116f30f667f8c9888617df0751.zip",
    ),
)


def vastai(*args: str, raw: bool = False) -> str:
    """The CLI, through uvx so nothing has to be installed globally."""
    base = ["vastai"] if shutil.which("vastai") else ["uvx", "vastai"]
    command = [*base, *args, *(["--raw"] if raw else [])]
    print("$", " ".join(shlex.quote(part) for part in command), file=sys.stderr)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(completed.stdout, completed.stderr, file=sys.stderr)
        raise SystemExit(f"vastai exited {completed.returncode}")
    return completed.stdout


def instance(instance_id: str) -> dict[str, object]:
    data = json.loads(vastai("show", "instance", instance_id, raw=True))
    return data if isinstance(data, dict) else {}


def ssh_target(instance_id: str, *, direct: bool = False) -> tuple[str, str]:
    """The proxy endpoint vast.ai gives, or with `direct` the host's own
    address and port, which carries large transfers far faster."""
    info = instance(instance_id)
    if direct:
        host = str(info.get("public_ipaddr") or "")
        port = str(info.get("direct_port_start") or "")
        if not host or port in {"", "-1"}:
            raise SystemExit(f"instance {instance_id} has no direct ssh port; drop --direct")
        return host, port
    host = str(info.get("ssh_host") or "")
    port = str(info.get("ssh_port") or "")
    if not host or not port:
        raise SystemExit(
            f"instance {instance_id} has no ssh endpoint yet: {info.get('actual_status')}"
        )
    return host, port


def ssh_options(port: str, key: str | None = None) -> list[str]:
    """ssh/scp/rsync options: the port, first-contact host keys accepted, and
    the identity when the operator's ssh config does not offer it (the direct
    address is a bare IP, which a `Host *.vast.ai` entry does not match)."""
    options = ["-p", port, "-o", "StrictHostKeyChecking=accept-new"]
    if key:
        options += ["-i", str(Path(key).expanduser()), "-o", "IdentitiesOnly=yes"]
    return options


def ssh(instance_id: str, command: str, *, check: bool = True) -> int:
    host, port = ssh_target(instance_id)
    argv = [
        "ssh",
        "-p",
        port,
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        f"root@{host}",
        command,
    ]
    print("$", " ".join(shlex.quote(part) for part in argv), file=sys.stderr)
    code = subprocess.run(argv, check=False).returncode
    if check and code != 0:
        raise SystemExit(f"remote command exited {code}")
    return code


def scp(instance_id: str, sources: list[Path], remote: str) -> None:
    host, port = ssh_target(instance_id)
    argv = [
        "scp",
        "-P",
        port,
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-r",
        *[str(source) for source in sources],
        f"root@{host}:{remote}",
    ]
    print("$", " ".join(shlex.quote(part) for part in argv), file=sys.stderr)
    subprocess.run(argv, check=True)


def remote_log(machine: str) -> str:
    """Where `run` writes the campaign log on the instance: outside the checkout."""
    return f"{REMOTE_WORK}/campaign-{machine}.log"


def head_commit() -> str:
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit("the tree is dirty; commit first, the results must name a commit")
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def campaign_commit(requested: str | None) -> str:
    """HEAD of a clean tree, or the commit asked for with --commit, resolved
    to its full id; the tree's state does not matter then, the bundle is
    made from the commit and nothing else."""
    if not requested:
        return head_commit()
    resolved = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", f"{requested}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        raise SystemExit(f"{requested} is not a commit of this repository")
    return resolved.stdout.strip()


def make_bundle(commit: str, bundle: Path) -> None:
    """A bundle of one commit and its history. `git bundle` wants a ref, not
    a bare id ("Refusing to create empty bundle"), so a temporary one is
    made under refs/campaign/ and removed after."""
    ref = f"refs/campaign/{commit[:12]}"
    subprocess.run(["git", "-C", str(REPO), "update-ref", ref, commit], check=True)
    try:
        subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(bundle), ref], check=True)
    finally:
        subprocess.run(["git", "-C", str(REPO), "update-ref", "-d", ref], check=False)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_search(args: argparse.Namespace) -> int:
    query = (
        f"gpu_name={args.gpu} num_gpus=1 rentable=true verified=true "
        f"cuda_max_good>={args.cuda} disk_space>={args.disk} inet_down>=200 "
        f"reliability>=0.95 dph_total<={args.max_dph}"
    )
    offers = json.loads(vastai("search", "offers", query, "-o", "dph_total", raw=True))
    rows = []
    for offer in offers:
        cpu = str(offer.get("cpu_name") or "")
        consumer = any(marker in cpu for marker in CONSUMER_CPU_MARKERS)
        if args.consumer_cpu and not consumer:
            continue
        rows.append(
            (
                offer.get("id"),
                offer.get("gpu_name"),
                offer.get("gpu_ram"),
                f"{float(offer.get('dph_total') or 0):.3f}",
                offer.get("driver_version"),
                offer.get("cuda_max_good"),
                cpu[:40],
                offer.get("cpu_cores_effective"),
                offer.get("cpu_ram"),
                offer.get("geolocation"),
                f"{float(offer.get('reliability2') or offer.get('reliability') or 0):.3f}",
            )
        )
    header = (
        "offer",
        "gpu",
        "vram",
        "$/h",
        "driver",
        "cuda",
        "cpu",
        "cores",
        "ram MB",
        "where",
        "reliab",
    )
    print("\t".join(header))
    for row in rows[: args.limit]:
        print("\t".join(str(item) for item in row))
    if not rows:
        print("no offer matched; loosen --max-dph or drop --consumer-cpu", file=sys.stderr)
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    out = vastai(
        "create",
        "instance",
        str(args.offer),
        "--image",
        args.image,
        "--disk",
        str(args.disk),
        "--ssh",
        "--direct",
        "--label",
        f"thesis-{args.machine}",
        "--env",
        f"-e UNPROJECT_MACHINE_ID={args.machine}",
        raw=True,
    )
    print(out.strip())
    try:
        created = json.loads(out)
        print(f"instance id: {created.get('new_contract')}")
    except ValueError:
        pass
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        info = instance(args.instance)
        status = info.get("actual_status")
        print(f"{status} ({info.get('status_msg') or ''})".strip(), file=sys.stderr)
        up = status == "running" and info.get("ssh_host")
        if up and ssh(args.instance, "true", check=False) == 0:
            print(
                f"instance {args.instance} is up: ssh -p {info['ssh_port']} root@{info['ssh_host']}"
            )
            return 0
        time.sleep(15)
    raise SystemExit("timed out waiting for the instance")


def cmd_push(args: argparse.Namespace) -> int:
    commit = campaign_commit(args.commit)
    bundle = SCRATCH / f"unproject-{commit[:12]}.bundle"
    make_bundle(commit, bundle)
    clips_dir = SCRATCH / "unproject-clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    # Made even when no clip is carried: scp needs every source to exist and
    # the instance's `mv` expects the directory, empty or not.
    clips_dir.mkdir(parents=True)
    for name in (*CLIPS, *OPTIONAL_CLIPS):
        source = REPO / "datasets" / name / "clip.mp4"
        if source.is_file():
            (clips_dir / name).mkdir()
            shutil.copy2(source, clips_dir / name / "clip.mp4")
        elif name in CLIPS:
            raise SystemExit(f"{source} is missing; run benchmarks/fetch-datasets.sh first")
    ssh(args.instance, f"mkdir -p {REMOTE_DIR}")
    scp(
        args.instance,
        [bundle, clips_dir, REPO / "benchmarks" / "vast" / "setup-instance.sh"],
        REMOTE_DIR,
    )
    ssh(
        args.instance,
        f"cd {REMOTE_DIR} && mv -f {bundle.name} unproject.bundle && "
        "mv -f unproject-clips clips && ls -la",
    )
    print(f"pushed commit {commit[:12]}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    commit = campaign_commit(args.commit)
    # The MuPoTS terms are the operator's to accept (ADR-0004); the flag says
    # they did, and the instance fetches the archive from MPI on that word.
    accept = "UNPROJECT_ACCEPT_MUPOTS_TERMS=1 " if args.accept_mupots_terms else ""
    return ssh(
        args.instance,
        f"cd {REMOTE_DIR} && {accept}bash setup-instance.sh {commit} {shlex.quote(args.machine)}",
    )


def cmd_seed(args: argparse.Namespace) -> int:
    """Copy the archives and the weights present here into the instance's
    checkout, so setup verifies them instead of fetching them. The checkout
    is cloned from the pushed bundle if it is not there yet (setup accepts
    an existing clone); rsync resumes a transfer the proxy dropped."""
    present = [REPO / name for name in SEED_FILES if (REPO / name).is_file()]
    if not present:
        raise SystemExit("nothing to seed: run benchmarks/fetch-datasets.sh here first")
    ssh(
        args.instance,
        f"test -d {REMOTE_REPO}/.git || git clone -q {REMOTE_DIR}/unproject.bundle {REMOTE_REPO}; "
        f"mkdir -p {REMOTE_REPO}/datasets/_download {REMOTE_REPO}/models",
    )
    host, port = ssh_target(args.instance, direct=args.direct)
    transport = " ".join(shlex.quote(part) for part in ["ssh", *ssh_options(port, args.ssh_key)])
    for source in present:
        remote = f"{REMOTE_REPO}/{source.relative_to(REPO).parent.as_posix()}/"
        argv = [
            "rsync",
            "--partial",
            "--inplace",
            "-e",
            transport,
            str(source),
            f"root@{host}:{remote}",
        ]
        print("$", " ".join(shlex.quote(part) for part in argv), file=sys.stderr)
        subprocess.run(argv, check=True)
    print(f"seeded {len(present)} file(s); setup verifies their digests")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """What the instance is like before anything is spent on it: the host's
    load (external, if nothing of ours runs), whether the card is idle, and
    what the origin servers, PyPI and GitHub give it - plus what this
    machine can push to it. Prints, judges nothing."""
    curls = " ".join(
        f'printf "%-22s " {shlex.quote(name)}; curl -sL -o /dev/null --max-time 8 '
        f'-w "%{{speed_download}}\\n" {shlex.quote(url)} | '
        "awk '{printf \"%.1f MB/s\\n\", $1/1e6}';"
        for name, url in PROBE_URLS
    )
    ssh(
        args.instance,
        "echo load $(cat /proc/loadavg | cut -d' ' -f1-3) on $(nproc) threads, quota "
        "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null); "
        "nvidia-smi --query-gpu=name,driver_version,utilization.gpu,power.draw,pstate "
        "--format=csv,noheader; " + curls,
        check=False,
    )
    host, port = ssh_target(args.instance, direct=args.direct)
    argv = ["ssh", *ssh_options(port, args.ssh_key), f"root@{host}", "cat > /dev/null"]
    print(
        "$ head -c 50000000 /dev/zero |",
        " ".join(shlex.quote(part) for part in argv),
        file=sys.stderr,
    )
    started = time.time()
    with subprocess.Popen(argv, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        proc.stdin.write(bytes(50_000_000))
        proc.stdin.close()
        proc.wait()
    elapsed = time.time() - started
    print(f"upload from here: {50 / elapsed:.1f} MB/s ({'direct' if args.direct else 'proxy'})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    targets = " ".join(
        f"bench-{suite.strip()}" for suite in args.suites.split(",") if suite.strip()
    )
    # nohup + a log, so a dropped ssh session does not kill an hour of work;
    # `tail -f` follows it and Ctrl-C detaches without stopping the run. The
    # log lives beside the checkout, not in it: an untracked file in the tree
    # is a dirty tree, which the preflight refuses and every job records.
    log = remote_log(args.machine)
    command = (
        f"cd {REMOTE_REPO} && export PATH=$HOME/.local/bin:$PATH && "
        f"nohup make {targets} MACHINE={shlex.quote(args.machine)} "
        f"> {log} 2>&1 & echo started; sleep 2; "
        f"tail -n 5 {log}"
    )
    ssh(args.instance, command)
    print(f"follow with: {sys.argv[0]} ssh {args.instance} -- tail -f {log}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    into = Path(args.into)
    into.mkdir(parents=True, exist_ok=True)
    ssh(
        args.instance,
        f"cd {REMOTE_REPO} && export PATH=$HOME/.local/bin:$PATH && "
        f"make bench-collect MACHINE={shlex.quote(args.machine)} && "
        f"ls -1t artifacts/campaign/*.tar.gz | head -1",
    )
    host, port = ssh_target(args.instance)
    listing = subprocess.run(
        [
            "ssh",
            "-p",
            port,
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"root@{host}",
            f"ls -1t {REMOTE_REPO}/artifacts/campaign/*.tar.gz | head -1",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not listing:
        raise SystemExit("no archive on the instance")
    argv = [
        "scp",
        "-P",
        port,
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"root@{host}:{listing}",
        str(into),
    ]
    print("$", " ".join(shlex.quote(part) for part in argv), file=sys.stderr)
    subprocess.run(argv, check=True)
    archive = into / Path(listing).name
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(into)], check=True)
    print(f"unpacked {archive} into {into}/{args.machine}/benchmark-v2/")
    log = remote_log(args.machine)
    subprocess.run(
        [
            "scp",
            "-P",
            port,
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"root@{host}:{log}",
            str(into / f"{args.machine}-campaign.log"),
        ],
        check=False,
    )
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    command = " ".join(shlex.quote(part) for part in args.command) if args.command else "bash -l"
    host, port = ssh_target(args.instance)
    argv = [
        "ssh",
        "-p",
        port,
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-t",
        f"root@{host}",
        command,
    ]
    return subprocess.run(argv, check=False).returncode


def cmd_destroy(args: argparse.Namespace) -> int:
    # `--yes`: the CLI asks for confirmation on a terminal and, without one,
    # prints "Aborted" and leaves the instance running - and billing.
    print(vastai("destroy", "instance", args.instance, "--yes"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search")
    search.add_argument(
        "--gpu", required=True, help="vast.ai gpu_name, e.g. RTX_4070, RTX_3060, RTX_4090, RTX_5090"
    )
    search.add_argument(
        "--consumer-cpu", action="store_true", help="only hosts with a Ryzen/Core CPU"
    )
    search.add_argument("--max-dph", type=float, default=1.5)
    search.add_argument(
        "--cuda", type=float, default=13.0, help="minimum CUDA the host driver supports"
    )
    search.add_argument("--disk", type=int, default=60)
    search.add_argument("--limit", type=int, default=15)
    search.set_defaults(func=cmd_search)

    create = sub.add_parser("create")
    create.add_argument("offer", type=int)
    create.add_argument("--machine", required=True)
    create.add_argument("--image", default=DEFAULT_IMAGE)
    create.add_argument("--disk", type=int, default=80)
    create.set_defaults(func=cmd_create)

    for name, func in (("wait", cmd_wait), ("push", cmd_push), ("destroy", cmd_destroy)):
        p = sub.add_parser(name)
        p.add_argument("instance")
        if name == "wait":
            p.add_argument("--timeout", type=int, default=900)
        if name == "push":
            p.add_argument("--commit", help="bundle this commit instead of a clean HEAD")
        p.set_defaults(func=func)

    for name, func in (("seed", cmd_seed), ("probe", cmd_probe)):
        p = sub.add_parser(name)
        p.add_argument("instance")
        p.add_argument(
            "--direct", action="store_true", help="the host's own address and port, not the proxy"
        )
        p.add_argument("--ssh-key", help="private key to offer, when your ssh config does not")
        p.set_defaults(func=func)

    for name, func in (("setup", cmd_setup), ("run", cmd_run), ("pull", cmd_pull)):
        p = sub.add_parser(name)
        p.add_argument("instance")
        p.add_argument("--machine", required=True)
        if name == "setup":
            p.add_argument("--commit", help="the commit pushed with --commit, if one was")
            p.add_argument(
                "--accept-mupots-terms",
                action="store_true",
                help="you have read the MuPoTS-3D terms; the instance may fetch TS10/TS11",
            )
        if name == "run":
            p.add_argument("--suites", default="matrix,people,cpu")
        if name == "pull":
            p.add_argument("--into", default=str(REPO / "artifacts" / "campaign"))
        p.set_defaults(func=func)

    remote = sub.add_parser("ssh")
    remote.add_argument("instance")
    remote.add_argument("command", nargs="*")
    remote.set_defaults(func=cmd_ssh)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
