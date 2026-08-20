#!/usr/bin/env python
r"""Small Paramiko helper for non-interactive cloud runs.

Connection settings are read from environment variables:
  FISH_REMOTE_HOST, FISH_REMOTE_PORT, FISH_REMOTE_USER, FISH_REMOTE_PASS

Examples:
  python tools/remote_paramiko.py exec "pwd && nvidia-smi"
  python tools/remote_paramiko.py upload local.py /root/work/local.py
  python tools/remote_paramiko.py download /root/work/out.json G:\fishnet\runs\out.json
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def _load_paramiko():
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("py_paramiko312", "py_paramiko38", "py_paramiko"):
        dep = root / ".deps" / name
        if dep.exists():
            sys.path.insert(0, str(dep))
    import paramiko  # type: ignore

    return paramiko


def _connect():
    paramiko = _load_paramiko()
    host = os.environ["FISH_REMOTE_HOST"]
    port = int(os.environ.get("FISH_REMOTE_PORT", "22"))
    user = os.environ["FISH_REMOTE_USER"]
    password = os.environ["FISH_REMOTE_PASS"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        timeout=20,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def cmd_exec(args: argparse.Namespace) -> int:
    client = _connect()
    try:
        stdin, stdout, stderr = client.exec_command(args.command, get_pty=args.pty, timeout=args.timeout)
        del stdin
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if out:
            print(out, end="")
        if err:
            print(err, end="", file=sys.stderr)
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


def _mkdir_remote(sftp, path: str) -> None:
    parts = pathlib.PurePosixPath(path).parts
    if not parts:
        return
    cur = "/" if parts[0] == "/" else ""
    for part in parts[1:] if cur == "/" else parts:
        if not part:
            continue
        cur = cur.rstrip("/") + "/" + part if cur else part
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def cmd_upload(args: argparse.Namespace) -> int:
    client = _connect()
    try:
        sftp = client.open_sftp()
        remote_parent = str(pathlib.PurePosixPath(args.remote).parent)
        if remote_parent and remote_parent != ".":
            _mkdir_remote(sftp, remote_parent)
        sftp.put(args.local, args.remote)
        sftp.close()
        return 0
    finally:
        client.close()


def cmd_download(args: argparse.Namespace) -> int:
    client = _connect()
    try:
        local = pathlib.Path(args.local)
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        sftp.get(args.remote, str(local))
        sftp.close()
        return 0
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exec = sub.add_parser("exec")
    p_exec.add_argument("command")
    p_exec.add_argument("--pty", action="store_true")
    p_exec.add_argument("--timeout", type=float, default=None)
    p_exec.set_defaults(func=cmd_exec)

    p_upload = sub.add_parser("upload")
    p_upload.add_argument("local")
    p_upload.add_argument("remote")
    p_upload.set_defaults(func=cmd_upload)

    p_download = sub.add_parser("download")
    p_download.add_argument("remote")
    p_download.add_argument("local")
    p_download.set_defaults(func=cmd_download)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
