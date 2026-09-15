"""Portable Go 1.22 from go.dev, then ``go build`` for ``chr``.

Neighbor PCs often have Python and CUDA but no Go and no ``chr.exe``.
Setup fetches the official archive (SHA-256 pinned), unpacks it under
``$DEEPFOLD_HOME/toolchains``, and writes ``chr`` to ``$DEEPFOLD_HOME/bin``
plus the checkout root. PATH Go 1.22+ is used when present. Not a
system-wide install: nothing is added to Program Files or ``/usr/local``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .paths import REPO, chr_exe_names, deepfold_home, find_chr_bin

__all__ = [
    "ARCHIVES",
    "GO_VERSION",
    "GoToolchainError",
    "ensure_chr",
    "go_version_at_least",
    "host_key",
    "main",
]

GO_VERSION = "1.22.12"
GO_DL = "https://go.dev/dl"
USER_AGENT = "deepfold-setup"

# go.dev/dl/?mode=json kind=archive for go1.22.12 (last 1.22, matches go.mod).
ARCHIVES: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "amd64"): (
        "go1.22.12.windows-amd64.zip",
        "2ceda04074eac51f4b0b85a9fcca38bcd49daee24bed9ea1f29958a8e22673a6",
    ),
    ("windows", "arm64"): (
        "go1.22.12.windows-arm64.zip",
        "6b9eaf160b155e02ffe9ed603f162ecc3264f6130c8fcf83bb77087f9807fdec",
    ),
    ("linux", "amd64"): (
        "go1.22.12.linux-amd64.tar.gz",
        "4fa4f869b0f7fc6bb1eb2660e74657fbf04cdd290b5aef905585c86051b34d43",
    ),
    ("linux", "arm64"): (
        "go1.22.12.linux-arm64.tar.gz",
        "fd017e647ec28525e86ae8203236e0653242722a7436929b1f775744e26278e7",
    ),
    ("darwin", "amd64"): (
        "go1.22.12.darwin-amd64.tar.gz",
        "e7bbe07e96f0bd3df04225090fe1e7852ed33af37c43a23e16edbbb3b90a5b7c",
    ),
    ("darwin", "arm64"): (
        "go1.22.12.darwin-arm64.tar.gz",
        "416c35218edb9d20990b5d8fc87be655d8b39926f15524ea35c66ee70273050d",
    ),
}


class GoToolchainError(RuntimeError):
    """Download, hash, unpack, or ``go build`` failed. Setup prints this."""


def host_key(
    system: str | None = None, machine: str | None = None
) -> tuple[str, str]:
    """Map ``platform.system`` / ``machine`` to a go.dev archive key."""
    sysname = (system or platform.system()).lower()
    mach = (machine or platform.machine()).lower()
    if sysname.startswith("win"):
        os_name = "windows"
    elif sysname == "darwin":
        os_name = "darwin"
    elif sysname == "linux":
        os_name = "linux"
    else:
        raise GoToolchainError(
            f"no portable Go archive for OS {sysname!r}; "
            "install Go 1.22+ or set DEEPFOLD_CHR_BIN"
        )
    if mach in ("amd64", "x86_64", "x64"):
        arch = "amd64"
    elif mach in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        raise GoToolchainError(
            f"no portable Go archive for arch {mach!r}; "
            "install Go 1.22+ or set DEEPFOLD_CHR_BIN"
        )
    if (os_name, arch) not in ARCHIVES:
        raise GoToolchainError(
            f"no portable Go archive for {os_name}/{arch}; "
            "install Go 1.22+ or set DEEPFOLD_CHR_BIN"
        )
    return os_name, arch


def go_version_at_least(text: str, major: int, minor: int) -> bool:
    """Parse ``go version go1.22.12 …`` from ``go version`` stdout."""
    match = re.search(r"go(\d+)\.(\d+)", text)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


def go_root(home: Path) -> Path:
    return home / "toolchains" / f"go{GO_VERSION}" / "go"


def _go_bin(root: Path) -> Path:
    name = "go.exe" if os.name == "nt" else "go"
    return root / "bin" / name


def _chr_name() -> str:
    return chr_exe_names()[0]


def _which_go() -> str | None:
    """Spy seam. Tests replace this."""
    return shutil.which("go")


def _go_version(go: str) -> str:
    """Spy seam. Tests replace this."""
    return subprocess.check_output(
        [go, "version"], text=True, stderr=subprocess.STDOUT
    )


def _download(url: str, dest: Path) -> None:
    """Spy seam. Tests replace this; live setup is the only caller."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as src, part.open("wb") as out:
            shutil.copyfileobj(src, out)
    except urllib.error.URLError as exc:
        part.unlink(missing_ok=True)
        raise GoToolchainError(
            f"download {url} failed ({exc}). "
            "Need network to go.dev, or install Go 1.22+, or set DEEPFOLD_CHR_BIN."
        ) from exc
    part.replace(dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(path: Path, expected: str) -> None:
    """Spy seam. Tests replace this."""
    got = _sha256(path)
    if got.lower() != expected.lower():
        path.unlink(missing_ok=True)
        raise GoToolchainError(
            f"SHA-256 mismatch for {path.name}: got {got}, expected {expected}"
        )


def _unpack(archive: Path, dest_parent: Path) -> None:
    """Unpack so ``dest_parent/go/bin/go`` exists. Spy seam."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    go_dir = dest_parent / "go"
    if go_dir.exists():
        shutil.rmtree(go_dir)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_parent)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(dest_parent, filter="data")
            else:
                tf.extractall(dest_parent)
    else:
        raise GoToolchainError(f"unknown Go archive type: {archive.name}")
    if not (dest_parent / "go").is_dir():
        raise GoToolchainError(f"archive {archive.name} has no go/ directory")


def _run(cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None) -> int:
    """Spy seam. Tests replace this."""
    return int(subprocess.call(cmd, cwd=cwd, env=env))


def _path_go() -> str | None:
    go = _which_go()
    if go is None:
        return None
    try:
        text = _go_version(go)
    except (OSError, subprocess.CalledProcessError):
        return None
    if not go_version_at_least(text, 1, 22):
        print(f"Go on PATH is too old ({text.strip()}); fetching {GO_VERSION}")
        return None
    return go


def _fetch_go(home: Path) -> Path:
    os_name, arch = host_key()
    filename, sha = ARCHIVES[(os_name, arch)]
    cache = home / "toolchains" / "cache" / filename
    if not cache.is_file():
        url = f"{GO_DL}/{filename}"
        print(f"downloading {url}", flush=True)
        _download(url, cache)
    _check_sha256(cache, sha)
    root = go_root(home)
    if not _go_bin(root).is_file():
        print(f"unpacking {filename}", flush=True)
        _unpack(cache, root.parent)
    go = _go_bin(root)
    if not go.is_file():
        raise GoToolchainError(f"unpacked Go has no {go}")
    if os.name != "nt":
        go.chmod(go.stat().st_mode | 0o111)
    return go


def resolve_go(home: Path) -> tuple[Path, bool]:
    """PATH Go 1.22+, else the unpacked toolchain, else download. ``bundled``."""
    path = _path_go()
    if path:
        return Path(path), False
    bundled = _go_bin(go_root(home))
    if bundled.is_file():
        return bundled, True
    return _fetch_go(home), True


def _build_env(go: Path, bundled: bool, home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CGO_ENABLED"] = "0"
    if bundled:
        env["GOROOT"] = str(go.parent.parent)
        env["GOPATH"] = str(home / "toolchains" / "gopath")
        env["GOCACHE"] = str(home / "toolchains" / "gocache")
    return env


def ensure_chr(*, home: Path | None = None, repo: Path | None = None) -> Path:
    """Return a ``chr`` binary, building it (and fetching Go) if needed."""
    existing = find_chr_bin()
    if existing is not None:
        return existing

    home = Path(home) if home is not None else deepfold_home()
    repo = Path(repo) if repo is not None else REPO
    dest = home / "bin" / _chr_name()
    go, bundled = resolve_go(home)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"building chr with {go} -> {dest}", flush=True)
    code = _run(
        [str(go), "build", "-o", str(dest), "./cmd/chr"],
        cwd=str(repo),
        env=_build_env(go, bundled, home),
    )
    if code != 0:
        raise GoToolchainError(
            f"go build chr failed (exit {code}). "
            "Need a checkout with ./cmd/chr, or set DEEPFOLD_CHR_BIN."
        )
    if not dest.is_file():
        raise GoToolchainError(f"go build reported success but {dest} is missing")
    if os.name != "nt":
        dest.chmod(dest.stat().st_mode | 0o111)

    repo_dest = repo / _chr_name()
    try:
        if repo_dest.resolve() != dest.resolve():
            shutil.copy2(dest, repo_dest)
    except OSError:
        pass
    return dest


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        path = ensure_chr()
    except GoToolchainError as exc:
        print(f"go_toolchain: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
