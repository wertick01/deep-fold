"""GPU-free: Linux ``.so`` is a first-class extension binary.

    python -m gpu.test_ext_bin
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.ext_bin import abi_tag, find_ext, list_ext, matches_abi  # noqa: E402

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        _FAILS.append(msg)


def main() -> int:
    tag = abi_tag()
    posix = f"cpython-{tag[2:]}"
    with tempfile.TemporaryDirectory(prefix="ext-bin-") as tmp:
        root = Path(tmp)
        so = root / f"chr_nf4_ext.{tag}-linux_x86_64.so"
        pyd = root / f"chr_nf4_ext.{tag}-win_amd64.pyd"
        linux = root / f"chr_nf4_ext.{posix}-x86_64-linux-gnu.so"
        wrong = root / "chr_nf4_ext.cp399-linux_x86_64.so"
        so.write_bytes(b"")
        pyd.write_bytes(b"")
        linux.write_bytes(b"")
        wrong.write_bytes(b"")
        listed = [p.name for p in list_ext(root, "chr_nf4_ext")]
        hits = [p.name for p in find_ext(root, "chr_nf4_ext")]
        check(so.name in listed, "list_ext finds .so")
        check(pyd.name in listed, "list_ext finds .pyd")
        check(wrong.name in listed, "list_ext keeps foreign ABI")
        check(wrong.name not in hits, "find_ext drops foreign ABI")
        check(all(Path(n).suffix in (".pyd", ".so") for n in hits), f"native hits {hits}")
        if sys.platform.startswith("win"):
            check(pyd.name in hits, "Windows find_ext prefers .pyd")
            check(so.name not in hits, "Windows find_ext skips .so")
        else:
            check(so.name in hits, "Linux find_ext prefers .so")
            check(pyd.name not in hits, "Linux find_ext skips .pyd")
            check(linux.name in hits, "Linux cpython-3xx tag is importable")
        check(matches_abi(linux.name, tag), "cpython-311 matches cp311")
        check(not matches_abi(wrong.name, tag), "cp399 does not match this Python")

    check("find_ext" in Path(_REPO / "gpu" / "nf4" / "__init__.py").read_text(encoding="utf-8"),
          "nf4 wrapper uses find_ext")
    check("find_ext" in Path(_REPO / "gpu" / "vq" / "__init__.py").read_text(encoding="utf-8"),
          "vq wrapper uses find_ext")
    nf4_src = Path(_REPO / "gpu" / "nf4" / "__init__.py").read_text(encoding="utf-8")
    check("have_host_compiler" in nf4_src, "nf4 JIT does not always call inject_msvc_env")
    check("nvcc/g++" in nf4_src, "nf4 missing-ext copy names POSIX compilers")

    if _FAILS:
        print(f"\n{len(_FAILS)} FAIL")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
