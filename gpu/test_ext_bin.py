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

from gpu.ext_bin import find_ext  # noqa: E402

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        _FAILS.append(msg)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ext-bin-") as tmp:
        root = Path(tmp)
        so = root / "chr_nf4_ext.cp311-linux_x86_64.so"
        pyd = root / "chr_nf4_ext.cp311-win_amd64.pyd"
        so.write_bytes(b"")
        pyd.write_bytes(b"")
        hits = find_ext(root, "chr_nf4_ext")
        names = [p.name for p in hits]
        check(so.name in names, "glob finds .so")
        check(pyd.name in names, "glob finds .pyd")
        check(len(hits) == 2, f"two artifacts {names}")

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
