"""Adversarial tests for the path-safety library (SPEC.md §4 "Path safety").

Authored by the orchestrator, not the implementor. Task-pack assembly and
artifact collection are **host-side operations over worker-influenced paths**
that run BEFORE container isolation can help (SPEC §4) — so this is a
primary trust boundary, and these assertions are its fixed spec. The
implementation in `stigmergy.pathsafety` must satisfy every one without
weakening.

Threat model exercised here: a compromised worker (or hostile upstream
artifact) supplies paths/archives designed to escape the intended root —
classic path traversal (`..`), absolute-path injection, symlink escape
(the resolved target leaves the root), special files (fifo/device) used to
block or trick host-side copying, and archive bombs (zip-slip entries,
oversized/over-deep/over-many entries).
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from stigmergy.pathsafety import (
    PathSafetyError,
    reject_special,
    resolve_beneath,
    safe_extract,
)

# --------------------------------------------------------------------------
# resolve_beneath — canonicalize beneath root, no escape, no symlink-following-out
# --------------------------------------------------------------------------


def test_resolve_beneath_allows_normal_relative_path(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    target = root / "sub" / "file.txt"
    target.write_text("ok")
    resolved = resolve_beneath(root, "sub/file.txt")
    assert resolved == target.resolve()
    assert str(resolved).startswith(str(root.resolve()))


def test_resolve_beneath_rejects_absolute_path(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(PathSafetyError):
        resolve_beneath(root, "/etc/passwd")


def test_resolve_beneath_rejects_dotdot_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(PathSafetyError):
        resolve_beneath(root, "../../etc/passwd")


def test_resolve_beneath_rejects_dotdot_even_if_lands_back_inside(tmp_path):
    # A path that uses .. to leave and come back must still be rejected — the
    # check is on the path components, not just the final location.
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    with pytest.raises(PathSafetyError):
        resolve_beneath(root, "a/../../root/a")


def test_resolve_beneath_rejects_symlink_escaping_root(tmp_path):
    # A symlink INSIDE root that points OUTSIDE root must not be followed to
    # an out-of-root target.
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("secret")
    link = root / "escape"
    link.symlink_to(outside)
    with pytest.raises(PathSafetyError):
        resolve_beneath(root, "escape/secret")


def test_resolve_beneath_rejects_symlink_to_absolute_system_path(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    link = root / "etc"
    link.symlink_to("/etc")
    with pytest.raises(PathSafetyError):
        resolve_beneath(root, "etc/passwd")


# --------------------------------------------------------------------------
# reject_special — no fifo / device / socket files
# --------------------------------------------------------------------------


def test_reject_special_passes_regular_file(tmp_path):
    f = tmp_path / "regular"
    f.write_text("x")
    reject_special(f)  # must not raise


def test_reject_special_passes_directory(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    reject_special(d)  # must not raise


def test_reject_special_rejects_fifo(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(PathSafetyError):
        reject_special(fifo)


def test_reject_special_rejects_device_file():
    # /dev/null is a character device — a special file.
    with pytest.raises(PathSafetyError):
        reject_special(Path("/dev/null"))


# --------------------------------------------------------------------------
# safe_extract — zip-slip / tar-bomb defenses (bounded expansion, SPEC §4)
# --------------------------------------------------------------------------


def _tar_bytes(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> io.BytesIO:
    """Build an in-memory tar from (TarInfo, data) pairs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for info, data in members:
            if data is not None:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)
    buf.seek(0)
    return buf


def _write_tar(path: Path, members):
    path.write_bytes(_tar_bytes(members).getvalue())


_LIMITS = dict(max_total_bytes=1_000_000, max_entries=100, max_depth=8)


def test_safe_extract_accepts_benign_archive(tmp_path):
    archive = tmp_path / "ok.tar"
    _write_tar(archive, [(tarfile.TarInfo("a/b.txt"), b"hello")])
    dest = tmp_path / "dest"
    dest.mkdir()
    safe_extract(archive, dest, **_LIMITS)
    assert (dest / "a" / "b.txt").read_text() == "hello"


def test_safe_extract_rejects_absolute_member(tmp_path):
    archive = tmp_path / "abs.tar"
    _write_tar(archive, [(tarfile.TarInfo("/etc/evil"), b"x")])
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(PathSafetyError):
        safe_extract(archive, dest, **_LIMITS)


def test_safe_extract_rejects_dotdot_member(tmp_path):
    archive = tmp_path / "dd.tar"
    _write_tar(archive, [(tarfile.TarInfo("../../etc/evil"), b"x")])
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(PathSafetyError):
        safe_extract(archive, dest, **_LIMITS)


def test_safe_extract_rejects_symlink_escape_member(tmp_path):
    # A symlink member whose target escapes dest (zip-slip via symlink).
    info = tarfile.TarInfo("link")
    info.type = tarfile.SYMTYPE
    info.linkname = "../../../../etc/passwd"
    archive = tmp_path / "sym.tar"
    _write_tar(archive, [(info, None)])
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(PathSafetyError):
        safe_extract(archive, dest, **_LIMITS)


def test_safe_extract_rejects_special_file_member(tmp_path):
    # A device/fifo member must be rejected (never materialized on the host).
    info = tarfile.TarInfo("dev")
    info.type = tarfile.FIFOTYPE
    archive = tmp_path / "fifo.tar"
    _write_tar(archive, [(info, None)])
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(PathSafetyError):
        safe_extract(archive, dest, **_LIMITS)


def test_safe_extract_enforces_total_size_cap(tmp_path):
    # Sum of member sizes exceeding max_total_bytes → reject (tar bomb).
    big = b"x" * 2_000_000
    archive = tmp_path / "big.tar"
    _write_tar(archive, [(tarfile.TarInfo("big.bin"), big)])
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(PathSafetyError):
        safe_extract(archive, dest, max_total_bytes=1_000_000, max_entries=100, max_depth=8)
    # nothing (or at least not the full bomb) was written past the cap
    assert not (dest / "big.bin").exists() or (dest / "big.bin").stat().st_size <= 1_000_000


def test_safe_extract_enforces_entry_count_cap(tmp_path):
    members = [(tarfile.TarInfo(f"f{i}.txt"), b"x") for i in range(50)]
    archive = tmp_path / "many.tar"
    _write_tar(archive, members)
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(PathSafetyError):
        safe_extract(archive, dest, max_total_bytes=1_000_000, max_entries=10, max_depth=8)


def test_safe_extract_enforces_depth_cap(tmp_path):
    deep = "/".join(f"d{i}" for i in range(20)) + "/f.txt"
    archive = tmp_path / "deep.tar"
    _write_tar(archive, [(tarfile.TarInfo(deep), b"x")])
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(PathSafetyError):
        safe_extract(archive, dest, max_total_bytes=1_000_000, max_entries=100, max_depth=8)
