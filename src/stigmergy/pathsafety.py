"""Path safety library (SPEC.md §4 "Path safety").

Task-pack assembly and artifact collection are **host-side operations over
worker-influenced paths** that run *before* container isolation can help
(SPEC §4) — a compromised worker (or a hostile upstream artifact) can supply
paths and archives crafted to escape the intended root. This module is the
primary trust boundary for that class of attack: classic `..` traversal,
absolute-path injection, symlink escape (a path component that resolves
outside the root), special files (fifo/device/socket) used to block or
trick host-side copying, and archive bombs (zip-slip entries, oversized/
over-deep/over-numerous tar members).

Three primitives, deliberately small and composable:

- :func:`resolve_beneath` — canonicalize a worker-supplied relative path
  beneath a trusted root, with no symlink-following-out and no `..`
  tolerated anywhere in the path, even if the final resolved location would
  land back inside the root. The check is on path *components*, not just
  the destination.
- :func:`reject_special` — refuse fifo/socket/block/char special files
  (`/dev/null`-style tricks) so host-side copying/reading never blocks on
  or is fooled by a non-regular file.
- :func:`safe_extract` — bounded, validated tar extraction: every member is
  checked for zip-slip (absolute name, `..` component), special-file type,
  symlink/hardlink escape, and archive-bomb limits (cumulative size, entry
  count, path depth) *before* anything is written. A single violation
  anywhere in the archive aborts the whole extraction — nothing (or at
  most work already capped below the limit) reaches disk from a rejected
  archive.

Fail-closed throughout: any ambiguity or error resolving a path or member
raises :class:`PathSafetyError` rather than silently allowing it through.
"""

from __future__ import annotations

import os
import stat
import tarfile
from pathlib import Path, PurePosixPath

# Tar member types that are safe to materialize on the host once validated.
# Deliberately an allowlist, not a blocklist: anything not explicitly named
# here (CHRTYPE, BLKTYPE, FIFOTYPE, LNKTYPE hard links, GNU sparse/long-name
# types, CONTTYPE, ...) is rejected by construction rather than by trying to
# enumerate every dangerous type. Hard links (LNKTYPE) are deliberately
# excluded: tar resolves a hard-link `linkname` relative to the *archive
# root*, not the member's own directory, which does not match the
# member's-directory-relative containment check used for symlinks below —
# rather than build and carry untested dual semantics for a type no test
# exercises, hard links are rejected outright (fail closed).
_SAFE_TAR_TYPES = frozenset({tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE})


class PathSafetyError(Exception):
    """Raised on any path-safety violation: traversal, absolute path,
    symlink escape, special file, or archive-bomb limit breach."""


def resolve_beneath(root: str | os.PathLike[str], candidate: str | os.PathLike[str]) -> Path:
    """Resolve ``candidate`` (relative) beneath ``root``, or raise.

    Rejects, in order:

    - ``candidate`` being absolute.
    - any ``..`` path *component* in ``candidate`` — checked on the
      components themselves, not just where the path ends up. A path like
      ``a/../../root/a`` that happens to resolve back inside ``root`` is
      still rejected: tolerating "it lands back inside" would let a worker
      walk out through a symlinked or bind-mounted ancestor and back in via
      a different route, and the component check is what closes that.
    - the real path (following any symlinks, via `os.path.realpath`)
      landing outside ``root`` — a symlink *inside* root whose target
      points outside root must not be followed there.

    Returns the resolved absolute :class:`Path` on success.
    """
    root_path = Path(root)
    candidate_path = Path(candidate)

    if candidate_path.is_absolute():
        raise PathSafetyError(f"candidate path must be relative, got absolute: {candidate!r}")

    if any(part == ".." for part in candidate_path.parts):
        raise PathSafetyError(f"candidate path must not contain '..' components: {candidate!r}")

    root_real = Path(os.path.realpath(root_path))
    joined = root_path / candidate_path
    resolved_real = Path(os.path.realpath(joined))

    if resolved_real != root_real and not resolved_real.is_relative_to(root_real):
        raise PathSafetyError(
            f"candidate path escapes root via symlink or otherwise: {candidate!r} "
            f"resolves to {resolved_real} outside {root_real}"
        )

    return resolved_real


def reject_special(path: str | os.PathLike[str]) -> None:
    """Raise :class:`PathSafetyError` if ``path`` is a special file.

    Special = fifo, socket, block device, or character device
    (`os.mkfifo`-created files, `/dev/null`-style device nodes, unix
    sockets). Regular files and directories pass silently. Uses
    `os.lstat` so a symlink *pointing at* a special file is judged by
    what it points to would resolve to being examined at the call site —
    callers that need no-symlink-following semantics should resolve the
    path (e.g. via :func:`resolve_beneath`) before calling this.
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        raise PathSafetyError(f"cannot stat {path!r}: {exc}") from exc

    mode = st.st_mode
    if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
        raise PathSafetyError(f"special file not allowed: {path!r}")


def _posix_parts(name: str) -> tuple[str, ...]:
    """Split a tar member name on `/` (tar names are always POSIX-style,
    regardless of host platform — never use `pathlib.Path` for this, it
    would apply host path semantics to an archive-supplied string)."""
    return PurePosixPath(name).parts


def _validate_member_path(name: str, dest_real: Path, *, max_depth: int) -> Path:
    """Validate one member's ``name`` (or a symlink/hardlink ``linkname``)
    for zip-slip and depth, returning the resolved-beneath-dest path.

    Rejects an absolute name outright and any ``..`` component, then
    requires the joined-and-realpath'd result stay beneath ``dest_real``
    (mirrors :func:`resolve_beneath`, reimplemented against a
    `PurePosixPath` split since tar member names are archive-relative
    strings, not host `Path` objects, and the destination directory tree
    does not exist yet during validation — the joined path's parent
    directories may still need to be created).
    """
    parts = _posix_parts(name)
    if PurePosixPath(name).is_absolute():
        raise PathSafetyError(f"tar member has absolute path: {name!r}")
    if any(part == ".." for part in parts):
        raise PathSafetyError(f"tar member path contains '..': {name!r}")
    if len(parts) > max_depth:
        raise PathSafetyError(
            f"tar member path depth {len(parts)} exceeds max_depth {max_depth}: {name!r}"
        )

    joined = dest_real / Path(*parts) if parts else dest_real
    # The member's on-disk target does not exist yet, so realpath only
    # resolves symlinks in components that already exist (i.e. within
    # dest_real itself); the member's own not-yet-created components are
    # never symlinks at this point, so this cannot be tricked by the
    # archive under validation.
    resolved = Path(os.path.realpath(joined))
    if resolved != dest_real and not resolved.is_relative_to(dest_real):
        raise PathSafetyError(f"tar member escapes destination: {name!r}")
    return resolved


def _validate_link_target(member: tarfile.TarInfo, dest_real: Path) -> None:
    """Validate a symlink member's target stays beneath ``dest_real``.

    The link target is resolved relative to the member's own directory
    (real symlink semantics), then required to land beneath ``dest_real``.
    Any ``..`` component in the target is rejected outright — strictly
    stronger than "only reject if it actually escapes", but provably safe:
    since no ``..`` is tolerated, the target can only ever descend from the
    member's directory, so containment beneath ``dest_real`` holds without
    needing to further resolve where a partially-escaping-and-returning
    target would land.
    """
    linkname = member.linkname
    if not linkname:
        raise PathSafetyError(f"tar symlink member has empty linkname: {member.name!r}")

    link_parts = PurePosixPath(linkname).parts
    if PurePosixPath(linkname).is_absolute():
        raise PathSafetyError(
            f"tar symlink member {member.name!r} has absolute target: {linkname!r}"
        )
    if any(part == ".." for part in link_parts):
        raise PathSafetyError(
            f"tar symlink member {member.name!r} target contains '..': {linkname!r}"
        )

    member_dir_parts = _posix_parts(member.name)[:-1]
    target_joined = (
        dest_real / Path(*member_dir_parts, *link_parts)
        if (member_dir_parts or link_parts)
        else dest_real
    )
    resolved = Path(os.path.realpath(target_joined))
    if resolved != dest_real and not resolved.is_relative_to(dest_real):
        raise PathSafetyError(
            f"tar symlink member {member.name!r} target escapes destination: {linkname!r}"
        )


def safe_extract(
    archive: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    max_total_bytes: int,
    max_entries: int,
    max_depth: int,
) -> None:
    """Extract a tar archive into ``dest`` with tar-bomb / zip-slip defenses.

    Every member is validated *before* anything is extracted; the first
    violation raises :class:`PathSafetyError` and nothing from the archive
    is written. Rejected, for any member:

    - absolute name or any ``..`` component (zip-slip).
    - a type other than regular file, directory, or symlink (special files
      — fifo/device/socket — and hard links are never materialized).
    - a symlink whose target (resolved relative to the member's directory)
      escapes ``dest``.
    - cumulative uncompressed size (sum of `member.size`) exceeding
      ``max_total_bytes``.
    - entry count exceeding ``max_entries``.
    - path depth (component count) exceeding ``max_depth``.

    Only after every member in the archive validates does extraction
    happen: directories are created, regular files are written, and
    in-bounds symlinks are created.
    """
    dest_path = Path(dest)
    dest_real = Path(os.path.realpath(dest_path))

    with tarfile.open(archive, mode="r") as tar:
        validated: list[tarfile.TarInfo] = []
        total_size = 0
        entry_count = 0

        for member in tar:
            entry_count += 1
            if entry_count > max_entries:
                raise PathSafetyError(
                    f"tar archive exceeds max_entries {max_entries}: {archive!r}"
                )

            if member.isreg():
                total_size += member.size
                if total_size > max_total_bytes:
                    raise PathSafetyError(
                        f"tar archive exceeds max_total_bytes {max_total_bytes}: {archive!r}"
                    )
            elif member.size:
                # Non-regular members should carry no payload; still count
                # any declared size toward the cap defensively.
                total_size += member.size
                if total_size > max_total_bytes:
                    raise PathSafetyError(
                        f"tar archive exceeds max_total_bytes {max_total_bytes}: {archive!r}"
                    )

            if member.type not in _SAFE_TAR_TYPES:
                raise PathSafetyError(
                    f"tar member has disallowed type {member.type!r}: {member.name!r}"
                )

            _validate_member_path(member.name, dest_real, max_depth=max_depth)

            if member.issym():
                _validate_link_target(member, dest_real)

            validated.append(member)

        # Second pass: nothing on disk changes between validation and
        # extraction (the archive is read-only, `dest` untouched so far),
        # so no TOCTOU window opens here. Directories first, then files,
        # then symlinks, so parent directories exist before their contents.
        for member in validated:
            if member.isdir():
                target = _validate_member_path(member.name, dest_real, max_depth=max_depth)
                target.mkdir(parents=True, exist_ok=True)

        for member in validated:
            if member.isreg():
                target = _validate_member_path(member.name, dest_real, max_depth=max_depth)
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise PathSafetyError(
                        f"tar member could not be read as a regular file: {member.name!r}"
                    )
                with extracted, open(target, "wb") as out:
                    out.write(extracted.read())

        for member in validated:
            if member.issym():
                target = _validate_member_path(member.name, dest_real, max_depth=max_depth)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise PathSafetyError(
                        f"tar symlink member target already exists: {member.name!r}"
                    )
                os.symlink(member.linkname, target)
