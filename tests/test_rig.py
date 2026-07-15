"""Tests for stigmergy.rig (SPEC.md §3 `provision` station, §3 rig object,
§6 bead work-order contract).

Design decision under test: the rig's bead + loop-state store is a
loop-owned, self-contained SQLite database (`beads.db`, stdlib `sqlite3`)
— never the `bd` issue tracker. `create_rig` scaffolds the full rig
directory tree, validates the charter BEFORE creating anything, and is
all-or-nothing: any failure (invalid charter, failed git clone) leaves no
partial rig directory behind.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from stigmergy import __version__
from stigmergy.charter import CharterError, load_charter
from stigmergy.cli import main
from stigmergy.rig import RigError, RigStore, create_rig

FIXTURES = Path(__file__).parent / "fixtures"
VALID_CHARTER_PATH = FIXTURES / "charter_valid.toml"
MODELS_REGISTRY_PATH = FIXTURES / "models.toml"

BASE_CHARTER_TOML = VALID_CHARTER_PATH.read_text()


def make_local_repo(tmp_path: Path) -> Path:
    """Create a minimal local git repo with one commit; return its path."""
    repo_dir = tmp_path / "source_repo"
    repo_dir.mkdir()
    env_cfg = [
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Test User",
    ]
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    (repo_dir / "README.md").write_text("hello from the fixture repo\n")
    subprocess.run(["git", *env_cfg, "-C", str(repo_dir), "add", "README.md"], check=True)
    subprocess.run(
        ["git", *env_cfg, "-C", str(repo_dir), "commit", "-q", "-m", "initial commit"],
        check=True,
    )
    return repo_dir


def make_charter(tmp_path: Path, repo: Path | str, content: str | None = None) -> Path:
    """Write a charter.toml (+ copied models.toml) pointed at ``repo``."""
    charter_dir = tmp_path / "charter_src"
    charter_dir.mkdir(exist_ok=True)
    text = BASE_CHARTER_TOML if content is None else content
    text = text.replace('repo = "path-or-url"', f'repo = "{repo}"')
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(text)
    shutil.copy(MODELS_REGISTRY_PATH, charter_dir / "models.toml")
    return charter_path


def mutate(old: str, new: str) -> str:
    assert BASE_CHARTER_TOML.count(old) == 1, f"expected exactly one occurrence of {old!r}"
    return BASE_CHARTER_TOML.replace(old, new)


# --- case 1: full directory structure --------------------------------------


def test_create_rig_produces_full_directory_structure(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    rig_root = create_rig(charter_path, base_dir=base_dir)

    assert rig_root == base_dir / "shipyard"
    assert (rig_root / "charter.toml").is_file()
    assert (rig_root / "models.toml").is_file()
    assert (rig_root / "beads.db").is_file()
    assert (rig_root / "repo").is_dir()
    assert (rig_root / "repo" / ".git").is_dir()
    assert (rig_root / "context").is_dir()
    assert (rig_root / "records").is_dir()
    assert (rig_root / "images" / "worker" / "Containerfile").is_file()
    assert (rig_root / "clones").is_dir()


# --- case 2: copied charter reloads cleanly against rig-local models.toml --


def test_copied_charter_reloads_against_rig_local_registry(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    rig_root = create_rig(charter_path, base_dir=base_dir)

    copied_charter = load_charter(rig_root / "charter.toml", env={})
    assert copied_charter.raw["models"]["registry"] == "models.toml"
    # The registry it resolves against must be the rig-local copy, not the
    # original fixture path (which may not even exist relative to the rig).
    assert (rig_root / "models.toml").read_bytes() == MODELS_REGISTRY_PATH.read_bytes()


# --- case 3: rig_meta populated ---------------------------------------------


def test_copied_charter_rewrites_registry_path_when_source_differs(tmp_path: Path) -> None:
    """If the source charter's [models].registry points somewhere other than
    'models.toml', the copy in the rig must be rewritten so it still
    resolves to the rig-local copy — the rig stays self-contained."""
    repo = make_local_repo(tmp_path)
    charter_dir = tmp_path / "charter_src2"
    charter_dir.mkdir()
    nested = charter_dir / "nested"
    nested.mkdir()
    shutil.copy(MODELS_REGISTRY_PATH, nested / "registry.toml")

    content = BASE_CHARTER_TOML.replace(
        'repo = "path-or-url"', f'repo = "{repo}"'
    ).replace('registry = "models.toml"', 'registry = "nested/registry.toml"')
    charter_path = charter_dir / "charter.toml"
    charter_path.write_text(content)

    base_dir = tmp_path / "rigs"
    rig_root = create_rig(charter_path, base_dir=base_dir)

    assert (rig_root / "models.toml").is_file()
    copied_charter = load_charter(rig_root / "charter.toml", env={})
    assert copied_charter.raw["models"]["registry"] == "models.toml"


def test_rig_meta_populated(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    source_charter = load_charter(charter_path, env={})
    rig_root = create_rig(charter_path, base_dir=base_dir)

    store = RigStore(rig_root / "beads.db")
    try:
        assert store.get_meta("schema_version") == "1"
        assert store.get_meta("stigmergy_version") == __version__
        charter_hash = store.get_meta("charter_hash")
        assert charter_hash
        assert charter_hash == source_charter.resolved_hash
        assert store.get_meta("rig_name") == "shipyard"
    finally:
        store.close()


# --- case 4: invalid charter -> no partial state ----------------------------


def test_invalid_charter_raises_and_leaves_no_rig_dir(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    content = mutate("workers = 1", "workers = 2")
    charter_path = make_charter(tmp_path, repo, content=content)
    base_dir = tmp_path / "rigs"

    with pytest.raises(CharterError):
        create_rig(charter_path, base_dir=base_dir)

    assert not (base_dir / "shipyard").exists()
    # base_dir itself may or may not exist, but no partial rig content.
    if base_dir.exists():
        assert list(base_dir.iterdir()) == []


def test_cli_rig_new_invalid_charter_exits_nonzero(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    content = mutate("workers = 1", "workers = 2")
    charter_path = make_charter(tmp_path, repo, content=content)
    base_dir = tmp_path / "rigs"

    exit_code = main(["rig", "new", "--charter", str(charter_path), "--path", str(base_dir)])

    assert exit_code != 0
    assert not (base_dir / "shipyard").exists()


# --- case 5: local git repo clone -------------------------------------------


def test_repo_clone_from_local_git_repo(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    rig_root = create_rig(charter_path, base_dir=base_dir)

    cloned_readme = rig_root / "repo" / "README.md"
    assert cloned_readme.is_file()
    assert cloned_readme.read_text() == "hello from the fixture repo\n"


def test_repo_clone_failure_cleans_up_partial_rig(tmp_path: Path) -> None:
    nonexistent_repo = tmp_path / "does-not-exist"
    charter_path = make_charter(tmp_path, nonexistent_repo)
    base_dir = tmp_path / "rigs"

    with pytest.raises(RigError):
        create_rig(charter_path, base_dir=base_dir)

    assert not (base_dir / "shipyard").exists()


# --- case 6: RigStore add_bead / get_bead round-trip ------------------------


def test_rigstore_add_and_get_bead_round_trips(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "beads.db")
    try:
        store.add_bead(
            id="bead-1",
            title="Do the thing",
            goal="Ship the feature end to end.",
            required_reading=["context/architecture.md", "repo/src/foo.py"],
            target_scope=["src/foo.py", "tests/test_foo.py"],
            work_product="A passing implementation of foo().",
            acceptance_criteria=["foo() returns 42", "foo() raises on negative input"],
            tier1_checks={"named": ["pytest", "lint"], "paths": ["tests/test_foo.py"]},
            difficulty="medium",
            lane_hint="default",
        )

        bead = store.get_bead("bead-1")
        assert bead is not None
        assert bead["id"] == "bead-1"
        assert bead["title"] == "Do the thing"
        assert bead["goal"] == "Ship the feature end to end."
        assert bead["required_reading"] == ["context/architecture.md", "repo/src/foo.py"]
        assert bead["target_scope"] == ["src/foo.py", "tests/test_foo.py"]
        assert bead["work_product"] == "A passing implementation of foo()."
        assert bead["acceptance_criteria"] == [
            "foo() returns 42",
            "foo() raises on negative input",
        ]
        assert bead["tier1_checks"] == {"named": ["pytest", "lint"], "paths": ["tests/test_foo.py"]}
        assert bead["difficulty"] == "medium"
        assert bead["lane_hint"] == "default"

        # defaults
        assert bead["approved"] == 0
        assert bead["state"] == "pool"
        assert bead["attempts_used"] == 0
        assert bead["integration_failures"] == 0
        assert bead["rubric_only"] == 0
    finally:
        store.close()


def test_rigstore_get_bead_missing_returns_none(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "beads.db")
    try:
        assert store.get_bead("nonexistent") is None
    finally:
        store.close()


# --- case 7: add_dep / deps_of + list_beads(state=) filter ------------------


def test_rigstore_add_dep_and_deps_of_round_trip(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "beads.db")
    try:
        store.add_bead(id="bead-a", title="A")
        store.add_bead(id="bead-b", title="B")
        store.add_bead(id="bead-c", title="C")

        store.add_dep("bead-c", "bead-a")
        store.add_dep("bead-c", "bead-b")

        deps = store.deps_of("bead-c")
        assert sorted(deps) == ["bead-a", "bead-b"]
        assert store.deps_of("bead-a") == []
    finally:
        store.close()


def test_rigstore_list_beads_filters_by_state(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "beads.db")
    try:
        store.add_bead(id="bead-pool", title="pool one")
        store.add_bead(id="bead-claimed", title="claimed one", state="claimed")
        store.add_bead(id="bead-landed", title="landed one", state="landed")

        pool_beads = store.list_beads(state="pool")
        assert [b["id"] for b in pool_beads] == ["bead-pool"]

        claimed_beads = store.list_beads(state="claimed")
        assert [b["id"] for b in claimed_beads] == ["bead-claimed"]

        all_beads = store.list_beads()
        assert {b["id"] for b in all_beads} == {"bead-pool", "bead-claimed", "bead-landed"}
    finally:
        store.close()


# --- case 8: CLI dispatch exit codes ----------------------------------------


def test_cli_rig_new_valid_charter_returns_zero(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    exit_code = main(["rig", "new", "--charter", str(charter_path), "--path", str(base_dir)])

    assert exit_code == 0
    assert (base_dir / "shipyard" / "beads.db").is_file()


def test_cli_rig_new_bad_charter_path_returns_nonzero(tmp_path: Path) -> None:
    base_dir = tmp_path / "rigs"
    exit_code = main(
        [
            "rig",
            "new",
            "--charter",
            str(tmp_path / "does-not-exist.toml"),
            "--path",
            str(base_dir),
        ]
    )
    assert exit_code != 0
    assert not base_dir.exists()


def test_main_bare_invocation_returns_zero() -> None:
    assert main([]) == 0


# --- case 9: DB isolation — plain stdlib sqlite3, no bd/dolt ----------------


def test_beads_db_is_self_contained_plain_sqlite(tmp_path: Path) -> None:
    """Open beads.db fresh in a second stdlib sqlite3 connection (independent
    of RigStore) and read a rig_meta value straight out of the file — the
    rig's data plane is a portable SQLite file, not an external DB service."""
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    rig_root = create_rig(charter_path, base_dir=base_dir)
    db_path = rig_root / "beads.db"
    assert db_path.is_file()

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT value FROM rig_meta WHERE key = 'schema_version'").fetchall()
        assert rows == [("1",)]
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"beads", "bead_deps", "rig_meta"} <= tables
    finally:
        conn.close()
