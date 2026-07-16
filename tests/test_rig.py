"""Tests for stigmergy.rig (SPEC.md §3 `provision` station, §3 rig object,
§6 ticket work-order contract).

Design decision under test: the rig's ticket + loop-state store is a
loop-owned, self-contained SQLite database (`tickets.db`, stdlib `sqlite3`)
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
from stigmergy.charter import Charter, CharterError, load_charter
from stigmergy.cli import main
from stigmergy.daemon import _REQUIRED_RIG_PATH_KEYS
from stigmergy.registry import Registry, UnbudgetableError
from stigmergy.rig import ResolvedRig, RigError, RigStore, create_rig, resolve_rig

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
    assert (rig_root / "tickets.db").is_file()
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

    store = RigStore(rig_root / "tickets.db")
    try:
        assert store.get_meta("schema_version") == "2"
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


# --- case 6: RigStore add_ticket / get_ticket round-trip ------------------------


def test_rigstore_add_and_get_ticket_round_trips(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_ticket(
            id="ticket-1",
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

        ticket = store.get_ticket("ticket-1")
        assert ticket is not None
        assert ticket["id"] == "ticket-1"
        assert ticket["title"] == "Do the thing"
        assert ticket["goal"] == "Ship the feature end to end."
        assert ticket["required_reading"] == ["context/architecture.md", "repo/src/foo.py"]
        assert ticket["target_scope"] == ["src/foo.py", "tests/test_foo.py"]
        assert ticket["work_product"] == "A passing implementation of foo()."
        assert ticket["acceptance_criteria"] == [
            "foo() returns 42",
            "foo() raises on negative input",
        ]
        assert ticket["tier1_checks"] == {
            "named": ["pytest", "lint"],
            "paths": ["tests/test_foo.py"],
        }
        assert ticket["difficulty"] == "medium"
        assert ticket["lane_hint"] == "default"

        # defaults
        assert ticket["approved"] == 0
        assert ticket["state"] == "pool"
        assert ticket["attempts_used"] == 0
        assert ticket["integration_failures"] == 0
        assert ticket["rubric_only"] == 0
    finally:
        store.close()


def test_rigstore_get_ticket_missing_returns_none(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        assert store.get_ticket("nonexistent") is None
    finally:
        store.close()


# --- case 7: add_dep / deps_of + list_tickets(state=) filter ------------------


def test_rigstore_add_dep_and_deps_of_round_trip(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_ticket(id="ticket-a", title="A")
        store.add_ticket(id="ticket-b", title="B")
        store.add_ticket(id="ticket-c", title="C")

        store.add_dep("ticket-c", "ticket-a")
        store.add_dep("ticket-c", "ticket-b")

        deps = store.deps_of("ticket-c")
        assert sorted(deps) == ["ticket-a", "ticket-b"]
        assert store.deps_of("ticket-a") == []
    finally:
        store.close()


def test_rigstore_list_tickets_filters_by_state(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_ticket(id="ticket-pool", title="pool one")
        store.add_ticket(id="ticket-claimed", title="claimed one", state="claimed")
        store.add_ticket(id="ticket-landed", title="landed one", state="landed")

        pool_tickets = store.list_tickets(state="pool")
        assert [b["id"] for b in pool_tickets] == ["ticket-pool"]

        claimed_tickets = store.list_tickets(state="claimed")
        assert [b["id"] for b in claimed_tickets] == ["ticket-claimed"]

        all_tickets = store.list_tickets()
        assert {b["id"] for b in all_tickets} == {"ticket-pool", "ticket-claimed", "ticket-landed"}
    finally:
        store.close()


# --- case 8: CLI dispatch exit codes ----------------------------------------


def test_cli_rig_new_valid_charter_returns_zero(tmp_path: Path) -> None:
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    exit_code = main(["rig", "new", "--charter", str(charter_path), "--path", str(base_dir)])

    assert exit_code == 0
    assert (base_dir / "shipyard" / "tickets.db").is_file()


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


def test_tickets_db_is_self_contained_plain_sqlite(tmp_path: Path) -> None:
    """Open tickets.db fresh in a second stdlib sqlite3 connection (independent
    of RigStore) and read a rig_meta value straight out of the file — the
    rig's data plane is a portable SQLite file, not an external DB service."""
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    base_dir = tmp_path / "rigs"

    rig_root = create_rig(charter_path, base_dir=base_dir)
    db_path = rig_root / "tickets.db"
    assert db_path.is_file()

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT value FROM rig_meta WHERE key = 'schema_version'").fetchall()
        assert rows == [("2",)]
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"tickets", "ticket_deps", "rig_meta", "worker_names"} <= tables
    finally:
        conn.close()


# ==========================================================================
# resolve_rig / ResolvedRig  (bead .27 build spec §1, frozen case list §4
# cases 1-12) — the read-side inverse of create_rig.
#
# Fail-closed / error-path cases scaffold a rig dir directly (no git clone):
# resolve_rig never validates repo-ness, so the four artifacts it gates on
# (rig_root dir + charter.toml + models.toml + tickets.db) are all it needs.
# The happy-path / prompts-readback cases go through the REAL create_rig
# (real local git clone) so repo/prompts/{code01,critic01} actually exist.
# ==========================================================================


def make_local_repo_with_prompts(tmp_path: Path, name: str = "source_repo") -> Path:
    """Like make_local_repo, but the working tree carries a versioned
    ``prompts/`` dir with ``code01``/``critic01`` template files — so a rig
    cloned from it has readable prompt artifacts under ``repo/prompts/``
    (build spec §1.1: prompts_dir resolves relative to repo_root)."""
    repo_dir = tmp_path / name
    (repo_dir / "prompts").mkdir(parents=True)
    (repo_dir / "prompts" / "code01").write_text("code01 template: $goal\n")
    (repo_dir / "prompts" / "critic01").write_text("critic01 template\n")
    (repo_dir / "README.md").write_text("hello from the fixture repo\n")
    env_cfg = ["-c", "user.email=test@example.com", "-c", "user.name=Test User"]
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", *env_cfg, "-C", str(repo_dir), "add", "-A"], check=True)
    subprocess.run(
        ["git", *env_cfg, "-C", str(repo_dir), "commit", "-q", "-m", "initial commit"],
        check=True,
    )
    return repo_dir


def _write_scaffold(
    rig_root: Path,
    *,
    charter_content: str = BASE_CHARTER_TOML,
    models: bool = True,
    db: bool = True,
) -> None:
    """Create a rig directory on disk WITHOUT create_rig's git clone — just
    the artifacts resolve_rig gates on. ``models``/``db`` toggle whether the
    corresponding artifact is written (for the missing-artifact cases)."""
    rig_root.mkdir(parents=True, exist_ok=True)
    (rig_root / "charter.toml").write_text(charter_content)
    if models:
        shutil.copy(MODELS_REGISTRY_PATH, rig_root / "models.toml")
    if db:
        RigStore.create(rig_root / "tickets.db").close()


# --- case 1: happy path -----------------------------------------------------


def test_resolve_rig_happy_path(tmp_path: Path) -> None:
    repo = make_local_repo_with_prompts(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)

    resolved = resolve_rig("shipyard", rigs_root=rigs_root)
    try:
        assert isinstance(resolved, ResolvedRig)
        assert resolved.rig_root == rigs_root / "shipyard"
        assert isinstance(resolved.charter, Charter)
        assert isinstance(resolved.registry, Registry)
        assert isinstance(resolved.store, RigStore)

        # rig_paths carries EXACTLY the 5 daemon-required keys.
        assert set(resolved.rig_paths) == set(_REQUIRED_RIG_PATH_KEYS)
        rig_root = rigs_root / "shipyard"
        assert resolved.rig_paths["context_root"] == rig_root / "context"
        assert resolved.rig_paths["repo_root"] == rig_root / "repo"
        assert resolved.rig_paths["clones_root"] == rig_root / "clones"
        assert resolved.rig_paths["records_dir"] == rig_root / "records"
        # prompts_dir resolves relative to repo_root (§1.1), NOT rig_root.
        assert resolved.rig_paths["prompts_dir"] == rig_root / "repo" / "prompts"

        # Prove the artifacts are readable THROUGH the resolved path, not just
        # that the path string matches.
        prompts_dir = resolved.rig_paths["prompts_dir"]
        assert (prompts_dir / "code01").read_text().startswith("code01")
        assert (prompts_dir / "critic01").read_text().startswith("critic01")
    finally:
        resolved.store.close()


# --- case 2: ~/rigs default -------------------------------------------------


def test_resolve_rig_defaults_to_home_rigs(tmp_path: Path, monkeypatch) -> None:
    """rigs_root omitted -> resolves <~/rigs>/<name>, the SAME base create_rig
    itself defaults to. Both are Path('~/rigs').expanduser(), driven by $HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = make_local_repo_with_prompts(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    create_rig(charter_path)  # default base_dir -> ~/rigs == tmp_path/rigs

    resolved = resolve_rig("shipyard")  # default rigs_root -> ~/rigs
    try:
        assert resolved.rig_root == tmp_path / "rigs" / "shipyard"
    finally:
        resolved.store.close()


# --- case 3: rig_root missing ----------------------------------------------


def test_resolve_rig_missing_rig_root_raises(tmp_path: Path) -> None:
    with pytest.raises(RigError) as exc:
        resolve_rig("ghost", rigs_root=tmp_path)
    msg = str(exc.value)
    assert "ghost" in msg
    assert str(tmp_path / "ghost") in msg


# --- case 4: charter missing (+ multi-missing message) ----------------------


def test_resolve_rig_missing_charter_raises(tmp_path: Path) -> None:
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    _write_scaffold(rig_root)
    (rig_root / "charter.toml").unlink()

    with pytest.raises(RigError) as exc:
        resolve_rig("shipyard", rigs_root=rigs_root)
    assert "charter" in str(exc.value).lower()


def test_resolve_rig_lists_all_missing_artifacts(tmp_path: Path) -> None:
    """§4 case 4 (second half): BOTH charter.toml AND tickets.db missing — the
    single error names BOTH, not just the first checked."""
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    rig_root.mkdir(parents=True)
    shutil.copy(MODELS_REGISTRY_PATH, rig_root / "models.toml")  # only models present

    with pytest.raises(RigError) as exc:
        resolve_rig("shipyard", rigs_root=rigs_root)
    msg = str(exc.value).lower()
    assert "charter" in msg
    assert "ticket" in msg  # names the ticket store too, not just charter


# --- case 5: models.toml missing --------------------------------------------


def test_resolve_rig_missing_models_raises(tmp_path: Path) -> None:
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    _write_scaffold(rig_root, models=False)

    with pytest.raises(RigError) as exc:
        resolve_rig("shipyard", rigs_root=rigs_root)
    assert "model" in str(exc.value).lower()


# --- case 6: tickets.db missing ---------------------------------------------


def test_resolve_rig_missing_ticket_db_raises(tmp_path: Path) -> None:
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    _write_scaffold(rig_root, db=False)

    with pytest.raises(RigError) as exc:
        resolve_rig("shipyard", rigs_root=rigs_root)
    assert "ticket" in str(exc.value).lower()


# --- case 7: invalid charter -> CharterError propagates UNCAUGHT ------------


def test_resolve_rig_invalid_charter_propagates_charter_error(tmp_path: Path) -> None:
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    _write_scaffold(rig_root, charter_content=mutate("workers = 1", "workers = 2"))

    # NOT wrapped in RigError — a chartering bug surfaces as CharterError.
    with pytest.raises(CharterError):
        resolve_rig("shipyard", rigs_root=rigs_root)


# --- case 8: invalid registry -> fails closed -------------------------------
# DEVIATION from .27 build spec §1 step 2 (documented): the spec predicted
# UnbudgetableError here, but resolve_rig's step-1 load_charter ALREADY loads
# + validates the registry named by the charter's [models].registry and wraps
# any UnbudgetableError as CharterError. So when the charter-referenced
# registry is the broken one, CharterError escapes and resolve_rig's own
# step-2 load_registry is never reached. Fail-closed either way (both types
# are in _cmd_daemon_run's except tuple). Case 8b below pins the one scenario
# where the explicit load_registry's UnbudgetableError genuinely escapes.


def test_resolve_rig_invalid_charter_registry_propagates_charter_error(tmp_path: Path) -> None:
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    _write_scaffold(rig_root, models=False)
    # Valid TOML, but a model missing required fields. This IS the file the
    # charter's [models].registry names ("models.toml"), so load_charter fails.
    (rig_root / "models.toml").write_text('[brokenmodel]\nprovider = "anthropic"\n')

    with pytest.raises(CharterError):
        resolve_rig("shipyard", rigs_root=rigs_root)


def test_resolve_rig_broken_rig_local_registry_propagates_unbudgetable_error(
    tmp_path: Path,
) -> None:
    """Defense-in-depth edge (case 8b): resolve_rig's step-2 load_registry
    loads rig_root/models.toml DIRECTLY, whereas load_charter loads the
    charter's [models].registry path. If a (hand-edited) charter points its
    registry at a DIFFERENT, valid file but the rig-local models.toml is
    broken, the explicit load_registry's UnbudgetableError genuinely escapes
    uncaught — this is why UnbudgetableError stays in _cmd_daemon_run's catch."""
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    content = BASE_CHARTER_TOML.replace(
        'registry = "models.toml"', 'registry = "good_registry.toml"'
    )
    _write_scaffold(rig_root, charter_content=content, models=False)
    shutil.copy(MODELS_REGISTRY_PATH, rig_root / "good_registry.toml")  # load_charter OK
    # The rig-local models.toml (what resolve_rig loads directly) is broken.
    (rig_root / "models.toml").write_text('[brokenmodel]\nprovider = "anthropic"\n')

    with pytest.raises(UnbudgetableError):
        resolve_rig("shipyard", rigs_root=rigs_root)


# --- case 9: charter has no [prompts] table ---------------------------------


def test_resolve_rig_no_prompts_table_raises(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML.replace('[prompts]\ndir = "prompts/"\n', "")
    assert "[prompts]" not in content
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    _write_scaffold(rig_root, charter_content=content)

    with pytest.raises(RigError) as exc:
        resolve_rig("shipyard", rigs_root=rigs_root)
    assert "prompts" in str(exc.value).lower()


# --- case 10: [prompts] present but dir empty -------------------------------


def test_resolve_rig_empty_prompts_dir_raises(tmp_path: Path) -> None:
    content = BASE_CHARTER_TOML.replace('dir = "prompts/"', 'dir = ""')
    rigs_root = tmp_path / "rigs"
    rig_root = rigs_root / "shipyard"
    _write_scaffold(rig_root, charter_content=content)

    with pytest.raises(RigError) as exc:
        resolve_rig("shipyard", rigs_root=rigs_root)
    assert "prompts" in str(exc.value).lower()


# --- case 11: store returned OPEN + usable ----------------------------------


def test_resolve_rig_returns_open_usable_store(tmp_path: Path) -> None:
    repo = make_local_repo_with_prompts(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)

    resolved = resolve_rig("shipyard", rigs_root=rigs_root)
    try:
        # Not .create()'d and not closed — a live connection.
        resolved.store.add_ticket(id="t-1", title="live")
        got = resolved.store.get_ticket("t-1")
        assert got is not None
        assert got["id"] == "t-1"
    finally:
        resolved.store.close()


# --- case 12: side-effect free across repeated resolves ---------------------


def test_resolve_rig_is_side_effect_free_across_calls(tmp_path: Path) -> None:
    repo = make_local_repo_with_prompts(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    rigs_root = tmp_path / "rigs"
    create_rig(charter_path, base_dir=rigs_root)

    first = resolve_rig("shipyard", rigs_root=rigs_root)
    try:
        first.store.add_ticket(id="t-1", title="seed")
        before = sorted(t["id"] for t in first.store.list_tickets())
    finally:
        first.store.close()

    # A second resolve must not add/remove/mutate ticket rows — it only opens
    # the store (compare list_tickets across the second call).
    second = resolve_rig("shipyard", rigs_root=rigs_root)
    try:
        after = sorted(t["id"] for t in second.store.list_tickets())
    finally:
        second.store.close()

    assert before == after == ["t-1"]
