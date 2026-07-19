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
import tempfile
from pathlib import Path

import pytest

from stigmergy import __version__
from stigmergy.charter import Charter, CharterError, load_charter
from stigmergy.checks import CheckOutcome, run_check
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
        assert store.get_meta("schema_version") == "4"
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
        assert rows == [("4",)]
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"tickets", "ticket_deps", "rig_meta", "worker_names", "filed_tickets"} <= tables
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


# ==========================================================================
# filed_tickets store + migration (D14; bead workspace-e2uh.38)
#
# The `filed_tickets` table is DELIBERATELY separate from `tickets`: a filed
# proposal is never a claimable row (structural un-claimability). Migration
# is a `CREATE TABLE IF NOT EXISTS` run on every RigStore open, so a rig
# scaffolded before D14 self-heals when reopened by D14 code.
# ==========================================================================


def test_add_and_read_filed_ticket_roundtrip(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_filed_ticket(
            id="filed-d1-1",
            title="Flaky test",
            description="fails 1/10",
            evidence="tests/x.py:1",
            origin_role="worker",
            origin_worker="worker-1",
            origin_dispatch_id="d1",
            origin_parent_ticket="workspace-e2uh.8",
            discovered_from="d1@workspace-e2uh.8",
            proposal_hash="abc123",
        )
        rows = store.list_filed_tickets()
        assert len(rows) == 1
        r = rows[0]
        assert r["id"] == "filed-d1-1"
        assert r["title"] == "Flaky test"
        assert r["description"] == "fails 1/10"
        assert r["evidence"] == "tests/x.py:1"
        assert r["origin_role"] == "worker"
        assert r["origin_worker"] == "worker-1"
        assert r["origin_dispatch_id"] == "d1"
        assert r["origin_parent_ticket"] == "workspace-e2uh.8"
        assert r["discovered_from"] == "d1@workspace-e2uh.8"
        assert r["proposal_hash"] == "abc123"
        assert r["triaged"] == 0
        assert r["triage_outcome"] is None
        assert r["resulting_ticket_id"] is None
        assert r["created_at"] > 0
    finally:
        store.close()


def test_add_filed_ticket_evidence_optional(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_filed_ticket(
            id="filed-d1-1",
            title="t",
            description="d",
            origin_role="critic",
            origin_worker="c-1",
            origin_dispatch_id="d1",
            origin_parent_ticket="p",
            discovered_from="d1@p",
            proposal_hash="h",
        )
        assert store.list_filed_tickets()[0]["evidence"] is None
    finally:
        store.close()


def test_count_untriaged_and_triaged_filter(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        for n in (1, 2, 3):
            store.add_filed_ticket(
                id=f"filed-d1-{n}",
                title=f"t{n}",
                description="d",
                origin_role="worker",
                origin_worker="w",
                origin_dispatch_id="d1",
                origin_parent_ticket="p",
                discovered_from="d1@p",
                proposal_hash=f"h{n}",
            )
        assert store.count_untriaged_filings() == 3
        assert len(store.list_filed_tickets(triaged=False)) == 3
        assert store.list_filed_tickets(triaged=True) == []
        assert len(store.list_filed_tickets()) == 3  # no filter -> all
    finally:
        store.close()


def test_filed_tickets_table_absent_from_pre_d14_rig_is_migrated_on_open(tmp_path: Path) -> None:
    """A rig store created, then its filed_tickets table dropped (simulating a
    pre-D14 scaffold), gains the table again on the NEXT plain open — the
    migration is idempotent and self-healing."""
    db_path = tmp_path / "tickets.db"
    store = RigStore.create(db_path)
    store._conn.execute("DROP TABLE filed_tickets")
    store._conn.commit()
    store.close()

    reopened = RigStore(db_path)  # plain open (mirrors resolve_rig)
    try:
        # table exists again and is usable.
        reopened.add_filed_ticket(
            id="filed-d1-1",
            title="t",
            description="d",
            origin_role="worker",
            origin_worker="w",
            origin_dispatch_id="d1",
            origin_parent_ticket="p",
            discovered_from="d1@p",
            proposal_hash="h",
        )
        assert reopened.count_untriaged_filings() == 1
    finally:
        reopened.close()


# ==========================================================================
# Bead .42 — schema v4 (functional_summary column) + triage-promotion store
# methods (SPEC §6 item 10 / D2 / D15).
# ==========================================================================


def _seed_filed(store: RigStore, filed_id: str = "filed-x-1") -> str:
    """Seed one untriaged filed proposal; return its id."""
    store.add_filed_ticket(
        id=filed_id,
        title="proposal title",
        description="proposal description",
        origin_role="worker",
        origin_worker="worker-haiku-code01-broom-casino-flock",
        origin_dispatch_id="dispatch-1",
        origin_parent_ticket="workspace-e2uh.8",
        discovered_from="dispatch-1@workspace-e2uh.8",
        proposal_hash="proposalhash",
    )
    return filed_id


def _full_ticket_fields() -> dict:
    """The §6 completion fields promote_filed_ticket accepts (minus id/title)."""
    return {
        "goal": "make it correct",
        "functional_summary": "Operator-facing: the thing now works.",
        "required_reading": ["repo:src/foo.py"],
        "target_scope": ["src/foo.py"],
        "acceptance_criteria": ["foo() returns 42"],
        "tier1_checks": {"pytest": "pytest -q"},
        "difficulty": "easy",
    }


def test_A5_fresh_create_has_functional_summary_column(tmp_path: Path) -> None:
    """A fresh RigStore.create carries functional_summary in the tickets DDL.
    (schema_version == "4" after scaffold is asserted by test_rig_meta_populated.)"""
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(tickets)")}
        assert "functional_summary" in cols
    finally:
        store.close()


def test_A6_functional_summary_column_migrated_on_open(tmp_path: Path) -> None:
    """A tickets table WITHOUT functional_summary (pre-.42 scaffold) gains the
    column on the next plain open; idempotent on a second open."""
    db_path = tmp_path / "tickets.db"
    store = RigStore.create(db_path)
    store._conn.execute("ALTER TABLE tickets DROP COLUMN functional_summary")
    store._conn.commit()
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(tickets)")}
    assert "functional_summary" not in cols  # precondition: really absent
    store.close()

    reopened = RigStore(db_path)  # plain open triggers the guarded-ALTER migration
    try:
        cols = {row["name"] for row in reopened._conn.execute("PRAGMA table_info(tickets)")}
        assert "functional_summary" in cols
    finally:
        reopened.close()

    # idempotent: a second open does not raise / double-add.
    reopened2 = RigStore(db_path)
    try:
        cols = {row["name"] for row in reopened2._conn.execute("PRAGMA table_info(tickets)")}
        assert "functional_summary" in cols
    finally:
        reopened2.close()


def test_A7_functional_summary_round_trips_as_plain_text(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_ticket(id="t-1", title="t", functional_summary="a plain string, not JSON")
        assert store.get_ticket("t-1")["functional_summary"] == "a plain string, not JSON"
        store.update_ticket("t-1", functional_summary="edited")
        assert store.get_ticket("t-1")["functional_summary"] == "edited"
    finally:
        store.close()


def test_B1_promote_filed_ticket_inserts_unapproved_and_tombstones(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        filed_id = _seed_filed(store)
        store.promote_filed_ticket(
            filed_id=filed_id, ticket_id="t-promoted", title="promoted ticket",
            **_full_ticket_fields(),
        )
        ticket = store.get_ticket("t-promoted")
        assert ticket is not None
        assert ticket["approved"] == 0  # UNAPPROVED — approval is a separate act
        assert ticket["functional_summary"] == "Operator-facing: the thing now works."
        assert ticket["target_scope"] == ["src/foo.py"]

        filed = store.list_filed_tickets(triaged=True)
        assert len(filed) == 1
        assert filed[0]["id"] == filed_id
        assert filed[0]["triaged"] == 1
        assert filed[0]["triage_outcome"] == "promoted"
        assert filed[0]["resulting_ticket_id"] == "t-promoted"
        assert store.count_untriaged_filings() == 0
    finally:
        store.close()


def test_B2_promote_filed_ticket_rejects_approved_fields(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        filed_id = _seed_filed(store)
        with pytest.raises(ValueError):
            store.promote_filed_ticket(
                filed_id=filed_id, ticket_id="t-x", title="x", approved=1,
            )
        with pytest.raises(ValueError):
            store.promote_filed_ticket(
                filed_id=filed_id, ticket_id="t-y", title="y", approval_hash="deadbeef",
            )
        # nothing landed / nothing tombstoned
        assert store.get_ticket("t-x") is None
        assert store.count_untriaged_filings() == 1
    finally:
        store.close()


def test_B3_promote_filed_ticket_duplicate_ticket_id_rolls_back(tmp_path: Path) -> None:
    """Atomicity: a failing INSERT (duplicate ticket id) leaves the filed row
    UNtriaged — the tombstone and the ticket insert are one transaction."""
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        store.add_ticket(id="t-dup", title="pre-existing")
        filed_id = _seed_filed(store)
        with pytest.raises(ValueError):
            store.promote_filed_ticket(
                filed_id=filed_id, ticket_id="t-dup", title="collides",
                **_full_ticket_fields(),
            )
        assert store.count_untriaged_filings() == 1  # filed row NOT tombstoned
        assert store.list_filed_tickets(triaged=True) == []
    finally:
        store.close()


def test_B4_promote_filed_ticket_bad_filed_id(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        with pytest.raises(ValueError):
            store.promote_filed_ticket(
                filed_id="nonexistent", ticket_id="t-z", title="z", **_full_ticket_fields()
            )
        # already-triaged filed row cannot be promoted twice
        filed_id = _seed_filed(store)
        store.mark_filed_ticket_triaged(filed_id, outcome="rejected")
        with pytest.raises(ValueError):
            store.promote_filed_ticket(
                filed_id=filed_id, ticket_id="t-z2", title="z2", **_full_ticket_fields()
            )
    finally:
        store.close()


def test_B5_mark_filed_ticket_triaged_tombstone(tmp_path: Path) -> None:
    store = RigStore.create(tmp_path / "tickets.db")
    try:
        filed_id = _seed_filed(store)
        store.mark_filed_ticket_triaged(filed_id, outcome="rejected")
        filed = store.list_filed_tickets(triaged=True)
        assert len(filed) == 1
        assert filed[0]["triage_outcome"] == "rejected"
        assert filed[0]["resulting_ticket_id"] is None
        assert store.count_untriaged_filings() == 0

        with pytest.raises(ValueError):
            store.mark_filed_ticket_triaged("nonexistent", outcome="rejected")
        with pytest.raises(ValueError):  # already triaged
            store.mark_filed_ticket_triaged(filed_id, outcome="rejected")
    finally:
        store.close()


# --- bead .90: scaffold creates the dispatch_base branch --------------------


def test_create_rig_creates_dispatch_base_branch(tmp_path: Path) -> None:
    # Workers dispatch FROM and the weaver lands ONTO refs/heads/<dispatch_base>
    # (fixture charter: "staging"). A fresh clone only has the source default
    # branch, so scaffold must create it or the daemon dispatches against
    # base_oid=None. Previously a manual ceremony step.
    repo = make_local_repo(tmp_path)
    charter_path = make_charter(tmp_path, repo)
    rig_root = create_rig(charter_path, base_dir=tmp_path / "rigs")
    r = subprocess.run(
        ["git", "-C", str(rig_root / "repo"), "rev-parse", "--verify", "refs/heads/staging"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, f"staging branch absent after scaffold: {r.stderr}"


def test_create_rig_dispatch_base_idempotent_when_it_is_default_branch(tmp_path: Path) -> None:
    # If dispatch_base already exists (it IS the repo's default branch),
    # scaffold must succeed (no "branch already exists" error) and the branch
    # must still be present.
    repo = make_local_repo(tmp_path)
    default_branch = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    charter_text = mutate('dispatch_base = "staging"', f'dispatch_base = "{default_branch}"')
    charter_path = make_charter(tmp_path, repo, content=charter_text)
    rig_root = create_rig(charter_path, base_dir=tmp_path / "rigs")  # must not raise
    r = subprocess.run(
        ["git", "-C", str(rig_root / "repo"), "rev-parse", "--verify",
         f"refs/heads/{default_branch}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0


# ==========================================================================
# bead .79 — per-rig worker image provisioning
# ==========================================================================

_PROVISION_CHARTER_TAIL = '\n[provision]\npip = ["ruff", "pytest"]\n'
_FAKE_DIGEST = "sha256:" + "a" * 64  # the (faked) BUILT per-rig image digest
_FAKE_BASE = "sha256:" + "b" * 64  # a pinned base [rig].image for unit tests
_BASE_WORKER_IMAGE = "localhost/stigmergy-worker:latest"
PODMAN = shutil.which("podman")


def _base_worker_image_id() -> str | None:
    if PODMAN is None:
        return None
    r = subprocess.run(
        [PODMAN, "inspect", "--format", "{{.Id}}", _BASE_WORKER_IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    val = r.stdout.strip()
    if r.returncode != 0 or not val:
        return None
    return val if val.startswith("sha256:") else f"sha256:{val}"


requires_base_worker_image = pytest.mark.skipif(
    _base_worker_image_id() is None,
    reason="requires podman + the built localhost/stigmergy-worker:latest base image",
)


def _charter_with_base(base_image: str, *, provision: bool) -> str:
    """BASE_CHARTER_TOML with [rig].image swapped to ``base_image`` and,
    optionally, a [provision] table appended."""
    out = []
    for ln in BASE_CHARTER_TOML.splitlines():
        out.append(f'image = "{base_image}"' if ln.strip().startswith("image =") else ln)
    text = "\n".join(out)
    return text + _PROVISION_CHARTER_TAIL if provision else text


def test_provision_wiring_faked_build(tmp_path, monkeypatch):
    """create_rig with [provision] generates the per-rig Containerfile, calls
    build_image, stores the digest in rig meta, and resolve_rig surfaces it as
    worker_image — with the real (slow) podman build faked out."""
    import stigmergy.rig as rig_mod

    captured: dict = {}

    def fake_build_image(containerfile_dir, tag, *, no_secrets=True):
        d = Path(containerfile_dir)
        captured["dir"] = d
        captured["tag"] = tag
        captured["containerfile"] = (d / "Containerfile").read_text()
        return _FAKE_DIGEST

    monkeypatch.setattr(rig_mod, "build_image", fake_build_image)

    repo = make_local_repo(tmp_path)
    base_dir = tmp_path / "rigs"
    charter_path = make_charter(
        tmp_path, repo, content=_charter_with_base(_FAKE_BASE, provision=True)
    )
    rig_root = create_rig(charter_path, base_dir=base_dir)

    assert captured["dir"] == rig_root / "images" / "worker"
    cf = captured["containerfile"]
    assert cf.startswith(f"FROM {_FAKE_BASE}")  # FROM the charter's pinned base digest
    assert "python3-pip" in cf  # pip bootstrapped onto the base
    assert "pip3 install --break-system-packages" in cf
    assert "ruff" in cf and "pytest" in cf  # the declared check tools
    # anti-trap: the project package itself is NEVER baked (would shadow /work)
    assert "install -e" not in cf

    resolved = resolve_rig("shipyard", rigs_root=base_dir)
    try:
        assert resolved.worker_image == _FAKE_DIGEST
    finally:
        resolved.store.close()


def test_no_provision_table_skips_build_and_falls_back_to_base(tmp_path, monkeypatch):
    """No [provision] table -> no per-rig build, and resolve_rig.worker_image
    falls back to the charter base [rig].image (the mechanical grep-smoke path
    stays unchanged)."""
    import stigmergy.rig as rig_mod

    called = {"n": 0}

    def fake_build_image(*a, **k):
        called["n"] += 1
        return _FAKE_DIGEST

    monkeypatch.setattr(rig_mod, "build_image", fake_build_image)

    repo = make_local_repo(tmp_path)
    base_dir = tmp_path / "rigs"
    charter_path = make_charter(tmp_path, repo)  # plain BASE, no [provision]
    create_rig(charter_path, base_dir=base_dir)

    assert called["n"] == 0
    resolved = resolve_rig("shipyard", rigs_root=base_dir)
    try:
        assert resolved.worker_image == resolved.charter.raw["rig"]["image"]
    finally:
        resolved.store.close()


def test_provision_containerfile_omits_pip_lines_when_no_specs(tmp_path, monkeypatch):
    """[provision] present but pip empty -> FROM only, no apt/pip layers."""
    import stigmergy.rig as rig_mod

    captured: dict = {}

    def fake_build_image(containerfile_dir, tag, *, no_secrets=True):
        captured["cf"] = (Path(containerfile_dir) / "Containerfile").read_text()
        return _FAKE_DIGEST

    monkeypatch.setattr(rig_mod, "build_image", fake_build_image)

    repo = make_local_repo(tmp_path)
    base_dir = tmp_path / "rigs"
    content = _charter_with_base(_FAKE_BASE, provision=False) + "\n[provision]\npip = []\n"
    charter_path = make_charter(tmp_path, repo, content=content)
    create_rig(charter_path, base_dir=base_dir)

    cf = captured["cf"]
    assert cf.strip().startswith(f"FROM {_FAKE_BASE}")
    assert "pip3 install" not in cf and "python3-pip" not in cf


@pytest.fixture(scope="module")
def provisioned_worker_image(tmp_path_factory):
    """Scaffold a rig whose base is the REAL built worker image + [provision]
    pip=[ruff,pytest], triggering a real per-rig image build. Module-scoped —
    the build is slow, so both live tests share the one image."""
    base_id = _base_worker_image_id()
    if base_id is None:
        pytest.skip("base worker image not available")
    tmp_path = tmp_path_factory.mktemp("prov79")
    repo = make_local_repo(tmp_path)
    base_dir = tmp_path / "rigs"
    charter_path = make_charter(
        tmp_path, repo, content=_charter_with_base(base_id, provision=True)
    )
    create_rig(charter_path, base_dir=base_dir)
    resolved = resolve_rig("shipyard", rigs_root=base_dir)
    try:
        worker_image = resolved.worker_image
    finally:
        resolved.store.close()
    assert worker_image.startswith("sha256:")
    assert worker_image != base_id  # a NEW image was built, not the base echoed back
    return worker_image


@requires_base_worker_image
def test_provision_image_has_check_tools_in_cage(provisioned_worker_image):
    """The per-rig image can actually run ruff + pytest inside a --network=none
    checker container (the .79 capability: real Tier-1 tools present in-cage)."""
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "f").write_text("x\n")
        res = run_check(
            "tools",
            "ruff --version && python3 -m pytest --version",
            work,
            image=provisioned_worker_image,
            flake_reruns=0,
        )
    assert res.outcome is CheckOutcome.PASS, res.output


@requires_base_worker_image
def test_anti_trap_checker_reads_work_candidate(provisioned_worker_image):
    """FIDELITY: the checker reads the /work CANDIDATE, not a stale baked copy.
    A passing synthetic test passes in-cage; mutating the /work file to break it
    makes the SAME check FAIL — proving the candidate is what's under test."""
    cmd = "cd /work && python3 -m pytest -q tests/test_synthetic.py"
    with tempfile.TemporaryDirectory() as work:
        tests_dir = Path(work) / "tests"
        tests_dir.mkdir()
        synthetic = tests_dir / "test_synthetic.py"

        synthetic.write_text("def test_ok():\n    assert True\n")
        passing = run_check(
            "anti-trap", cmd, work, image=provisioned_worker_image, flake_reruns=0
        )
        assert passing.outcome is CheckOutcome.PASS, passing.output

        synthetic.write_text("def test_ok():\n    assert False\n")
        failing = run_check(
            "anti-trap", cmd, work, image=provisioned_worker_image, flake_reruns=0
        )
        assert failing.outcome is CheckOutcome.FAIL, failing.output
