"""Beads .166/.167 (Decision 18, Station Contract): the staging-gate and
range critics as ephemeral grounded agents.

The station exec is exercised through a SCRIPTED SEAM (`exec_fn`) — the
same discipline as `decompose._run_exec`'s monkeypatchable seam: no live
`openalph exec`, no provider calls, no `op` reads in the unit suite. The
seam scripts the exec envelope OA's terminal-tool mechanism returns: ONE
JSON line on stdout with `status` / `usage` / `tool_trace` / `result`
(the terminal tool's arguments ARE the return value) and — bead .162 — an
optional top-level `filed_proposals` list (the accepted file_ticket
filings, machine-assembled by the OA handler — the NEW filing channel; the
terminal payload carries NO filed_tickets field anymore).

Contract under test:
- judge/review return the EXACT upstream contract (Verdict + gate_fields +
  filings / RangeCriticResult) so the weaver and the range CLI consume
  them unchanged;
- ANY failure to obtain a valid, contract-conformant verdict is INFRA
  (CriticInfraError / RangeReportError), never a quality rejection;
- the retry budget is bounded (2 attempts, one retry, never a third), and
  an INFRA-CLASSIFIED episode (non-`done` status / deny_reason /
  ceiling_trip) does NOT retry — it surfaces after ONE attempt (bead .162);
- grammar-enforced FIELD defects (bad outcome, missing evidence_log) do
  NOT consume the retry — they cannot be fixed by re-running the same
  episode (strict-grammar violations are environment problems); only
  transport/exec-level failures retry;
- the task file carries the rubric (+ the machinery-appended standing
  item), the nonce-fenced artifact, the advisory evidence bundles, and
  the grounding repo path;
- the persisted tool_trace is bounded (.109/.112 discipline);
- construction is fail-closed (missing prompt artifact / agent TOML).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stigmergy.critic import STANDING_RUBRIC_ITEM, CriticInfraError
from stigmergy.oa_critic import CriticOAUnavailableError
from stigmergy.rangereport import RangeReportError
from stigmergy.station_critic import (
    _SUBMIT_RANGE_REVIEW_SCHEMA,
    _SUBMIT_VERDICT_SCHEMA,
    GATE_EXEC_TIMEOUT_SECONDS,
    StationGateCritic,
    StationRangeCritic,
    bound_tool_trace,
)
from stigmergy.verdicts import Outcome, Verdict

# --------------------------------------------------------------------------
# fixtures / seams
# --------------------------------------------------------------------------


class FakeRegistry:
    """Resolves any name to a Synthetic-routed entry (the qualified form the
    exec layer expects)."""

    def resolve(self, name: str) -> Any:
        return SimpleNamespace(
            oa_provider_key="synthetic", version="hf:moonshotai/Kimi-K3"
        )


class UnbudgetableRegistry:
    from stigmergy.registry import UnbudgetableError

    def resolve(self, name: str) -> Any:
        raise self.UnbudgetableError(name)


def write_prompts(tmp_path: Path) -> Path:
    prompts = tmp_path / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "critic04").write_text("critic04 template\n", encoding="utf-8")
    (prompts / "rangecrit03").write_text("rangecrit03 template\n", encoding="utf-8")
    return prompts


def make_gate_critic(tmp_path: Path, exec_fn, **overrides) -> StationGateCritic:
    toml = Path(__file__).parent.parent / "src/stigmergy/agents/stigmergy-decomposer.toml"
    defaults: dict[str, Any] = dict(
        registry=FakeRegistry(),
        model="synkimi3",
        prompts_dir=write_prompts(tmp_path),
        exec_fn=exec_fn,
        scratch_root=tmp_path / "scratch",
        agent_toml=toml,
    )
    defaults.update(overrides)
    return StationGateCritic(**defaults)


def make_range_critic(tmp_path: Path, exec_fn, **overrides) -> StationRangeCritic:
    toml = Path(__file__).parent.parent / "src/stigmergy/agents/stigmergy-decomposer.toml"
    defaults: dict[str, Any] = dict(
        registry=FakeRegistry(),
        model="synkimi3",
        prompts_dir=write_prompts(tmp_path),
        grounding_repo=tmp_path / "staging-repo",
        exec_fn=exec_fn,
        scratch_root=tmp_path / "scratch",
        agent_toml=toml,
    )
    defaults.update(overrides)
    return StationRangeCritic(**defaults)


def exec_envelope(
    *,
    status: str = "done",
    result: Any = None,
    tool_trace: Any = None,
    usage: Any = None,
    filed_proposals: Any = None,
    returncode: int = 0,
    stdout: str | None = None,
) -> subprocess.CompletedProcess:
    """One scripted OA exec envelope (ONE JSON line on stdout).

    `filed_proposals` (bead .162) is the envelope's TOP-LEVEL filing
    channel — added ONLY when not None (mirroring OA exec's additive
    field behavior; `None` = field absent).
    """
    if stdout is None:
        line = {
            "status": status,
            "content": "",
            "stop_reason": "end_turn",
            "ceiling_trip": None,
            "deny_reason": None,
            "detail": "",
            "usage": usage
            if usage is not None
            else {"in": 100, "cached": 10, "out": 40, "reasoning": 5},
            "tool_trace": tool_trace if tool_trace is not None else [],
        }
        if result is not None:
            line["result"] = result
        if filed_proposals is not None:
            line["filed_proposals"] = filed_proposals
        stdout = json.dumps(line) + "\n"
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


def verdict_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "outcome": "met",
        "tier": 2,
        "reason": "all items verified against the tree",
        "severity": "none",
        "evidence_log": [
            {
                "claim_checked": "diff adds handle_input branch",
                "method": "file_read src/stigmergy/weaver.py",
                "found": "branch present at the described site",
            }
        ],
        # bead .162: NO filed_tickets key — filings ride the exec
        # envelope's top-level `filed_proposals` (exec_envelope), not the
        # terminal payload.
    }
    base.update(overrides)
    return base


class ScriptedExec:
    """Exec seam scripting a SEQUENCE of outcomes (one per attempt)."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def __call__(self, argv, env, timeout):
        self.calls.append({"argv": argv, "env": env, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --------------------------------------------------------------------------
# schema shape
# --------------------------------------------------------------------------


def test_submit_verdict_schema_shape():
    assert _SUBMIT_VERDICT_SCHEMA["name"] == "submit_verdict"
    assert _SUBMIT_VERDICT_SCHEMA["strict"] is True
    schema = _SUBMIT_VERDICT_SCHEMA["input_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "outcome",
        "tier",
        "reason",
        "severity",
        "evidence_log",
    }
    assert schema["properties"]["outcome"]["enum"] == ["met", "unmet"]
    assert schema["properties"]["severity"]["enum"] == ["none", "low", "medium", "high"]
    assert schema["properties"]["tier"]["type"] == "integer"
    # bead .162: the filed_tickets field is GONE from the terminal payload
    # (filings moved to the file_ticket tool channel; the envelope's
    # top-level `filed_proposals` is the harvest surface).
    assert "filed_tickets" not in schema["properties"]


def test_submit_range_review_schema_shape():
    assert _SUBMIT_RANGE_REVIEW_SCHEMA["name"] == "submit_range_review"
    assert _SUBMIT_RANGE_REVIEW_SCHEMA["strict"] is True
    schema = _SUBMIT_RANGE_REVIEW_SCHEMA["input_schema"]
    assert set(schema["required"]) == {"text", "evidence_log"}
    assert "filed_tickets" not in schema["properties"]


# --------------------------------------------------------------------------
# gate station: happy path + provenance
# --------------------------------------------------------------------------


def test_gate_station_happy_path(tmp_path):
    trace = [
        {"name": "file_read", "path": "/work/src/app.py"},
        {"name": "grep", "pattern": "handle_input"},
    ]
    script = ScriptedExec(
        [exec_envelope(result=verdict_payload(), tool_trace=trace)]
    )
    critic = make_gate_critic(tmp_path, script)
    prompts_dir = tmp_path / "prompts"

    verdict, gate_fields, filed = critic.judge(
        "ticket: t1\nsome diff",
        ["item one", "item two"],
        check_evidence="CHECKS PASS",
        rename_evidence=None,
        grounding_repo="/tmp/candidate-clone",
    )

    # the verdict contract is the EXACT upstream shape
    assert isinstance(verdict, Verdict)
    assert verdict.outcome is Outcome.MET
    assert verdict.tier == 2
    assert filed == []

    # GATE-event provenance
    critic04 = prompts_dir / "critic04"
    assert gate_fields["prompt_artifact_hash"] == hashlib.sha256(
        critic04.read_bytes()
    ).hexdigest()
    assert gate_fields["model"] == "synthetic/hf:moonshotai/Kimi-K3"
    assert gate_fields["decoding_params"] == {}
    assert gate_fields["tokens"] == {"in": 100, "cached": 10, "out": 40, "reasoning": 5}
    assert gate_fields["wall_time_seconds"] >= 0.0
    assert gate_fields["ts"] > 0
    assert gate_fields["tool_trace"] == trace  # small trace passes through unbounded
    assert gate_fields["station"] == {
        "agent": "stigmergy-decomposer",
        "submit_tool": "submit_verdict",
        "prompt": "critic04",
    }
    assert gate_fields["station_attempts"] == 1

    # ONE exec, correct argv
    assert len(script.calls) == 1
    argv = script.calls[0]["argv"]
    assert argv[0].endswith("openalph") or "openalph" in argv[0]
    assert argv[argv.index("--agent") + 1] == "stigmergy-decomposer"
    assert argv[argv.index("--system-prompt-file") + 1] == str(critic04)
    assert argv[argv.index("--model") + 1] == "synthetic/hf:moonshotai/Kimi-K3"
    assert argv[argv.index("--effort") + 1] == "none"
    assert argv[argv.index("--tools") + 1] == "file_read,glob,grep,file_ticket"
    schema_path = Path(argv[argv.index("--submit-schema") + 1])
    assert schema_path.is_file()
    assert json.loads(schema_path.read_text()) == _SUBMIT_VERDICT_SCHEMA
    task_path = Path(argv[argv.index("--task-file") + 1])
    task = task_path.read_text()
    # rubric verbatim + the machinery-appended standing item
    assert "1. item one" in task and "2. item two" in task
    assert f"3. {STANDING_RUBRIC_ITEM}" in task
    # the artifact is embedded between nonce fence markers
    assert "===ARTIFACT-BEGIN " in task and "===ARTIFACT-END " in task
    assert "some diff" in task
    # the grounding repo is named
    assert "repo root: /tmp/candidate-clone" in task
    # the pre-seeded materials exist on disk where the task points
    assert "CHECKS PASS" in (task_path.parent / "check-evidence.md").read_text()
    assert (task_path.parent / "rubric.txt").is_file()


def test_gate_station_nonce_is_fresh_per_call(tmp_path):
    task_texts = []

    def record_and_succeed(argv, env, timeout):
        task_texts.append(Path(argv[argv.index("--task-file") + 1]).read_text())
        return exec_envelope(result=verdict_payload())

    critic = make_gate_critic(tmp_path, record_and_succeed)
    critic.judge("same artifact", ["item"], grounding_repo=None)
    critic.judge("same artifact", ["item"], grounding_repo=None)
    begins = [
        next(line for line in t.splitlines() if line.startswith("===ARTIFACT-BEGIN"))
        for t in task_texts
    ]
    assert begins[0] != begins[1]  # fresh nonce per call — a guessed fence forges nothing


def test_gate_station_rename_evidence_material_seeded(tmp_path):
    script = ScriptedExec([exec_envelope(result=verdict_payload())])
    critic = make_gate_critic(tmp_path, script)
    critic.judge("diff", ["item"], rename_evidence="R100 old -> new\nbody")
    script2 = ScriptedExec([exec_envelope(result=verdict_payload())])
    critic2 = make_gate_critic(tmp_path, script2)
    critic2.judge("diff", ["item"], rename_evidence=None)
    # the None case renders the explicit no-renames note in the task
    task2 = Path(
        script2.calls[0]["argv"][script2.calls[0]["argv"].index("--task-file") + 1]
    ).read_text()
    assert "no rename-touched paths" in task2


# --------------------------------------------------------------------------
# gate station: filings + bounded retries + infra semantics
# --------------------------------------------------------------------------


def test_gate_station_filed_proposals_flow_through(tmp_path):
    """bead .162 contract migration: filings arrive on the exec ENVELOPE's
    top-level `filed_proposals` (NOT the terminal payload) and flow through
    verbatim and unvalidated (filing.file_proposals remains the shape
    authority downstream)."""
    filing = {"title": "Add regression test", "description": "surfaced while judging"}
    script = ScriptedExec(
        [
            exec_envelope(
                result=verdict_payload(),
                tool_trace=[{"name": "file_ticket", "is_error": False}],
                filed_proposals=[filing],
            )
        ]
    )
    critic = make_gate_critic(tmp_path, script)
    _, _, filed = critic.judge("diff", ["item"])
    assert filed == [filing]


def test_gate_station_retries_once_then_succeeds(tmp_path):
    script = ScriptedExec(
        [
            subprocess.TimeoutExpired(cmd="openalph", timeout=600),
            exec_envelope(result=verdict_payload()),
        ]
    )
    critic = make_gate_critic(tmp_path, script)
    _, gate_fields, _ = critic.judge("diff", ["item"])
    assert len(script.calls) == 2  # exactly ONE retry, never a third
    assert gate_fields["station_attempts"] == 2


def test_gate_station_exhaustion_is_infra_never_rejection(tmp_path):
    # `done` exec that never submitted = station failure, retried once
    script = ScriptedExec(
        [exec_envelope(result=None), exec_envelope(result=None)]
    )
    critic = make_gate_critic(tmp_path, script)
    with pytest.raises(CriticInfraError) as excinfo:
        critic.judge("diff", ["item"])
    assert "never called" in str(excinfo.value)
    assert len(script.calls) == 2


def test_gate_station_exec_failure_class_is_infra(tmp_path):
    script = ScriptedExec(
        [exec_envelope(returncode=1), exec_envelope(returncode=1)]
    )
    critic = make_gate_critic(tmp_path, script)
    with pytest.raises(CriticInfraError) as excinfo:
        critic.judge("diff", ["item"])
    assert "exit 1" in str(excinfo.value)


def test_gate_station_deny_and_ceiling_are_infra(tmp_path):
    script = ScriptedExec(
        [
            exec_envelope(status="failed", result=None, stdout=json.dumps({
                "status": "failed", "ceiling_trip": None,
                "deny_reason": "revoked", "usage": {}, "tool_trace": [], "result": None,
            })),
        ]
    )
    critic = make_gate_critic(tmp_path, script)
    with pytest.raises(CriticInfraError) as excinfo:
        critic.judge("diff", ["item"])
    assert "deny_reason='revoked'" in str(excinfo.value)
    # bead .162: an infra-classified episode (bad status / deny / ceiling)
    # does NOT consume the retry budget — ONE attempt, surfaced bare
    # (v1: a re-run cannot fix a deny/ceiling; it would only double the cost).
    assert len(script.calls) == 1


def test_gate_station_missing_evidence_log_is_infra_no_retry(tmp_path):
    """A grammar-enforced FIELD defect cannot be fixed by re-running the
    same episode: no retry, straight to infra (the strict grammar makes it
    structurally unreachable on a conformant provider — reaching it means
    the grammar was NOT enforced)."""
    payload = verdict_payload()
    del payload["evidence_log"]
    script = ScriptedExec([exec_envelope(result=payload)])
    critic = make_gate_critic(tmp_path, script)
    with pytest.raises(CriticInfraError) as excinfo:
        critic.judge("diff", ["item"])
    assert "evidence_log" in str(excinfo.value)
    assert len(script.calls) == 1  # NO retry on field-level contract violations


def test_gate_station_bad_outcome_is_infra_no_retry(tmp_path):
    script = ScriptedExec(
        [exec_envelope(result=verdict_payload(outcome="maybe"))]
    )
    critic = make_gate_critic(tmp_path, script)
    with pytest.raises(CriticInfraError):
        critic.judge("diff", ["item"])
    assert len(script.calls) == 1


def test_gate_station_unparseable_stdout_is_infra(tmp_path):
    script = ScriptedExec(
        [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="not json", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="still not json", stderr=""),
        ]
    )
    critic = make_gate_critic(tmp_path, script)
    with pytest.raises(CriticInfraError):
        critic.judge("diff", ["item"])
    assert len(script.calls) == 2


# --------------------------------------------------------------------------
# tool_trace bounding (.109/.112 discipline)
# --------------------------------------------------------------------------


def test_tool_trace_bounded_entries_and_strings():
    # small fields: the ENTRY cap (64) binds, every string field intact
    big = [
        {"name": "file_read", "path": "x" * 10, "n": i} for i in range(100)
    ]
    bounded = bound_tool_trace(big)
    assert len(bounded) == 64  # _TRACE_MAX_ENTRIES
    assert all(len(e.get("path", "")) <= 200 for e in bounded)  # _TRACE_STR_CAP
    assert all(isinstance(e["n"], int) for e in bounded)  # non-strings preserved


def test_tool_trace_total_cap():
    # one huge string field: the aggregate cap stops the bleed
    bounded = bound_tool_trace([{"out": "y" * 20_000}])
    total = sum(len(str(v)) for e in bounded for v in e.values())
    assert total <= 8_192


def test_tool_trace_non_list_degrades_to_empty():
    assert bound_tool_trace(None) == []
    assert bound_tool_trace("junk") == []


def test_gate_station_gate_fields_carry_bounded_trace(tmp_path):
    trace = [{"name": "grep", "pattern": "z" * 900}]
    script = ScriptedExec([exec_envelope(result=verdict_payload(), tool_trace=trace)])
    critic = make_gate_critic(tmp_path, script)
    _, gate_fields, _ = critic.judge("diff", ["item"])
    assert len(gate_fields["tool_trace"][0]["pattern"]) == 200


# --------------------------------------------------------------------------
# fail-closed construction (rig launch, never the first gate)
# --------------------------------------------------------------------------


def test_gate_station_missing_prompt_artifact_fails_closed(tmp_path):
    prompts = write_prompts(tmp_path)
    (prompts / "critic04").unlink()
    with pytest.raises(CriticOAUnavailableError) as excinfo:
        StationGateCritic(
            registry=FakeRegistry(),
            model="synkimi3",
            prompts_dir=prompts,
            exec_fn=lambda *a: None,
            scratch_root=tmp_path / "scratch",
            agent_toml=tmp_path / "unused.toml",
        )
    assert "critic04" in str(excinfo.value)


def test_gate_station_missing_agent_toml_fails_closed(tmp_path):
    with pytest.raises(CriticOAUnavailableError) as excinfo:
        make_gate_critic(
            tmp_path,
            lambda *a: None,
            agent_toml=tmp_path / "not-installed.toml",
        )
    assert "not-installed.toml" in str(excinfo.value)


def test_gate_station_unbudgetable_registry_passthrough(tmp_path):
    """A registry miss resolves the CHARTER model through unchanged (the
    exec layer is the authority on model existence)."""

    class Miss:
        def resolve(self, name):
            from stigmergy.registry import UnbudgetableError

            raise UnbudgetableError(name)

    script = ScriptedExec([exec_envelope(result=verdict_payload())])
    critic = make_gate_critic(tmp_path, script, registry=Miss(), model="plain-name")
    critic.judge("diff", ["item"])
    assert script.calls[0]["argv"][script.calls[0]["argv"].index("--model") + 1] == "plain-name"


def test_gate_station_timeout_constant_is_gate_sized():
    """600s (not the decomposer's 1800s): the gate runs on the SERIALIZED
    weaver — 2 attempts x 30min could hold a weave for an hour."""
    assert GATE_EXEC_TIMEOUT_SECONDS == 600.0


# --------------------------------------------------------------------------
# range station
# --------------------------------------------------------------------------


def _range_report(tmp_path: Path):
    from stigmergy.rangereport import CommitInfo, RangeReport

    return RangeReport(
        base_oid="a" * 40,
        staging_oid="b" * 40,
        commits=(CommitInfo(oid="b" * 40, subject="feat: thing"),),
        diffstat=" app.py | 2 +-\n",
        diff="--- a/app.py\n+++ b/app.py\n+hello\n",
    )


def test_range_station_happy_path(tmp_path):
    script = ScriptedExec(
        [
            exec_envelope(
                result={
                    "text": "Overview: one commit.\n\nRisks: none observed.",
                    "evidence_log": [
                        {
                            "claim_checked": "app.py change is additive",
                            "method": "file_read app.py at staging tip",
                            "found": "one added line, no deletions",
                        }
                    ],
                },
                tool_trace=[
                    {"name": "file_read", "path": "app.py"},
                    {"name": "file_ticket", "is_error": False},
                ],
                filed_proposals=[{"title": "Follow up", "description": "because"}],
                usage={"in": 50, "cached": 0, "out": 200, "reasoning": 0},
            )
        ]
    )
    critic = make_range_critic(tmp_path, script)
    report = _range_report(tmp_path)

    result = critic.review(report)

    assert result.findings.startswith("Overview:")
    assert result.filed_tickets == [{"title": "Follow up", "description": "because"}]
    rangecrit03 = tmp_path / "prompts" / "rangecrit03"
    assert result.prompt_artifact_hash == hashlib.sha256(
        rangecrit03.read_bytes()
    ).hexdigest()
    assert result.model == "synthetic/hf:moonshotai/Kimi-K3"
    assert result.usage == {"in": 50, "cached": 0, "out": 200, "reasoning": 0}

    task = Path(
        script.calls[0]["argv"][script.calls[0]["argv"].index("--task-file") + 1]
    ).read_text()
    assert "===RANGE-BEGIN " in task
    assert "feat: thing" in task
    assert f"repo root: {tmp_path / 'staging-repo'}" in task
    argv = script.calls[0]["argv"]
    assert argv[argv.index("--system-prompt-file") + 1] == str(rangecrit03)
    schema_path = Path(argv[argv.index("--submit-schema") + 1])
    assert json.loads(schema_path.read_text()) == _SUBMIT_RANGE_REVIEW_SCHEMA


def test_range_station_missing_text_is_infra(tmp_path):
    script = ScriptedExec(
        [exec_envelope(result={"evidence_log": []})]
    )
    critic = make_range_critic(tmp_path, script)
    with pytest.raises(RangeReportError) as excinfo:
        critic.review(_range_report(tmp_path))
    assert "text" in str(excinfo.value)
    assert len(script.calls) == 1


def test_range_station_missing_evidence_log_is_infra(tmp_path):
    script = ScriptedExec([exec_envelope(result={"text": "prose"})])
    critic = make_range_critic(tmp_path, script)
    with pytest.raises(RangeReportError) as excinfo:
        critic.review(_range_report(tmp_path))
    assert "evidence_log" in str(excinfo.value)


def test_range_station_exhaustion_is_infra(tmp_path):
    script = ScriptedExec([exec_envelope(result=None), exec_envelope(result=None)])
    critic = make_range_critic(tmp_path, script)
    with pytest.raises(RangeReportError):
        critic.review(_range_report(tmp_path))
    assert len(script.calls) == 2


def test_range_station_malformed_filed_proposals_tolerated(tmp_path):
    """bead .162: the envelope's `filed_proposals` channel is tolerant — a
    non-list degrades to [] and never sinks the advisory prose (filing.py
    is the shape authority)."""
    script = ScriptedExec(
        [
            exec_envelope(
                result={"text": "prose", "evidence_log": []},
                filed_proposals="not-a-list",
            )
        ]
    )
    critic = make_range_critic(tmp_path, script)
    result = critic.review(_range_report(tmp_path))
    assert result.filed_tickets == []
