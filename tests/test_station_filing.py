"""Bead workspace-e2uh.162 (station half): filings leave the terminal payload.

The station contract moves filings OUT of the grammar-constrained terminal
call (submit_verdict / submit_range_review) into the file_ticket tool channel:

- the terminal schemas carry NO filed_tickets field (verdict shrinks to
  outcome/tier/reason/severity/evidence_log; range review to
  text/evidence_log) — the one-big-generation risk drops on the
  load-bearing call;
- critics file via file_ticket DURING the episode (call-time validation +
  in-session retry in OA); the accepted filings ride the exec envelope as a
  top-level `filed_proposals` list — machine-assembled by the OA handler
  from already-validated calls, NEVER model-generated payload;
- the station harvest reads envelope `filed_proposals` (defensive,
  never-raises) and returns it verbatim — the weaver/range-CLI seam
  (file_proposals, origin_role critic/range-critic) is UNCHANGED;
- per-item shape authority stays filing.file_proposals (bad-shape
  isolation, count-cap backstop, audit events);
- tool_trace lost-batch tripwire: a file_ticket call batched with the
  terminal call is NOT executed (OA terminal-turn semantics — the trace
  entry carries executed=False) — the station surfaces the count of ONLY
  those not-executed calls as gate_fields["filings_lost_batch"] so the
  GATE event records it (bounded observability, no behavior change);
  healthy rejections (is_error=True, executed=True) are the steering loop
  working and NEVER count as losses;
- infra-episode salvage (bead .162 audit fix, HIGH): CriticInfraError STILL
  raises (verdict semantics unchanged — no verdict), but the exception now
  carries the envelope's accepted filings (station_filed_proposals) + a
  bounded trace (station_tool_trace) so the weaver's gate-infra catch files
  them through the normal path instead of dropping them without trace;
- the worker driver passes FILE_TICKET_TRANSPORT/FILE_TICKET_MAX_FILINGS so
  in-cage workers file through the SAME tool (one channel fleet-wide).

Call shapes (judge(artifact, rubric_items), review(RangeReport), fixture
helpers) come from test_station_critic's shared scripted-seam surface; this
file adds only new-contract assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stigmergy.critic import CriticInfraError
from stigmergy import station_critic
from stigmergy.station_critic import (
    _SUBMIT_RANGE_REVIEW_SCHEMA,
    _SUBMIT_VERDICT_SCHEMA,
    STATION_TOOLS,
)

from test_station_critic import (  # scripted-seam helpers (stable surface)
    ScriptedExec,
    _range_report,
    exec_envelope,
    make_gate_critic,
    make_range_critic,
)


def new_contract_payload(**overrides: Any) -> dict[str, Any]:
    """A conformant submit_verdict payload WITHOUT filed_tickets."""
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
    }
    base.update(overrides)
    return base


def range_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "text": "## Overview\n\nOne ticket landed.\n\n## Risks & concerns\n\nNone observed.",
        "evidence_log": [
            {
                "claim_checked": "range-base resolution",
                "method": "grep weaver.py rangereport.py",
                "found": "single resolution path",
            }
        ],
    }
    base.update(overrides)
    return base


FILING = {
    "title": "Add a regression test for the empty-usage pricing branch",
    "description": "Judging surfaced an untested branch adjacent to the rubric.",
    "evidence": "src/stigmergy/pricing.py:88",
}

# bead .162: trace entries now carry "executed" (OA terminal-turn semantics:
# a call batched WITH the terminal call is not executed -> executed=False;
# every call that ran, whether accepted or rejected, is executed=True — a
# healthy rejection is is_error=True, executed=True). Absent key is treated
# as executed. The lost-batch tripwire counts ONLY the executed=False class.
TT_FILE_TICKET = {"name": "file_ticket", "is_error": False, "executed": True}
TT_FILE_TICKET_REJECTED = {"name": "file_ticket", "is_error": True, "executed": True}
TT_FILE_TICKET_NOT_EXECUTED = {"name": "file_ticket", "executed": False}


# --------------------------------------------------------------------------
# terminal schemas: filings are GONE from the payload
# --------------------------------------------------------------------------

def test_submit_verdict_schema_drops_filed_tickets():
    schema = _SUBMIT_VERDICT_SCHEMA["input_schema"]
    assert "filed_tickets" not in schema["properties"]
    assert set(schema["required"]) == {
        "outcome",
        "tier",
        "reason",
        "severity",
        "evidence_log",
    }


def test_submit_range_review_schema_drops_filed_tickets():
    schema = _SUBMIT_RANGE_REVIEW_SCHEMA["input_schema"]
    assert "filed_tickets" not in schema["properties"]
    assert set(schema["required"]) == {"text", "evidence_log"}


def test_filed_tickets_schema_helper_removed():
    # The tolerant array-of-objects channel is gone, not deprecated-unused.
    assert not hasattr(station_critic, "_filed_tickets_schema")


def test_station_tools_include_file_ticket():
    assert STATION_TOOLS == "file_read,glob,grep,file_ticket"


# --------------------------------------------------------------------------
# gate critic: filings ride the exec envelope
# --------------------------------------------------------------------------

def test_gate_filings_from_envelope(tmp_path: Path):
    exec_fn = ScriptedExec(
        [
            exec_envelope(
                result=new_contract_payload(),
                tool_trace=[{"name": "file_read", "is_error": False}, TT_FILE_TICKET],
                filed_proposals=[dict(FILING)],
            )
        ]
    )
    critic = make_gate_critic(tmp_path, exec_fn)
    verdict, gate_fields, filed_tickets = critic.judge("diff", ["item one"])
    assert verdict.outcome == "met"
    assert filed_tickets == [dict(FILING)]  # verbatim passthrough, unvalidated


def test_gate_filings_absent_returns_empty(tmp_path: Path):
    exec_fn = ScriptedExec([exec_envelope(result=new_contract_payload())])
    critic = make_gate_critic(tmp_path, exec_fn)
    _, _, filed_tickets = critic.judge("diff", ["item one"])
    assert filed_tickets == []


def test_gate_filings_non_list_returns_empty_never_raises(tmp_path: Path):
    exec_fn = ScriptedExec(
        [exec_envelope(result=new_contract_payload(), filed_proposals="garbage")]
    )
    critic = make_gate_critic(tmp_path, exec_fn)
    _, _, filed_tickets = critic.judge("diff", ["item one"])
    assert filed_tickets == []


def test_gate_payload_carrying_filed_tickets_is_ignored(tmp_path: Path):
    # A stale/lazy provider stuffing filed_tickets INTO the terminal payload
    # must not resurrect the old channel: payload field is not the source.
    payload = new_contract_payload(filed_tickets=[dict(FILING)])
    exec_fn = ScriptedExec([exec_envelope(result=payload)])
    critic = make_gate_critic(tmp_path, exec_fn)
    _, _, filed_tickets = critic.judge("diff", ["item one"])
    assert filed_tickets == []


def test_filings_lost_batch_counted(tmp_path: Path):
    # One healthy rejection (executed=True, is_error=True — the steering
    # loop working, NOT a loss) + one batched-with-terminal call the
    # terminal turn ended before executing (executed=False). Only the
    # not-executed call counts: lost_batch == 1.
    exec_fn = ScriptedExec(
        [
            exec_envelope(
                result=new_contract_payload(),
                tool_trace=[TT_FILE_TICKET_REJECTED, TT_FILE_TICKET_NOT_EXECUTED],
                filed_proposals=[dict(FILING)],
            )
        ]
    )
    critic = make_gate_critic(tmp_path, exec_fn)
    _, gate_fields, filed_tickets = critic.judge("diff", ["item one"])
    assert len(filed_tickets) == 1
    assert gate_fields["filings_lost_batch"] == 1


def test_no_lost_batch_when_consistent(tmp_path: Path):
    # One file_ticket call that executed and delivered -> no loss.
    exec_fn = ScriptedExec(
        [
            exec_envelope(
                result=new_contract_payload(),
                tool_trace=[TT_FILE_TICKET],
                filed_proposals=[dict(FILING)],
            )
        ]
    )
    critic = make_gate_critic(tmp_path, exec_fn)
    _, gate_fields, _ = critic.judge("diff", ["item one"])
    assert gate_fields["filings_lost_batch"] == 0


def test_rejected_call_does_not_inflate_lost_batch(tmp_path: Path):
    # A rejected-then-retried healthy loop: the rejected call
    # (is_error=True, executed=True) must NOT count as lost; the retry
    # delivered the filing. lost_batch == 0.
    exec_fn = ScriptedExec(
        [
            exec_envelope(
                result=new_contract_payload(),
                tool_trace=[TT_FILE_TICKET_REJECTED, TT_FILE_TICKET],
                filed_proposals=[dict(FILING)],
            )
        ]
    )
    critic = make_gate_critic(tmp_path, exec_fn)
    _, gate_fields, filed_tickets = critic.judge("diff", ["item one"])
    assert len(filed_tickets) == 1
    assert gate_fields["filings_lost_batch"] == 0


def test_infra_episode_salvages_filings_and_still_raises(tmp_path: Path):
    # Bead .162 audit fix (HIGH): an infra-classified episode STILL raises
    # CriticInfraError (verdict semantics unchanged — no verdict), but the
    # exception now carries the envelope's accepted filings + bounded trace
    # so the weaver's gate-infra catch can salvage them through the normal
    # filing path instead of dropping them with no record-plane trace.
    envelope = exec_envelope(
        status="infra",
        result=None,
        filed_proposals=[dict(FILING)],
        tool_trace=[TT_FILE_TICKET],
    )
    exec_fn = ScriptedExec([envelope])
    critic = make_gate_critic(tmp_path, exec_fn)
    with pytest.raises(CriticInfraError) as excinfo:
        critic.judge("diff", ["item one"])
    assert excinfo.value.station_filed_proposals == [dict(FILING)]
    assert isinstance(excinfo.value.station_tool_trace, list)
    assert excinfo.value.station_tool_trace == [TT_FILE_TICKET]


# --------------------------------------------------------------------------
# range critic: same envelope contract
# --------------------------------------------------------------------------

def test_range_filings_from_envelope(tmp_path: Path):
    exec_fn = ScriptedExec(
        [
            exec_envelope(
                result=range_payload(),
                tool_trace=[TT_FILE_TICKET],
                filed_proposals=[dict(FILING)],
            )
        ]
    )
    critic = make_range_critic(tmp_path, exec_fn)
    result = critic.review(_range_report(tmp_path))
    assert result.filed_tickets == [dict(FILING)]


def test_range_filings_absent_returns_empty(tmp_path: Path):
    exec_fn = ScriptedExec([exec_envelope(result=range_payload())])
    critic = make_range_critic(tmp_path, exec_fn)
    result = critic.review(_range_report(tmp_path))
    assert result.filed_tickets == []


# --------------------------------------------------------------------------
# child env: the filing cap rides to the tool
# --------------------------------------------------------------------------

def test_child_env_carries_max_filings_override(tmp_path: Path):
    exec_fn = ScriptedExec([exec_envelope(result=new_contract_payload())])
    critic = make_gate_critic(tmp_path, exec_fn, max_filings=5)
    critic.judge("diff", ["item one"])
    env = exec_fn.calls[0]["env"]
    assert env["FILE_TICKET_MAX_FILINGS"] == "5"


def test_child_env_default_matches_oa_default(tmp_path: Path):
    exec_fn = ScriptedExec([exec_envelope(result=new_contract_payload())])
    critic = make_gate_critic(tmp_path, exec_fn)
    critic.judge("diff", ["item one"])
    env = exec_fn.calls[0]["env"]
    assert env["FILE_TICKET_MAX_FILINGS"] == "8"


# --------------------------------------------------------------------------
# worker driver: same tool, file transport
# --------------------------------------------------------------------------

def test_worker_toolset_includes_file_ticket():
    from stigmergy.drivers.openalph_exec import _WORKER_TOOLS

    assert "file_ticket" in _WORKER_TOOLS.split(",")
