# SPDX-License-Identifier: Apache-2.0
"""
Tests for audit.store.AuditStore — append-only immutability guarantees.

These are compliance-critical: the audit log is the evidentiary record for
every governance decision. If UPDATE or DELETE succeeded, a compromised
application process could forge, alter, or erase records after the fact.
The SQLite triggers enforce immutability at the DB level, independent of
the Python API surface.
"""

import sqlite3

import pytest

from kitelogik.audit.store import AuditStore
from kitelogik.tether.models import PolicyDecision, RiskTier, SessionContext


@pytest.fixture
def ctx() -> SessionContext:
    return SessionContext(
        session_id="audit_sess_1",
        user_role="support_agent",
        session_scopes=["read_customer"],
    )


@pytest.fixture
def decision() -> PolicyDecision:
    return PolicyDecision(
        allow=True,
        deny=False,
        risk_tier=RiskTier.INFORMATIONAL,
        requires_hitl=False,
        reason="Allowed — informational",
    )


@pytest.fixture
async def store(tmp_path) -> AuditStore:
    s = AuditStore(db_path=str(tmp_path / "audit.db"))
    await s.setup()
    return s


async def test_record_and_query_roundtrip(
    store: AuditStore, ctx: SessionContext, decision: PolicyDecision
):
    record_id = await store.record(
        session_id=ctx.session_id,
        tool_name="read_customer",
        args={"customer_id": "cust_001"},
        decision=decision,
        context=ctx,
        outcome="allowed",
    )
    assert record_id
    records = await store.query(session_id=ctx.session_id)
    assert len(records) == 1
    assert records[0].id == record_id
    assert records[0].tool_name == "read_customer"
    assert records[0].outcome == "allowed"
    # Policy version is captured per-record so evidence is tied to the
    # exact Rego policy state at decision time.
    assert records[0].policy_version is not None
    assert len(records[0].policy_version) == 16


async def test_update_raises_via_trigger(
    store: AuditStore, ctx: SessionContext, decision: PolicyDecision
):
    """Regression: audit_log must reject UPDATE at the SQL level, not only
    via application-layer discipline. A compromised process with direct DB
    access must not be able to rewrite the outcome column."""
    await store.record(
        session_id=ctx.session_id,
        tool_name="read_customer",
        args={"customer_id": "cust_001"},
        decision=decision,
        context=ctx,
        outcome="allowed",
    )

    with sqlite3.connect(store._db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE audit_log SET outcome = 'blocked'")


async def test_delete_raises_via_trigger(
    store: AuditStore, ctx: SessionContext, decision: PolicyDecision
):
    """Regression: DELETE against audit_log must abort. A record, once
    written, is evidence — it must never be removable through the normal
    SQL path."""
    await store.record(
        session_id=ctx.session_id,
        tool_name="read_customer",
        args={"customer_id": "cust_001"},
        decision=decision,
        context=ctx,
        outcome="allowed",
    )

    with sqlite3.connect(store._db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM audit_log")


async def test_targeted_update_on_specific_row_raises(
    store: AuditStore, ctx: SessionContext, decision: PolicyDecision
):
    """Even an UPDATE targeting a single row by primary key must abort."""
    record_id = await store.record(
        session_id=ctx.session_id,
        tool_name="read_customer",
        args={},
        decision=decision,
        context=ctx,
        outcome="allowed",
    )
    with sqlite3.connect(store._db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE audit_log SET hitl_decided_by = 'forged' WHERE id = ?",
                (record_id,),
            )


async def test_triggers_survive_reconnect(tmp_path, ctx, decision):
    """The triggers are created with IF NOT EXISTS on setup(). After the
    process restarts (i.e. a fresh AuditStore pointing at the same DB),
    the triggers must still be in force."""
    db_path = str(tmp_path / "restart.db")
    first = AuditStore(db_path=db_path)
    await first.setup()
    await first.record(
        session_id=ctx.session_id,
        tool_name="read_customer",
        args={},
        decision=decision,
        context=ctx,
        outcome="allowed",
    )

    # Second process — triggers should already be there from the first setup,
    # and running setup() again should be idempotent (IF NOT EXISTS).
    second = AuditStore(db_path=db_path)
    await second.setup()

    with sqlite3.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE audit_log SET outcome = 'tampered'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM audit_log")


async def test_export_session_integrity_hash_detects_tampering(
    store: AuditStore, ctx: SessionContext, decision: PolicyDecision
):
    """export_session embeds a SHA-256 of the records list. Editing any
    field in the exported dict must change the hash — this is the
    recipient-side integrity check."""
    import hashlib
    import json

    await store.record(
        session_id=ctx.session_id,
        tool_name="read_customer",
        args={"customer_id": "cust_001"},
        decision=decision,
        context=ctx,
        outcome="allowed",
    )

    report = await store.export_session(ctx.session_id)
    assert report["record_count"] == 1
    original_hash = report["integrity_hash"]

    # Recompute: hash should match.
    reproduced = hashlib.sha256(json.dumps(report["records"], sort_keys=True).encode()).hexdigest()
    assert reproduced == original_hash

    # Tamper: flip outcome field in the exported copy, rehash, compare.
    tampered_records = json.loads(json.dumps(report["records"]))
    tampered_records[0]["outcome"] = "blocked"
    tampered_hash = hashlib.sha256(
        json.dumps(tampered_records, sort_keys=True).encode()
    ).hexdigest()
    assert tampered_hash != original_hash
