#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example 05 — HITL escalation.

When a policy sets ``requires_hitl=true``, the tool call is not allowed and not
hard-denied — it enters a review queue. The agent pauses; a human approver
decides; the agent resumes with the outcome recorded.

This example uses a $5,000 refund against a session scoped to $100. The
shipped ``main.rego`` classifies it as ``TRANSACTIONAL_HIGH`` and sets
``requires_hitl=true`` via the soft-deny fallback.

Prerequisite:
    docker compose up -d opa

Run:
    python examples/05_hitl_escalation.py
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from kitelogik import HITLQueue, OPAClient, PolicyGate, SessionContext, ToolCallInput
from kitelogik.anchor.models import ActionStatus, PendingAction


async def main() -> None:
    gate = PolicyGate(opa_client=OPAClient())
    # Short-lived file-backed queue — `:memory:` won't work here because
    # HITLQueue hops into a worker thread with its own SQLite connection.
    tmp = Path(tempfile.mkdtemp()) / "hitl.db"
    queue = HITLQueue(db_path=str(tmp))
    await queue.setup()  # create the pending_actions table

    context = SessionContext(
        session_id="example_05",
        user_role="support_agent",
        session_scopes=["read_customer", "approve_refund_under_100"],
    )

    # 1. Ask the gate — the model would do this before executing the tool.
    tc = ToolCallInput(
        action="approve_refund",
        tool_name="approve_refund",
        args={"customer_id": "cust_001", "amount": 5000.0},
    )
    decision = await gate.evaluate_tool_call(tc, context)
    print(
        f"Gate decision: allow={decision.allow}  "
        f"requires_hitl={decision.requires_hitl}  "
        f"risk_tier={decision.risk_tier}"
    )

    if not decision.requires_hitl:
        print("Unexpected — this scenario should require HITL. Check your policies.")
        return

    # 2. Enqueue the action for human review.
    pending = PendingAction(
        id="",
        session_id=context.session_id,
        tool_name=tc.tool_name,
        args=tc.args,
        risk_tier=decision.risk_tier,
        status=ActionStatus.PENDING,
        created_at=datetime.now(UTC),
    )
    action_id = await queue.enqueue(pending)
    print(f"Enqueued for review — action_id={action_id}")

    # 3. In production, an approver UI calls HITLQueue.approve() or .deny().
    #    We simulate that here with a tiny background task.
    async def approver() -> None:
        await asyncio.sleep(0.2)
        await queue.approve(action_id, decided_by="example_reviewer")

    approver_task = asyncio.create_task(approver())

    # 4. Poll for the decision — agents that want to block call get_status()
    #    in a short loop, or use a higher-level helper in the HITL queue.
    while True:
        row = await queue.get_status(action_id)
        if row and row.status != ActionStatus.PENDING:
            break
        await asyncio.sleep(0.05)

    await approver_task
    print(f"Decision: {row.status}  by={row.decided_by}  at={row.decided_at}")

    # 5. Agent would now execute (or skip) the tool based on row.status.
    if row.status == ActionStatus.APPROVED:
        print("Agent resumes — executing approve_refund($5,000)")
    else:
        print("Agent halts — refund denied by reviewer")


if __name__ == "__main__":
    asyncio.run(main())
