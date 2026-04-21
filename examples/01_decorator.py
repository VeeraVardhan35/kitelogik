#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example 01 — The @governed decorator.

Wrap a single async function with policy enforcement. The gate runs before the
function body; a policy denial raises ``GovernanceError`` with the full OPA
decision attached.

Prerequisite:
    docker compose up -d opa    # OPA at http://localhost:8181

Run:
    python examples/01_decorator.py
"""

from __future__ import annotations

import asyncio

from kitelogik import (
    GovernanceError,
    OPAClient,
    PolicyGate,
    SessionContext,
    governed,
)


async def main() -> None:
    gate = PolicyGate(opa_client=OPAClient())
    context = SessionContext(
        session_id="example_01",
        user_role="support_agent",
        session_scopes=["read_customer", "approve_refund_under_100"],
    )

    # Business function — no governance code inside. The decorator adds it.
    @governed(gate=gate, context=context)
    async def approve_refund(customer_id: str, amount: float) -> str:
        return f"Refunded ${amount:.2f} to {customer_id}"

    # Allowed — amount within the session scope.
    print(await approve_refund(customer_id="cust_001", amount=42.0))

    # Denied — amount exceeds the scope, policy returns allow=False.
    try:
        await approve_refund(customer_id="cust_001", amount=5000.0)
    except GovernanceError as exc:
        print(f"BLOCKED: {exc}")
        print(f"  risk_tier={exc.decision.risk_tier}  rule={exc.decision.rule_matched}")


if __name__ == "__main__":
    asyncio.run(main())
