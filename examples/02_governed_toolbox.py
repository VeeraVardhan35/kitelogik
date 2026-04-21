#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example 02 — GovernedToolbox.

Register many tool functions under names, then call them by name through the
policy gate. Framework-agnostic — pairs with any agent loop that dispatches
``(tool_name, args_dict)`` pairs.

Prerequisite:
    docker compose up -d opa

Run:
    python examples/02_governed_toolbox.py
"""

from __future__ import annotations

import asyncio

from kitelogik import (
    GovernanceError,
    GovernedToolbox,
    OPAClient,
    PolicyGate,
    SessionContext,
)


def list_transactions(customer_id: str, limit: int = 10) -> list[dict]:
    return [{"id": f"tx_{i}", "customer_id": customer_id} for i in range(limit)]


async def approve_refund(customer_id: str, amount: float) -> str:
    return f"Refunded ${amount:.2f} to {customer_id}"


async def main() -> None:
    gate = PolicyGate(opa_client=OPAClient())
    context = SessionContext(
        session_id="example_02",
        user_role="support_agent",
        session_scopes=["read_customer", "approve_refund_under_100"],
    )

    toolbox = (
        GovernedToolbox(gate=gate, context=context)
        .register("list_transactions", list_transactions)
        .register("approve_refund", approve_refund)
    )

    print(f"Registered tools: {toolbox.tool_names()}")

    # Allowed — read scope covers list_transactions.
    rows = await toolbox.call("list_transactions", {"customer_id": "cust_001", "limit": 3})
    print(f"ALLOW  list_transactions -> {len(rows)} rows")

    # Allowed — within the $100 scope.
    result = await toolbox.call("approve_refund", {"customer_id": "cust_001", "amount": 42.0})
    print(f"ALLOW  approve_refund    -> {result}")

    # Denied — amount exceeds scope.
    try:
        await toolbox.call("approve_refund", {"customer_id": "cust_001", "amount": 5000.0})
    except GovernanceError as exc:
        print(f"BLOCK  approve_refund    -> {exc}")


if __name__ == "__main__":
    asyncio.run(main())
