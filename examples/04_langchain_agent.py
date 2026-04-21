#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example 04 — LangChain adapter.

Two patterns:
  1. ``as_governed_tool`` wraps a plain callable as a governed ``BaseTool``.
  2. ``govern_toolkit`` wraps an entire list of existing ``BaseTool`` objects
     (e.g. from a community toolkit) without mutating them.

Denials return a ``[BLOCKED] ...`` string — the agent loop continues and the
model can decide how to recover.

Prerequisites:
    docker compose up -d opa
    pip install langchain-core     # only needed for this adapter

Run:
    python examples/04_langchain_agent.py
"""

from __future__ import annotations

import asyncio

from kitelogik import OPAClient, PolicyGate, SessionContext
from kitelogik.adapters.langchain import as_governed_tool, govern_toolkit


def approve_refund(customer_id: str, amount: float) -> str:
    return f"Refunded ${amount:.2f} to {customer_id}"


async def main() -> None:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        print("Install langchain-core to run this example:  pip install langchain-core")
        return

    gate = PolicyGate(opa_client=OPAClient())
    context = SessionContext(
        session_id="example_04",
        user_role="support_agent",
        session_scopes=["read_customer", "approve_refund_under_100"],
    )

    # ── Pattern 1: as_governed_tool ───────────────────────────────────────────
    refund_tool = as_governed_tool(
        name="approve_refund",
        fn=approve_refund,
        gate=gate,
        context=context,
        description="Approve a refund. Args: customer_id (str), amount (float).",
    )

    print("Pattern 1 — as_governed_tool")
    print(f"  ALLOW: {await refund_tool.ainvoke({'customer_id': 'cust_001', 'amount': 42.0})}")
    print(f"  BLOCK: {await refund_tool.ainvoke({'customer_id': 'cust_001', 'amount': 5000.0})}")

    # ── Pattern 2: govern_toolkit ─────────────────────────────────────────────
    # Pretend `existing_tools` came from a LangChain community toolkit.
    existing_tools = [
        StructuredTool.from_function(
            func=approve_refund,
            name="approve_refund",
            description="Approve a refund.",
        )
    ]
    governed_tools = govern_toolkit(existing_tools, gate=gate, context=context)

    print("\nPattern 2 — govern_toolkit")
    result = await governed_tools[0].ainvoke({"customer_id": "cust_001", "amount": 42.0})
    print(f"  ALLOW: {result}")
    blocked = await governed_tools[0].ainvoke({"customer_id": "cust_001", "amount": 5000.0})
    print(f"  BLOCK: {blocked}")

    # In a real app, hand `governed_tools` to `create_react_agent(llm, tools=...)`.


if __name__ == "__main__":
    asyncio.run(main())
