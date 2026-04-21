#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example 03 — OpenAI adapter.

Routes OpenAI tool_calls through the policy gate before the underlying
function runs. Denials return a structured ``{"blocked": True, ...}`` message
so the agent loop can continue — the model sees the refusal and can respond.

This script fakes the OpenAI response shape with tiny stub objects so it runs
without an API key. In production, pass ``response.choices[0].message.tool_calls``
straight to ``adapter.execute_all()``.

Prerequisite:
    docker compose up -d opa

Run:
    python examples/03_openai_tools.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from kitelogik import OPAClient, PolicyGate, SessionContext
from kitelogik.adapters.openai import OpenAIAdapter


# ── Tool functions ────────────────────────────────────────────────────────────
async def approve_refund(customer_id: str, amount: float) -> str:
    return f"Refunded ${amount:.2f} to {customer_id}"


REFUND_SCHEMA = {
    "name": "approve_refund",
    "description": "Approve a refund for a customer.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "amount": {"type": "number"},
        },
        "required": ["customer_id", "amount"],
    },
}


# ── Minimal stubs that mimic OpenAI's tool_call shape ─────────────────────────
@dataclass
class _Function:
    name: str
    arguments: str  # JSON string, matching OpenAI's API


@dataclass
class _ToolCall:
    id: str
    function: _Function
    type: str = "function"


async def main() -> None:
    gate = PolicyGate(opa_client=OPAClient())
    context = SessionContext(
        session_id="example_03",
        user_role="support_agent",
        session_scopes=["read_customer", "approve_refund_under_100"],
    )

    adapter = OpenAIAdapter(gate=gate, context=context)
    adapter.register("approve_refund", approve_refund, schema=REFUND_SCHEMA)

    # These schemas feed your `client.chat.completions.create(tools=...)` call.
    tools_param = adapter.openai_tool_schemas()
    print(f"Schemas passed to the model: {[t['function']['name'] for t in tools_param]}")

    # Pretend the model returned two tool_calls — one allowed, one over-scope.
    model_tool_calls = [
        _ToolCall(
            id="call_1",
            function=_Function(
                name="approve_refund",
                arguments=json.dumps({"customer_id": "cust_001", "amount": 42.0}),
            ),
        ),
        _ToolCall(
            id="call_2",
            function=_Function(
                name="approve_refund",
                arguments=json.dumps({"customer_id": "cust_001", "amount": 5000.0}),
            ),
        ),
    ]

    # In production: `results = await adapter.execute_all(msg.tool_calls)`
    # then `messages.extend(results)` and loop again.
    results = await adapter.execute_all(model_tool_calls)
    for r in results:
        print(f"  tool_call_id={r['tool_call_id']}  content={r['content']}")


if __name__ == "__main__":
    asyncio.run(main())
