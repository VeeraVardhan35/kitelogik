#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Kite Logik — Policy Gate Latency Benchmark

Measures end-to-end latency of the policy gate across five scenarios that
exercise every path a production agent hits:

  simple_allow     — read_customer lookup, always allowed
  hard_deny        — read_file /app/.env, hard-blocked by security.rego
  hitl_trigger     — approve_refund $350, triggers HITL (allow=False, requires_hitl=True)
  agent_spawn      — agent.spawn event at depth 0
  agent_delegate   — agent.delegate event at depth 1

Reports p50, p95, p99, and max wall-clock latency for each scenario. This is
the number that sits on the agent's critical path — one call per tool use.

Prerequisites:
  docker compose up -d opa       # OPA policy engine on http://localhost:8181

Run:
  python benchmarks/bench_gate.py
  python benchmarks/bench_gate.py --runs 2000 --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

from kitelogik.anchor.credentials import CredentialBroker
from kitelogik.tether.gate import PolicyGate
from kitelogik.tether.models import GovernanceEvent, SessionContext, ToolCallInput
from kitelogik.tether.opa_client import OPAClient, OPAConnectionError

# ── ANSI ──────────────────────────────────────────────────────────────────────
B = "\033[1m"
D = "\033[2m"
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
RS = "\033[0m"
W = 74


def _pct(samples: list[float], p: int) -> float:
    if len(samples) < 100:
        return max(samples)
    return statistics.quantiles(samples, n=100)[p - 1]


def _bar() -> str:
    return "─" * W


def _fmt_ms(ms: float) -> str:
    color = G if ms < 10 else Y if ms < 30 else R
    return f"{color}{ms:>7.2f}ms{RS}"


async def _bench(
    name: str,
    op,  # noqa: ANN001 — callable with no args returning an awaitable
    runs: int,
    concurrency: int,
) -> list[float]:
    samples: list[float] = []
    sem = asyncio.Semaphore(concurrency)

    async def one() -> None:
        async with sem:
            t0 = time.perf_counter()
            await op()
            samples.append((time.perf_counter() - t0) * 1000)

    tasks = [asyncio.create_task(one()) for _ in range(runs)]
    total = len(tasks)
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % max(1, total // 10) == 0:
            pct = int(done / total * 100)
            print(f"  {D}{name:<22} {pct:>3}%{RS}", end="\r", flush=True)
    print(f"  {D}{name:<22} done {RS}" + " " * 20)
    return samples


async def main(runs: int, concurrency: int, opa_url: str, warmup: int) -> None:
    opa = OPAClient(base_url=opa_url)
    broker = CredentialBroker()
    gate = PolicyGate(opa_client=opa, credential_broker=broker)

    try:
        await opa.health()
    except OPAConnectionError:
        print(f"\n{R}OPA is not running.{RS}  Start it with: docker compose up -d opa\n")
        return

    context = SessionContext(
        session_id="bench_001",
        user_role="support_agent",
        session_scopes=["read_customer", "approve_refund_under_100"],
        sandbox_verified=False,
    )
    delegated_ctx = context.model_copy(
        update={"delegation_depth": 1, "parent_session_id": "bench_parent"}
    )

    scenarios: list[tuple[str, callable]] = [  # type: ignore[type-arg]
        (
            "simple_allow",
            lambda: gate.evaluate_tool_call(
                ToolCallInput(
                    action="list_transactions",
                    tool_name="list_transactions",
                    args={"customer_id": "cust_001"},
                ),
                context,
            ),
        ),
        (
            "hard_deny",
            lambda: gate.evaluate_tool_call(
                ToolCallInput(
                    action="read_file",
                    tool_name="read_file",
                    args={"path": "/app/.env"},
                    resource_path="/app/.env",
                ),
                context,
            ),
        ),
        (
            "hitl_trigger",
            lambda: gate.evaluate_tool_call(
                ToolCallInput(
                    action="approve_refund",
                    tool_name="approve_refund",
                    args={"customer_id": "cust_001", "amount": 350.0},
                ),
                context,
            ),
        ),
        (
            "agent_spawn",
            lambda: gate.evaluate(
                GovernanceEvent(
                    event_type="agent.spawn",
                    session_id="bench_spawn",
                    action="agent.spawn",
                    context=context,
                    requested_capabilities=list(context.session_scopes),
                )
            ),
        ),
        (
            "agent_delegate",
            lambda: gate.evaluate(
                GovernanceEvent(
                    event_type="agent.delegate",
                    session_id="bench_delegate",
                    action="agent.delegate",
                    context=delegated_ctx,
                    requested_capabilities=["read_customer"],
                )
            ),
        ),
    ]

    print(f"\n{B}{'═' * W}{RS}")
    print(f"{B}  Kite Logik — Policy Gate Benchmark{RS}")
    print(
        f"  {D}OPA: {opa_url}  ·  runs={runs}  ·  concurrency={concurrency}  "
        f"·  warmup={warmup}{RS}"
    )
    print(f"{B}{'═' * W}{RS}\n")

    # Warm-up — OPA compiles policies on first evaluation and httpx opens a
    # pooled connection; neither cost should pollute the sampled distribution.
    if warmup > 0:
        print(f"  {D}warmup {warmup} calls per scenario…{RS}")
        for _name, op in scenarios:
            for _ in range(warmup):
                await op()

    results: list[tuple[str, list[float]]] = []
    for name, op in scenarios:
        samples = await _bench(name, op, runs, concurrency)
        results.append((name, samples))

    col_name = 18
    col_num = 9

    print(
        f"\n{B}  {'Scenario':<{col_name}}  {'p50':>{col_num}}  {'p95':>{col_num}}"
        f"  {'p99':>{col_num}}  {'max':>{col_num}}{RS}"
    )
    print(f"  {D}{_bar()}{RS}")

    all_samples: list[float] = []
    for name, samples in results:
        p50 = statistics.median(samples)
        p95 = _pct(samples, 95)
        p99 = _pct(samples, 99)
        mx = max(samples)
        all_samples.extend(samples)
        print(
            f"  {name:<{col_name}}  {_fmt_ms(p50)}  {_fmt_ms(p95)}  "
            f"{_fmt_ms(p99)}  {_fmt_ms(mx)}"
        )

    p50 = statistics.median(all_samples)
    p95 = _pct(all_samples, 95)
    p99 = _pct(all_samples, 99)
    mx = max(all_samples)
    print(f"  {D}{_bar()}{RS}")
    print(
        f"  {'overall':<{col_name}}  {_fmt_ms(p50)}  {_fmt_ms(p95)}  "
        f"{_fmt_ms(p99)}  {_fmt_ms(mx)}"
    )
    print(f"\n  {D}total evaluations: {len(all_samples):,}  ·  OPA URL: {opa_url}{RS}")
    print(f"{B}{'═' * W}{RS}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kite Logik policy gate latency benchmark")
    parser.add_argument("--runs", type=int, default=1000, help="Evaluations per scenario")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent OPA requests")
    parser.add_argument("--warmup", type=int, default=50, help="Warm-up calls per scenario")
    parser.add_argument(
        "--opa",
        metavar="URL",
        default=os.getenv("OPA_BASE_URL", "http://localhost:8181"),
        help="OPA base URL",
    )
    args = parser.parse_args()
    asyncio.run(main(args.runs, args.concurrency, args.opa, args.warmup))
