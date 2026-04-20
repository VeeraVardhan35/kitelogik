#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Kite Logik — Memory + Credential Broker Benchmark

Measures the two supporting hot paths that sit next to the gate on every
agent session:

  memory_write_trusted     — MemoryStore.write, TRUSTED tier (no sanitizer)
  memory_write_untrusted   — MemoryStore.write, UNTRUSTED (runs sanitizer)
  memory_read              — MemoryStore.read by key
  cred_issue               — CredentialBroker.issue (in-memory broker)
  cred_validate            — CredentialBroker.validate on a live token
  cred_delegate            — CredentialBroker.delegate a child with subset scopes
  cred_revoke              — CredentialBroker.revoke_session

These are the paths users see in flamegraphs when they ask "what else does
Kite Logik do besides the gate call?". They should all sit well below the
gate latency — this benchmark confirms that.

Run:
  python benchmarks/bench_memory_session.py
  python benchmarks/bench_memory_session.py --runs 5000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import tempfile
import time

from kitelogik.anchor.credentials import CredentialBroker
from kitelogik.memory.models import TrustTier
from kitelogik.memory.store import MemoryStore

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


def _fmt_ms(ms: float) -> str:
    color = G if ms < 5 else Y if ms < 20 else R
    return f"{color}{ms:>7.3f}ms{RS}"


async def _bench_async(name: str, op, runs: int) -> list[float]:  # noqa: ANN001
    samples: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        await op(i)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def _bench_sync(name: str, op, runs: int) -> list[float]:  # noqa: ANN001
    samples: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        op(i)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


async def main(runs: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_path = os.path.join(tmpdir, "bench_memory.db")
        store = MemoryStore(db_path=mem_path)
        await store.setup()

        broker = CredentialBroker()
        session_id = "bench_sess"

        print(f"\n{B}{'═' * W}{RS}")
        print(f"{B}  Kite Logik — Memory + Credential Broker Benchmark{RS}")
        print(f"  {D}runs={runs}  ·  in-memory broker  ·  sqlite memory backend{RS}")
        print(f"{B}{'═' * W}{RS}\n")

        # ── memory ─────────────────────────────────────────────────────────
        async def write_trusted(i: int) -> None:
            await store.write(
                key=f"k_t_{i}",
                value=f'{{"customer_id":"cust_{i:04d}","name":"Alice"}}',
                trust_tier=TrustTier.TRUSTED,
                source="internal:crm",
                session_id=session_id,
            )

        async def write_untrusted(i: int) -> None:
            await store.write(
                key=f"k_u_{i}",
                value=f'{{"note":"customer report {i}"}} — arbitrary tool output.',
                trust_tier=TrustTier.UNTRUSTED,
                source="mcp:external",
                session_id=session_id,
            )

        async def read_one(i: int) -> None:
            await store.read(f"k_t_{i}")

        # Pre-seed so `read_one` always hits a row
        for i in range(runs):
            await store.write(
                key=f"k_t_{i}",
                value="seed",
                trust_tier=TrustTier.TRUSTED,
                source="seed",
                session_id=session_id,
            )

        mem_results: list[tuple[str, list[float]]] = []
        mem_results.append(("memory_write_trusted", await _bench_async("", write_trusted, runs)))
        mem_results.append(
            ("memory_write_untrusted", await _bench_async("", write_untrusted, runs))
        )
        mem_results.append(("memory_read", await _bench_async("", read_one, runs)))

        # ── credential broker ──────────────────────────────────────────────
        issued_tokens: list[str] = []

        def cred_issue(i: int) -> None:
            t = broker.issue(
                session_id=f"s_{i}",
                scopes=["read_customer", "approve_refund_under_100"],
            )
            issued_tokens.append(t.token_id)

        parent = broker.issue(session_id="parent_sess", scopes=["read_customer", "write_record"])

        def cred_validate(i: int) -> None:
            broker.validate(parent.token_id)

        def cred_delegate(i: int) -> None:
            broker.delegate(
                parent_token_id=parent.token_id,
                requested_scopes=["read_customer"],
                session_id=f"child_{i}",
            )

        def cred_revoke(i: int) -> None:
            broker.revoke_session(f"s_{i}")

        cred_results: list[tuple[str, list[float]]] = []
        cred_results.append(("cred_issue", _bench_sync("", cred_issue, runs)))
        cred_results.append(("cred_validate", _bench_sync("", cred_validate, runs)))
        cred_results.append(("cred_delegate", _bench_sync("", cred_delegate, runs)))
        cred_results.append(("cred_revoke", _bench_sync("", cred_revoke, runs)))

        # ── report ─────────────────────────────────────────────────────────
        col_name = 22
        col_num = 10
        print(
            f"{B}  {'Path':<{col_name}}  {'p50':>{col_num}}  {'p95':>{col_num}}"
            f"  {'p99':>{col_num}}  {'max':>{col_num}}{RS}"
        )
        print(f"  {D}{'─' * W}{RS}")

        for label, samples in mem_results + cred_results:
            p50 = statistics.median(samples)
            p95 = _pct(samples, 95)
            p99 = _pct(samples, 99)
            mx = max(samples)
            print(
                f"  {label:<{col_name}}  {_fmt_ms(p50)}  {_fmt_ms(p95)}  "
                f"{_fmt_ms(p99)}  {_fmt_ms(mx)}"
            )

        print(f"\n  {D}runs per path: {runs:,}{RS}")
        print(f"{B}{'═' * W}{RS}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kite Logik memory + credential benchmark")
    parser.add_argument("--runs", type=int, default=2000, help="Iterations per path")
    args = parser.parse_args()
    asyncio.run(main(args.runs))
