# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict parsing/comparison of observed Linux thread affinity in CPU E2E.

The parser consumes only observations, never claims to freeze a task, verify
CRIU image provenance, or decide whether a NUMA placement is GPU-local.
"""

from __future__ import annotations

from dataclasses import dataclass


PROBE_DONE = "__snapshot_affinity_probe_complete__"
THREAD_LABELS = ("snap-aff-a", "snap-aff-b")
MAX_CPUS = 65536
MAX_CPU_ID = 1048575


@dataclass(frozen=True)
class ThreadAffinity:
    label: str
    process_id: int
    thread_id: int
    cpus: tuple[int, ...]


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse Linux Cpus_allowed_list strictly, including noncontiguous masks."""
    if not value or value.strip() != value:
        raise ValueError(f"missing or padded CPU list: {value!r}")
    selected: set[int] = set()
    for part in value.split(","):
        terms = part.split("-")
        if len(terms) not in (1, 2) or any(not t.isascii() or not t.isdecimal() for t in terms):
            raise ValueError(f"invalid CPU interval: {part!r}")
        start = int(terms[0])
        stop = int(terms[-1])
        if stop < start or stop > MAX_CPU_ID:
            raise ValueError(f"invalid CPU interval bounds: {part!r}")
        if len(selected) + (stop - start + 1) > MAX_CPUS:
            raise ValueError("CPU set too large for experimental observer")
        for cpu in range(start, stop + 1):
            if cpu in selected:
                raise ValueError(f"overlapping CPU intervals at {cpu}")
            selected.add(cpu)
    if not selected:
        raise ValueError("empty CPU set")
    return tuple(sorted(selected))


def parse_thread_scan(output: str) -> dict[str, ThreadAffinity]:
    """Reject missing, duplicate, cross-process or incomplete thread records."""
    lines = output.strip().splitlines()
    if not lines or lines[-1] != PROBE_DONE:
        raise ValueError("thread scan did not reach its success marker")
    parsed: dict[str, ThreadAffinity] = {}
    for line in lines[:-1]:
        parts = line.split("\t")
        if len(parts) != 5 or parts[0] != "thread":
            raise ValueError(f"unexpected thread scan line: {line!r}")
        _, label, process_text, thread_text, cpu_text = parts
        if label not in THREAD_LABELS or label in parsed:
            raise ValueError(f"unknown or duplicate thread label: {label!r}")
        if not process_text.isascii() or not process_text.isdecimal() or not thread_text.isascii() or not thread_text.isdecimal():
            raise ValueError("missing or malformed process/thread ID")
        process_id, thread_id = int(process_text), int(thread_text)
        if process_id < 1 or thread_id < 1:
            raise ValueError("process/thread IDs must be positive")
        parsed[label] = ThreadAffinity(label, process_id, thread_id, parse_cpu_list(cpu_text))
    if set(parsed) != set(THREAD_LABELS):
        raise ValueError(f"expected {THREAD_LABELS!r}, observed {tuple(sorted(parsed))!r}")
    if len({v.process_id for v in parsed.values()}) != 1:
        raise ValueError("observed threads do not share one thread group")
    if len({v.thread_id for v in parsed.values()}) != len(parsed):
        raise ValueError("duplicate Linux thread identifiers")
    return parsed


def compare_affinity(
    source: dict[str, ThreadAffinity],
    restored: dict[str, ThreadAffinity],
    dest_allowed: tuple[int, ...],
) -> tuple[bool, str]:
    """A same-node CPU-only check, not a cross-topology NUMA plan.

    A successful CRIU command is deliberately not an input. Affinity must
    match actual per-thread readbacks. Changes to TIDs across restore are OK.
    """
    if set(source) != set(THREAD_LABELS) or set(restored) != set(THREAD_LABELS):
        return False, "missing source or restored thread observation"
    if not dest_allowed:
        return False, "destination effective allowed CPU set missing"
    allowed = set(dest_allowed)
    for label in THREAD_LABELS:
        old, new = source[label], restored[label]
        if not old.cpus or not new.cpus:
            return False, f"{label}: missing CPU mask"
        if not set(old.cpus).issubset(allowed):
            return False, f"{label}: destination does not permit source CPU mask {old.cpus}"
        if old.cpus != new.cpus:
            return False, f"{label}: source mask {old.cpus}, restored mask {new.cpus}"
    return True, "same-node per-thread CPU masks matched"
