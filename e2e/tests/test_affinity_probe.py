# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-free negative controls for the affinity E2E observer."""

from __future__ import annotations

import pytest

from snapshot_e2e.affinity_probe import (
    PROBE_DONE,
    ThreadAffinity,
    compare_affinity,
    parse_cpu_list,
    parse_thread_scan,
)


def sample_scan(a="2", b="8", *, tgid=42, tids=(43, 44)) -> str:
    return (
        f"thread\tsnap-aff-a\t{tgid}\t{tids[0]}\t{a}\n"
        f"thread\tsnap-aff-b\t{tgid}\t{tids[1]}\t{b}\n"
        f"{PROBE_DONE}\n"
    )


@pytest.mark.parametrize(
    ("source", "want"),
    [("0", (0,)), ("0-2,8,11-12", (0, 1, 2, 8, 11, 12)), ("1000,1100-1101", (1000, 1100, 1101))],
)
def test_cpu_masks(source, want):
    assert parse_cpu_list(source) == want


@pytest.mark.parametrize("bad", ["", " 1", "1 ", "-1", "1--2", "2-1", "1,,2", "1-2,2", "1-3,2-4", "x", "1-1048576", "0-90000", "2 - 3"])
def test_invalid_cpu_masks_fail_closed(bad):
    with pytest.raises(ValueError):
        parse_cpu_list(bad)


def test_same_logical_threads_with_new_tids_are_valid():
    source = parse_thread_scan(sample_scan())
    restored = parse_thread_scan(sample_scan(tgid=91, tids=(92, 93)))
    assert compare_affinity(source, restored, (2, 8, 9))[0]


@pytest.mark.parametrize(
    "bad",
    [
        "", sample_scan().replace(PROBE_DONE, ""),
        sample_scan().replace("snap-aff-b", "snap-aff-a"),
        sample_scan().replace("snap-aff-b", "not-the-worker"),
        sample_scan().replace("42\t44", "99\t44"),
        sample_scan().replace("42\t44", "42\t43"),
        sample_scan().replace("\t8\n", "\t\n"),
        sample_scan() + "unexpected after marker\n",
    ],
)
def test_missing_or_inconsistent_threads_are_refused(bad):
    with pytest.raises(ValueError):
        parse_thread_scan(bad)


def test_successful_criu_restore_but_wrong_affinity_does_not_pass():
    source = parse_thread_scan(sample_scan())
    # Restore success is not passed as an argument; the kernel result rules.
    restored = parse_thread_scan(sample_scan(a="2,3,8", b="2,3,8", tgid=91, tids=(92, 93)))
    ok, reason = compare_affinity(source, restored, (2, 3, 8))
    assert not ok and "source mask" in reason


def test_topology_incompatible_mask_not_claimed_restored():
    source = parse_thread_scan(sample_scan())
    restored = parse_thread_scan(sample_scan(tgid=91, tids=(92, 93)))
    ok, reason = compare_affinity(source, restored, (8, 9))
    assert not ok and "destination does not permit" in reason


def test_source_mask_not_replaced_by_an_inferred_plan():
    source = parse_thread_scan(sample_scan())
    restored = parse_thread_scan(sample_scan(a="9", b="8", tgid=91, tids=(92, 93)))
    ok, reason = compare_affinity(source, restored, (2, 8, 9))
    assert not ok and "source mask" in reason


def test_missing_observation_not_a_success():
    source = parse_thread_scan(sample_scan())
    source.pop("snap-aff-a")
    restored = parse_thread_scan(sample_scan(tgid=91, tids=(92, 93)))
    assert not compare_affinity(source, restored, (2, 8))[0]
