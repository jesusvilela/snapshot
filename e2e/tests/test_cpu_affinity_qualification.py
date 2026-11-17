# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual, real-CRIU qualification of two pinned Linux threads.

This test is intentionally opt-in: unmodified CRIU may restore the processes
without preserving per-thread affinity. When manually selected it MUST fail
on that mismatch, not interpret a successful checkpoint/restore as locality.
It exercises only same-host CPU identifiers, not GPU/NUMA proximity, memory
allocation policy, page residency, or the pre-resume release boundary.
"""

from __future__ import annotations

import os
import time

import pytest

from snapshot_e2e import k8s
from snapshot_e2e import lifecycle as snap
from snapshot_e2e import workloads
from snapshot_e2e.affinity_probe import (
    PROBE_DONE,
    THREAD_LABELS,
    ThreadAffinity,
    compare_affinity,
    parse_cpu_list,
    parse_thread_scan,
)


OPT_IN_ENV = "SNAPSHOT_E2E_QUALIFY_AFFINITY"
SHELL_OK = "__snapshot_affinity_shell_success__"

# The source CPU shell remains the usual Snapshot E2E workload. Add only an
# auxiliary two-thread Perl process. Both workers set distinct Linux comm names
# from their own thread, so restore PID/TID renumbering does not fake equality.
# Perl's 'threads' must exist in the actual agent-based workload image; if not,
# qualification fails as an environment capability failure, not a silent skip.
PERL_WORKERS = r"""
cat > /tmp/snapshot-affinity-e2e.pl <<'PERL_EOF'
use strict;
use warnings;
use threads;
sub worker {
    my ($label) = @_;
    open(my $comm, '>', '/proc/thread-self/comm') or die "comm open: $!";
    print {$comm} "$label\n" or die "comm write: $!";
    close($comm) or die "comm close: $!";
    while (1) { sleep 10; }
}
my @threads = (
    threads->create(\&worker, 'snap-aff-a'),
    threads->create(\&worker, 'snap-aff-b'),
);
while (1) { sleep 10; }
PERL_EOF
perl -Mthreads /tmp/snapshot-affinity-e2e.pl &
"""

# The shell runs in the source / restored container namespace and scans the
# actual /proc task directories. Labels are not sufficient identity in a
# general host: this is a disposable, isolated E2E container with one witness.
SCAN_COMMAND = rf"""
set -euo pipefail
for entry in /proc/[0-9]*/task/[0-9]*/comm; do
    [ -f "$entry" ] || continue
    label="$(cat "$entry")"
    case "$label" in
        snap-aff-a|snap-aff-b)
            task_dir="${{entry%/comm}}"
            tid="${{task_dir##*/}}"
            tgid="${{task_dir#/proc/}}"
            tgid="${{tgid%%/*}}"
            allowed="$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' "$task_dir/status")"
            test -n "$allowed"
            printf 'thread\t%s\t%s\t%s\t%s\n' "$label" "$tgid" "$tid" "$allowed"
            ;;
    esac
done
printf '{PROBE_DONE}\n'
"""


def inject_workers(command: str) -> str:
    """Keep the stock CPU workload and its readiness ordering intact."""
    anchor = f"mkdir -p {workloads.STATE_DIR}\n"
    if command.count(anchor) != 1:
        raise ValueError("CPU workload source command changed unexpectedly")
    return command.replace(anchor, anchor + PERL_WORKERS + "\n", 1)


def checked_shell(namespace: str, pod: str, commands: str) -> str:
    # k8s.exec_command uses a preloaded Kubernetes stream that can return
    # stdout even when the remote shell failed. The success marker prevents
    # an unexecuted readback from turning into test evidence.
    output = k8s.exec_payload(
        namespace,
        pod,
        f"set -euo pipefail\n{commands}\nprintf '{SHELL_OK}\\n'\n",
    )
    if SHELL_OK not in output.splitlines():
        raise AssertionError(f"remote affinity command did not complete: {output!r}")
    return output.rsplit(SHELL_OK, 1)[0]


def read_threads(namespace: str, pod: str) -> dict[str, ThreadAffinity]:
    return parse_thread_scan(k8s.exec_payload(namespace, pod, SCAN_COMMAND))


def wait_for_threads(namespace: str, pod: str) -> dict[str, ThreadAffinity]:
    last = "not observed"
    for _ in range(40):
        try:
            return read_threads(namespace, pod)
        except ValueError as exc:
            last = str(exc)
            time.sleep(1)
    raise AssertionError(f"two named OS threads were never observable: {last}")


def destination_allowed_cpus(namespace: str, pod: str) -> tuple[int, ...]:
    output = checked_shell(
        namespace,
        pod,
        "sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status",
    )
    # The remote shell's allowed mask is a destination compatibility control,
    # NOT a substitute for the restored workers' separate affinity readbacks.
    return parse_cpu_list(output.strip())


def pin_one_thread(namespace: str, pod: str, tid: int, cpu: int) -> None:
    output = checked_shell(
        namespace,
        pod,
        f"taskset -pc {cpu} {tid} >/dev/null\ntaskset -pc {tid}",
    )
    last = output.strip().splitlines()[-1] if output.strip() else ""
    if ": " not in last or parse_cpu_list(last.rsplit(": ", 1)[1]) != (cpu,):
        raise AssertionError(f"taskset syscall readback did not confirm CPU {cpu}: {output!r}")


@pytest.mark.cpu
@pytest.mark.snapshot_success
def test_cpu_affinity_roundtrip_requires_real_thread_readback(
    config: k8s.E2EConfig,
    run: snap.TestRun,
) -> None:
    if os.environ.get(OPT_IN_ENV) != "true":
        pytest.skip(f"manual-only CRIU qualification: set {OPT_IN_ENV}=true")
    if not config.cpu_only:
        pytest.fail("CPU affinity qualification must run in CPU-only E2E mode")

    try:
        source_pod_spec = snap.source_pod(config=config, run=run, gpu=False)
        cmd = source_pod_spec["spec"]["containers"][0]["command"]
        assert cmd[:2] == ["/bin/bash", "-lc"]
        cmd[2] = inject_workers(cmd[2])
        k8s.create_pod(source_pod_spec)
        source = snap.wait_for_pod_ready(config.namespace, run.source_pod)
        snap.wait_for_file(config.namespace, run.source_pod, snap.SOURCE_READY)

        initial = wait_for_threads(config.namespace, run.source_pod)
        allowed = set(initial[THREAD_LABELS[0]].cpus) & set(initial[THREAD_LABELS[1]].cpus)
        if len(allowed) < 2:
            pytest.fail("CPU-only E2E needs two CPUs allowed to both named worker threads")
        chosen = sorted(allowed)[:2]
        for label, cpu in zip(THREAD_LABELS, chosen):
            pin_one_thread(config.namespace, run.source_pod, initial[label].thread_id, cpu)
        source_observed = read_threads(config.namespace, run.source_pod)
        for label, cpu in zip(THREAD_LABELS, chosen):
            assert source_observed[label].cpus == (cpu,), (label, source_observed[label])
            # A second, syscall-backed taskset readback (not the status parser).
            output = checked_shell(
                config.namespace,
                run.source_pod,
                f"taskset -pc {source_observed[label].thread_id}",
            )
            assert parse_cpu_list(output.strip().splitlines()[-1].rsplit(": ", 1)[1]) == (cpu,)

        snap.create_podsnapshot(
            config.namespace, run.snapshot_name, run.source_pod, source.metadata.uid
        )
        snap.wait_for_snapshot_ready(config.namespace, run.snapshot_name)
        k8s.delete_pod(config.namespace, run.source_pod)
        snap.wait_for_pod_deleted(config.namespace, run.source_pod)

        k8s.create_pod(
            snap.restore_pod(config=config, run=run, gpu=False, source_node=source.spec.node_name)
        )
        snap.wait_for_restored_condition(
            config.namespace, run.restore_pod, "True", "RestoreSucceeded"
        )
        snap.wait_for_pod_ready(config.namespace, run.restore_pod, timeout=300)
        restored = wait_for_threads(config.namespace, run.restore_pod)
        target_allowed = destination_allowed_cpus(config.namespace, run.restore_pod)
        # Independent syscall readback of every destination thread, not just
        # /proc status: a second path must agree before interpreting equality.
        for label in THREAD_LABELS:
            output = checked_shell(
                config.namespace,
                run.restore_pod,
                f"taskset -pc {restored[label].thread_id}",
            )
            observed_syscall = parse_cpu_list(output.strip().splitlines()[-1].rsplit(": ", 1)[1])
            assert observed_syscall == restored[label].cpus, (
                f"destination {label} status/syscall disagreement: "
                f"{restored[label].cpus} vs {observed_syscall}"
            )
        ok, reason = compare_affinity(source_observed, restored, target_allowed)
        # This assertion MUST fail when CRIU reports success but drops saved
        # affinity, even if the workload's restoration and readiness passed.
        assert ok, f"CRIU returned success but per-thread CPU affinity is not preserved: {reason}"
    except Exception:
        snap.debug_dump(config, run)
        raise
