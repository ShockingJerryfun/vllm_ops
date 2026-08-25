# ReadCoreCycle Instrumentation Contract

> Read this before modifying any ReadCoreCycle tool, timing site, recorder,
> PyTorch timing source, vLLM timing source, FlashAttention timing source, or
> generated launcher timing source. This contract is fail-closed.

## Single source of truth

The timing tool is already designed. These files have exclusive ownership:

- `csrc/read_core_cycle.h` exclusively defines `ReadCoreCycle()` and contains
  the `PMCCNTR_EL0` inline assembly.
- `csrc/read_core_cycle_profile.h` exclusively defines build profiles and
  exact-site selection rules.
- `csrc/read_core_cycle_recorder.h` exclusively defines the event layout,
  preallocated single-producer buffer, lost counter, and `RecordAfterEnd()`.

PyTorch, vLLM, FlashAttention, Triton generated launchers, and other consumers
must include and use these definitions. Do not copy, rename, wrap, or recreate
their primitive, selector, event, buffer, or recording logic elsewhere.

## Exact auxiliary allowlist

Only these auxiliary files are allowed:

- `csrc/read_core_cycle_runtime.h` declares the shared cold-path ABI.
- `csrc/read_core_cycle_runtime.cpp` owns the single recorder instance and
  implements start, stop, counts, and post-stop CSV export.
- `tools/collect_readcorecycle.py` performs same-thread cold lifecycle calls,
  validates raw evidence, and derives microseconds after collection.

The runtime files must include the canonical headers and must not redefine
assembly, event layout, buffer management, selection rules, or recording
behavior. The Python file must never read a clock or create timing values; it
may only validate native cycle records and perform explicit post-run unit
conversion. No fourth auxiliary timing or collection file is allowed.

If the canonical tool cannot be included or linked from a target, stop before
implementing a workaround. Report the exact compile or link failure, propose
the smallest reuse-based change, and obtain explicit user approval before
changing the tool contract. Integration convenience never authorizes a second
implementation.

## Frozen tool boundary

**Status: ABI v1 is user-locked.** Do not modify the three canonical files or
the three auxiliary files. The off-scope smoke run from 2026-08-25 was removed
and is not acceptance evidence for the current task. This lock comes from the
agreed tool contract, not from that run.

Add future measurement sites only in confirmed target implementation sources.
A tool change requires explicit user approval and an ABI version increment.
Builds, server runs, observer-effect checks, and measurements also require an
explicit request; they must not be inferred from a request to add timing sites.

## Immutable timing boundary

Raw latency must come only from the canonical serialized reader:

```text
isb sy
mrs PMCCNTR_EL0
isb sy
```

Every measured interval is:

```text
begin = ReadCoreCycle()
real target code
end = ReadCoreCycle()
record only after end
```

Do not substitute Python timers, CNTVCT, `clock_gettime`, perf/kperf, NVTX,
CUDA events, or GPU duration. The hot interval must not contain logging,
strings, locks, dynamic allocation, file I/O, or system calls.

## Prohibited duplicates

Do not add `ReadCoreCycleProbe`, `DirectPmuProbe`, `OperatorCycleProbe`, a
second recorder, local `PMCCNTR_EL0` assembly, an alternate site/profile
selector, or any control/adapter/collector outside the exact allowlist. Any
change to the responsibilities of the canonical or auxiliary files requires
explicit user approval.

## Mandatory pre-build gate

Before any local or server build, inspect the complete source tree:

```bash
rg -n 'PMCCNTR_EL0' csrc pytorch flash_attention \
  --glob '*.{h,hpp,c,cc,cpp,cu,cuh}'
rg -n 'ReadCoreCycleEvent|ReadCoreCycleRecorder' csrc pytorch flash_attention \
  --glob '*.{h,hpp,c,cc,cpp,cu,cuh}'
rg -n 'ReadCoreCycleProbe|DirectPmuProbe|OperatorCycleProbe' \
  csrc pytorch flash_attention --glob '*.{h,hpp,c,cc,cpp,cu,cuh}'
rg -n 'perf_counter|perf_counter_ns|time_ns|clock_gettime|CNTVCT|NVTX|cudaEvent' \
  tools/collect_readcorecycle.py
```

The gate passes only when:

- `PMCCNTR_EL0` and the `ReadCoreCycle()` definition occur only in
  `csrc/read_core_cycle.h`.
- Event and recorder definitions occur only in
  `csrc/read_core_cycle_recorder.h`.
- No second probe or recorder implementation exists.
- The Python collector contains no timing source.

If the gate fails, do not build, deploy, or collect measurements. Correct the
duplication first and report the final search results.
