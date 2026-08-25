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

## Integration boundary

A cross-DSO adapter may expose control entry points such as start, stop, count,
and dump, and may own the single recorder instance. It must include the
canonical headers above and must not redefine assembly, event layout, buffer
management, selection rules, or recording behavior.

If the canonical tool cannot be included or linked from a target, stop before
implementing a workaround. Report the exact compile or link failure, propose
the smallest reuse-based change, and obtain explicit user approval before
changing the tool contract. Integration convenience never authorizes a second
implementation.

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
second recorder, local `PMCCNTR_EL0` assembly, or an alternate site/profile
selector. Any change to the responsibilities of the three canonical files
requires explicit user approval.

## Mandatory pre-build gate

Before any local or server build, inspect the complete source tree:

```bash
rg -n 'PMCCNTR_EL0' csrc pytorch flash_attention \
  --glob '*.{h,hpp,c,cc,cpp,cu,cuh}'
rg -n 'ReadCoreCycleEvent|ReadCoreCycleRecorder' csrc pytorch flash_attention \
  --glob '*.{h,hpp,c,cc,cpp,cu,cuh}'
rg -n 'ReadCoreCycleProbe|DirectPmuProbe|OperatorCycleProbe' \
  csrc pytorch flash_attention --glob '*.{h,hpp,c,cc,cpp,cu,cuh}'
```

The gate passes only when:

- `PMCCNTR_EL0` and the `ReadCoreCycle()` definition occur only in
  `csrc/read_core_cycle.h`.
- Event and recorder definitions occur only in
  `csrc/read_core_cycle_recorder.h`.
- No second probe or recorder implementation exists.

If the gate fails, do not build, deploy, or collect measurements. Correct the
duplication first and report the final search results.
