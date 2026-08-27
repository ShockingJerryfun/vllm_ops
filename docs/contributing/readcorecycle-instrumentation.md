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

PyTorch, vLLM, FlashAttention, generated-kernel launchers, and other consumers
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

`tools/run_qwen3_rcc_measurement.py` and `tools/run_qwen3_rcc_remote.sh` are
orchestration-only entrypoints. They may select a phase and exact stage, pin the
existing process, load/unload the PMU module, and call the collector lifecycle;
they must not read a timing source or synthesize a latency value.

If the canonical tool cannot be included or linked from a target, stop before
implementing a workaround. Report the exact compile or link failure, propose
the smallest reuse-based change, and obtain explicit user approval before
changing the tool contract. Integration convenience never authorizes a second
implementation.

## Freeze after validation

**Status: ABI v2 is under validation.** ABI v1 passed the server-84 gate on
2026-08-25 at repository commit `5c2a47143d2fb2310cf0055349acbb409bf9d514`.
It proved real nonzero PMCCNTR begin/end/delta consistency, complete sequence,
fixed owner CPU, migration zero, lost zero, successful cold CSV export, and
explicit cycles-to-microseconds conversion using a verified CPU frequency and
PMCR divider state.

ABI v2 adds only an exact-stage selector to the cold control path. It preserves
the serialized reader, the 32-byte event layout, the recorder, raw CSV columns,
and every hot interval boundary. Each measurement run selects exactly one site
and one stage so independent total and child-stage measurements cannot nest.

The ABI v2 stage layout for direct paths is fixed as follows:

- two-stage launchers use `base+0=prepare`, `base+1=submit`,
  `base+2=independent total`, `base+3=Dispatcher`, `base+4=return tail`,
  and `base+5=full dispatch`; bases are KV 30, RMSNorm 200, fused RMSNorm 210,
  SiLU 220, RoPE 230, embedding gather 240, and index-select 250;
- GEMM roles retain their four proven internal stages at bases 100, 110, 120,
  130, and 140, then use `base+4=independent total`, `base+5=Dispatcher`,
  `base+6=return tail`, and `base+7=full dispatch`;
- FlashAttention retains stages 40-44, then uses 45 total, 46 Dispatcher,
  47 return tail, 48 full dispatch, and 49 the full multi-launch submit group;
- generated Triton launchers use `base+0=prepare`, `base+1=submit`, and
  `base+2=independent total`; fixed bases 300 through 390 in steps of ten map
  the ten trace-confirmed Qwen3 compiled groups: first-layer embedding/norm,
  first-layer Q/K-Norm/RoPE reduction and pointwise launchers, repeated-layer
  Q/K-Norm/RoPE reduction and pointwise launchers, residual/RMSNorm, SiLU,
  and down-residual/next-norm. Dynamically classified Graph-external
  launchers use 500 through 580;
- auxiliary device copy uses base 400 with the same six-stage layout as the
  other two-stage launchers;
- FULL Graph Replay remains 50 total, 51 prologue, 52 stream lookup,
  53 `cudaGraphLaunch` Host submission, and 54 return tail.

Full-dispatch and return-tail spans cross function boundaries through fields in
the single thread-local selection object. Full dispatch begins at the public
operator handle and ends after the implementation returns to Dispatcher; it
does not synchronize the device. These spans never create another recorder or
event format. The collector requires a phase label for each derived artifact
and adds deterministic occurrence, semantic-role, layer, and stage-group
columns without altering raw CSV evidence.

After the ABI v2 gate passes, do not modify the three canonical files or the three
auxiliary files. Add future measurement sites only in target implementation
sources and external evidence mappings. A tool change after freeze requires
explicit user approval, an ABI version increment, a fresh observer-effect
check, and a fresh server-84 validation run.

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
rg -n 'PMCCNTR_EL0' csrc pytorch flash_attention triton \
  --glob '*.{h,hpp,c,cc,cpp,cu,cuh,py}'
rg -n 'ReadCoreCycleEvent|ReadCoreCycleRecorder' csrc pytorch flash_attention triton \
  --glob '*.{h,hpp,c,cc,cpp,cu,cuh,py}'
rg -n 'ReadCoreCycleProbe|DirectPmuProbe|OperatorCycleProbe' \
  csrc pytorch flash_attention triton --glob '*.{h,hpp,c,cc,cpp,cu,cuh,py}'
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

## Server-84 grouped measurement entrypoint

One command measures one exact phase/stage and writes raw, derived, summary,
controller, environment, and model-log artifacts into a new run directory:

```bash
EXPECTED_COMMIT=$(git -C /home/fj/vllm_ops_eager rev-parse HEAD) \
RCC_MODE=eager RCC_PHASE=eager-prefill RCC_STAGE_ID=202 \
/home/fj/vllm_ops_eager/tools/run_qwen3_rcc_remote.sh
```

Valid phase values are `eager-prefill`, `eager-decode`, `graph-capture-full`,
`graph-capture-piecewise`, `graph-prefill`, `graph-replay`, and
`graph-outside`. For `graph-outside`, step zero selects prefill and step one
selects the first decode execution. Use `RCC_TARGET_STEP=0` for the first
selected decode/replay step. The default
128-token input and four generated tokens are a data-quality pilot, not a
final workload; override `INPUT_TOKENS` and `MAX_TOKENS` only after the pilot
passes. Never run two stages in one process and never reuse an existing run
directory. Set `RCC_EXPECTED_COUNT` for roles with a fixed census so a missing
or duplicated occurrence fails the pilot immediately.
