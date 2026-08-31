# Qwen3 Graph CPU Stage Instrumentation

This directory contains the reproducible workspace files for the formal
Qwen3 Graph-only CPU submission-stage measurement project.

## Scope

- Measures CPU Host submission paths for Graph Capture, Graph Prefill, and
  whole-Graph steady replay.
- Uses paired `PMCCNTR_EL0` cycles and `CNTVCT_EL0` time through
  ReadCoreCycle.
- Does not measure GPU kernel execution time and does not fabricate
  per-operator replay latency.

## Source baselines

- vLLM: `568afb3a13806beb53bb2e6bd518269357b237c0`
- PyTorch: `70d99e998b4955e0049d13a98d77ae1b14db1f45`
- vLLM FlashAttention: `caaa4eb59845388a20b1f435ecaafb4bd9517ad8`

The vLLM instrumentation is committed directly on this branch. Apply
`patches/pytorch.patch` and `patches/flash_attention.patch` to the matching
source baselines before rebuilding their binaries.

## Layout

- `manifests/`: Stage definitions, run matrices, conditional N/A evidence,
  and source identity.
- `patches/`: PyTorch and FlashAttention instrumentation patches.
- `scripts/`: build, runtime assembly, Stage execution, aggregation, and
  wiring-audit entry points used by the server project.

The canonical server workspace is `/home/fj/vllm_ops`. Historical run IDs
remain unchanged because they are part of the measurement provenance.
