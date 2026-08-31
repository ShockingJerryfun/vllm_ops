#!/usr/bin/env python3
"""Audit that every planned Qwen3 stage has a concrete source wiring path."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceFile:
    label: str
    path: Path
    text: str
    lines: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--flash-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-tsv", type=Path, required=True)
    return parser.parse_args()


def load_source(label: str, path: Path) -> SourceFile:
    text = path.read_text(encoding="utf-8")
    return SourceFile(label, path, text, tuple(text.splitlines()))


def line_matches(source: SourceFile, pattern: re.Pattern[str]) -> list[str]:
    return [
        f"{source.label}:{line_number}"
        for line_number, line in enumerate(source.lines, start=1)
        if pattern.search(line)
    ]


def direct_matches(stage_id: int, sources: list[SourceFile]) -> list[str]:
    pattern = re.compile(rf"(?<!\d){stage_id}(?!\d)")
    return [match for source in sources for match in line_matches(source, pattern)]


def dynamic_matches(
    base: int,
    offset: int,
    base_sources: list[SourceFile],
    offset_sources: list[SourceFile],
    base_symbol: str,
) -> list[str]:
    base_pattern = re.compile(rf"(?<!\d){base}(?!\d)")
    offset_pattern = re.compile(rf"\b{re.escape(base_symbol)}\s*\+\s*{offset}\b")
    base_hits = [
        match for source in base_sources for match in line_matches(source, base_pattern)
    ]
    offset_hits = [
        match
        for source in offset_sources
        for match in line_matches(source, offset_pattern)
    ]
    if not base_hits or not offset_hits:
        return []
    return [base_hits[0], *offset_hits]


def source_matches(
    source: SourceFile,
    pattern: str,
) -> list[str]:
    return line_matches(source, re.compile(pattern))


def qwen3_increment_matches(
    stage_id: int,
    sources: dict[str, SourceFile],
) -> tuple[list[str], str]:
    qwen_header = sources["qwen_header"]
    if 11001 <= stage_id <= 11612:
        base = (stage_id // 100) * 100
        offset = stage_id - base
        source_key = {
            1: "empty_factory",
            2: "empty_cuda",
            3: "empty_generic",
            4: "empty_generic",
            5: "allocator",
            6: "allocator",
            7: "allocator",
            8: "allocator",
            9: "allocator",
            10: "allocator",
            11: "allocator",
            12: "empty_generic",
        }[offset]
        evidence = source_matches(qwen_header, rf"\b{base}\s*\+\s*offset\b")
        evidence.extend(
            source_matches(
                sources[source_key],
                rf"ReadCoreCycleQwen3LeafStageId\(\s*{offset}\s*\)",
            )
        )
        return evidence, "qwen3-allocator-root-plus-offset"

    if 11701 <= stage_id <= 11714:
        offset = stage_id - 11700
        source_keys = {
            1: ["copy"],
            2: ["copy"],
            3: ["copy"],
            4: ["copy"],
            5: ["copy_cuda"],
            6: ["copy_cuda"],
            7: ["copy_cuda"],
            8: ["copy_cuda"],
            9: ["loops"],
            10: ["copy_cuda"],
            11: ["cuda_loops"],
            12: ["copy_cuda"],
            13: ["copy_cuda"],
            14: ["copy"],
        }[offset]
        evidence = source_matches(qwen_header, r"\b11700\s*\+\s*offset\b")
        for source_key in source_keys:
            evidence.extend(
                source_matches(
                    sources[source_key],
                    rf"ReadCoreCycleQwen3LeafStageId\(\s*1057\s*,\s*{offset}\s*\)",
                )
            )
        return evidence, "qwen3-copy-root-plus-offset"

    if stage_id in {11801, 11814}:
        evidence = source_matches(
            sources["flash_api"],
            rf"ReadCoreCycleQwen3FlashStageId\(\s*{stage_id}\s*\)",
        )
        return evidence, "qwen3-flash-branch-condition"

    fill_group_by_stage = (
        (range(11802, 11806), 1, 11801),
        (range(11806, 11810), 2, 11805),
        (range(11810, 11814), 3, 11809),
        (range(11815, 11819), 4, 11814),
        (range(11819, 11823), 5, 11818),
    )
    for stage_range, marker, base in fill_group_by_stage:
        if stage_id not in stage_range:
            continue
        offset = stage_id - base
        source_key = {
            1: "flash_api",
            2: "fill",
            3: "cuda_loops",
            4: "cuda_loops",
        }[offset]
        evidence = source_matches(
            qwen_header,
            rf"case\s+{marker}:\s+return\s+static_cast<.*>\({base}\s*\+\s*offset\)",
        )
        evidence.extend(
            source_matches(
                sources["flash_api"],
                rf"ReadCoreCycleQwen3SubleafScope\s+\w+\({marker}\)",
            )
        )
        evidence.extend(
            source_matches(
                sources[source_key],
                rf"ReadCoreCycleQwen3FillStageId\(\s*{offset}\s*\)",
            )
        )
        return evidence, "qwen3-flash-fill-marker-plus-offset"

    if 11901 <= stage_id <= 11903:
        evidence = source_matches(
            qwen_header,
            r"stage_id\s*>=\s*11901\s*&&\s*stage_id\s*<=\s*11903",
        )
        evidence.extend(
            source_matches(
                sources["flash_interface"],
                r"maybe_contiguous\(x\).*for\s+x\s+in\s+\(q,\s*k,\s*v\)",
            )
        )
        return evidence, "source-conditional-runtime-evidence-required"

    return [], "direct"


def finalize_operator_total_metadata(stage: dict[str, Any]) -> None:
    """Replace workbook placeholders with the exact implemented total interval."""
    if stage["category"] != "operator_total":
        return
    if stage.get("operator_role") == "GEMM叶子":
        stage["stage_name"] = "aten::mm CUDA out实现总区间"
        stage["function_scope"] = (
            "at::native::mm_out_cuda() -> addmm_out_cuda_impl() -> "
            "cuBLAS/Lt Host提交 -> mm_out_cuda()返回"
        )
        stage["begin"] = (
            "TORCH_IMPL_FUNC(mm_out_cuda)入口完成shape/occurrence精确选择后"
        )
        stage["end"] = "addmm_out_cuda_impl()返回、mm_out_cuda()离开前"
        stage["include"] = (
            "CUDA out直接实现的shape/device检查、后端路径选择、"
            "参数构造、cuBLAS/Lt Host API提交与实现返回"
        )
        stage["exclude"] = (
            "Compiled Graph在mm_out_cuda之外的调用方工作、Dispatcher/Meta/"
            "输出分配；GPU Kernel设备执行"
        )
        stage["source_position"] = (
            "aten/src/ATen/native/cuda/Blas.cpp:88-157,423-698,833-838"
        )
    elif stage.get("operator_role") == "编译融合Launcher":
        stage["function_scope"] = (
            "_StaticCudaLauncher._launch_kernel() C入口 -> launch_kernel()参数解析/"
            "上下文与句柄准备 -> launch_kernel_inner()/launchKernel() -> "
            "cuLaunchKernel() -> Python None返回构造"
        )
        stage["begin"] = (
            "launch_kernel()识别rccOperatorStage后、PyArg_ParseTuple之前"
        )
        stage["end"] = "launch_kernel()各返回分支构造Py_None之后、返回调用方之前"
        stage["include"] = (
            "Static Triton C Launcher入口、参数解析、Context/句柄准备、"
            "参数打包、Launch配置、cuLaunchKernel Driver提交、状态检查与C返回尾部"
        )
        stage["source_position"] = (
            "torch/_inductor/runtime/static_triton_launcher.py:14-54,116-118,403-415; "
            "torch/csrc/inductor/static_launcher/cuda.cpp:442-505,588-790"
        )
    else:
        stage["function_scope"] = (
            "OperatorHandle::callBoxed()/TypedOperatorHandle::call()公开入口 -> "
            "Dispatcher::call()/callBoxed() -> KernelFunction调用 -> Dispatcher返回"
        )
        stage["begin"] = (
            "公开OperatorHandle入口完成精确算子/shape/父子occurrence选择后、"
            "进入Dispatcher之前"
        )
        stage["end"] = (
            "最外层选中Dispatcher调用的KernelFunction返回、FrontendChainGuard析构时"
        )
        stage["include"] = (
            "公开C++算子入口、Dispatcher key提取与kernel lookup、直接实现调用及"
            "该实现内部同步Host工作，直到最外层Dispatcher返回"
        )
        stage["source_position"] = (
            "aten/src/ATen/core/dispatch/Dispatcher.h:549-552,628-632,654-870,"
            "1011-1070,1092-1138"
        )
    if stage.get("operator_role") != "GEMM叶子":
        stage["exclude"] = (
            "GPU Kernel设备执行；调用方在公开算子入口之外的准备/尾部；"
            "嵌套子算子另行展示时不得与父总区间重复累计"
        )


def finalize_v2_lifecycle_metadata(stage: dict[str, Any]) -> None:
    """Bind lifecycle rows to the V2 Model Runner actually used at runtime."""
    stage_id = int(stage["stage_id"])
    updates: dict[int, dict[str, str]] = {
        5000: {
            "stage_name": "V2 FULL Graph Capture Host总区间",
            "function_scope": (
                "ModelCudaGraphManager.create_forward_fn(FULL,warmup=False) -> "
                "CudaGraphManager.capture() -> graph登记/计数"
            ),
            "begin": "FULL正式capture的fresh create_forward_fn入口（warmup之后）",
            "end": "graph写入self.graphs并更新num_cudagraph_captured之后",
            "include": (
                "FULL正式输入/元数据准备、Graph对象、offloader同步、"
                "capture_begin、模型体capture、capture_end/instantiate与graph登记"
            ),
            "exclude": "FULL前的PIECEWISE和warmup pass；GPU Kernel设备执行",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:333-389,504-615",
        },
        5001: {
            "stage_name": "FULL正式capture输入/前向闭包准备",
            "function_scope": "ModelCudaGraphManager.capture().create_forward_fn(desc,warmup=False)",
            "begin": "create_forward_fn入口",
            "end": "forward_fn闭包返回",
            "include": "LoRA状态、dummy model inputs、attention metadata、slot mapping与forward闭包",
            "exclude": "warmup准备、Graph对象和模型体执行",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:504-615",
        },
        5002: {
            "stage_name": "CUDAGraph对象创建",
            "function_scope": "CudaGraphManager.capture() -> torch.cuda.CUDAGraph()",
            "begin": "torch.cuda.CUDAGraph构造调用前",
            "end": "Graph对象返回",
            "include": "Python/PyBind CUDAGraph对象构造",
            "exclude": "Graph context、capture_begin和模型体",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:348-353",
        },
        5003: {
            "stage_name": "torch.cuda.graph上下文/Pool参数构造",
            "function_scope": "CudaGraphManager.capture() -> torch.cuda.graph(graph,self.pool)",
            "begin": "graph context manager构造前",
            "end": "context manager对象就绪（__enter__之前）",
            "include": "Graph和pool参数的Python context对象构造",
            "exclude": "__enter__内的capture_begin、模型体与capture_end",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:360-365",
        },
        5004: {
            "stage_name": "Capture前offloader依赖同步",
            "function_scope": "CudaGraphManager.capture() -> get_offloader().sync_prev_onload()",
            "begin": "sync_prev_onload入口",
            "end": "sync_prev_onload返回",
            "include": "copy stream依赖同步",
            "exclude": "Graph API和模型体",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:355-359",
        },
        5010: {
            "stage_name": "V2被捕获Qwen3模型体",
            "function_scope": "CudaGraphManager.capture() -> forward_fn(CUDAGraphMode.NONE)",
            "begin": "FULL capture context内forward_fn入口",
            "end": "forward_fn返回",
            "include": "Capture模型算子与细分Stage",
            "exclude": "capture_begin/end；内部算子不得再与本区间相加",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:365-371,550-610",
        },
        5011: {
            "stage_name": "Capture模型体后offloader join",
            "function_scope": "CudaGraphManager.capture() -> get_offloader().join_after_forward()",
            "begin": "forward_fn返回后",
            "end": "join_after_forward返回",
            "include": "offloader copy stream join",
            "exclude": "capture_end/instantiate",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:372-379",
        },
        5018: {
            "stage_name": "V2 Graph登记与capture计数",
            "function_scope": "self.graphs[desc]=graph; compilation_counter update",
            "begin": "torch.cuda.graph context退出后",
            "end": "graph字典登记和counter更新完成",
            "include": "Graph对象发布与capture计数",
            "exclude": "capture_end内部与函数返回后残差",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:382-389",
        },
        5100: {
            "stage_name": "V2 Graph Prefill模型下发总区间",
            "function_scope": "GPUModelRunner.execute_model() Run model branch",
            "begin": "Run model模式分支判断前",
            "end": "model_output就绪且forward context退出后",
            "include": "NONE/Prefill模式上下文、kv_connector pre_forward与模型主体",
            "exclude": "execute_model的输入整理/输出后处理；GPU Kernel设备执行",
            "source_position": "vllm/v1/worker/gpu/model_runner.py:1326-1380",
        },
        5101: {
            "stage_name": "Prefill模式分支与forward context进入",
            "function_scope": "BatchDescriptor() -> set_forward_context().__enter__()",
            "begin": "Run model模式分支判断前",
            "end": "set_forward_context进入完成",
            "include": "NONE模式选择、BatchDescriptor与forward context建立",
            "exclude": "kv_connector与模型体",
            "source_position": "vllm/v1/worker/gpu/model_runner.py:1329-1359",
        },
        5102: {
            "stage_name": "Prefill Qwen3模型体",
            "function_scope": "GPUModelRunner.execute_model() -> self.model(**model_inputs)",
            "begin": "self.model调用前",
            "end": "model_output返回",
            "include": "Prefill算子总区间与关键细分Stage",
            "exclude": "forward context进出与输出后处理",
            "source_position": "vllm/v1/worker/gpu/model_runner.py:1370-1374",
        },
        5103: {
            "stage_name": "Prefill forward context退出尾部",
            "function_scope": "set_forward_context().__exit__()",
            "begin": "model_output返回后",
            "end": "forward context退出完成",
            "include": "context manager退出与Python返回残差",
            "exclude": "模型体和execute_model输出处理",
            "source_position": "vllm/v1/worker/gpu/model_runner.py:1375-1378",
        },
        5200: {
            "stage_name": "V2 Steady Replay Host总区间",
            "function_scope": "ModelCudaGraphManager.run_fullgraph()",
            "begin": "ModelCudaGraphManager.run_fullgraph入口",
            "end": "静态hidden-state/output切片就绪、返回前",
            "include": "descriptor/graph查找、offloader同步、PyBind/C++ replay、cudaGraphLaunch与output取回",
            "exclude": "Graph内逐算子CPU时延与GPU执行",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:419-445,620-646",
        },
        5201: {
            "stage_name": "V2 Replay descriptor与Graph查找",
            "function_scope": "CudaGraphManager.run_fullgraph() assertions/dict lookup",
            "begin": "base run_fullgraph入口",
            "end": "FULL descriptor与self.graphs存在性确认后",
            "include": "mode、descriptor与Graph字典查找",
            "exclude": "offloader同步与replay",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:419-430",
        },
        5203: {
            "stage_name": "V2 Replay前offloader依赖同步",
            "function_scope": "CudaGraphManager.run_fullgraph() -> get_offloader().sync_prev_onload()",
            "begin": "sync_prev_onload入口",
            "end": "sync_prev_onload返回",
            "include": "prefill→replay的copy stream依赖同步",
            "exclude": "Graph replay",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:436-440",
        },
        5204: {
            "stage_name": "V2 Python→PyBind replay总调用",
            "function_scope": "self.graphs[desc].replay()",
            "begin": "Python replay方法调用前",
            "end": "PyBind/C++ replay返回Python后",
            "include": "Python属性/方法调用桥与C++ CUDAGraph::replay",
            "exclude": "offloader同步与output取回",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:441-445; torch/csrc/cuda/Graph.cpp",
        },
        5214: {
            "stage_name": "V2静态output切片与返回准备",
            "function_scope": "ModelCudaGraphManager.run_fullgraph() hidden/intermediate output selection",
            "begin": "base replay返回后",
            "end": "num_tokens切片/辅助hidden states组装完成",
            "include": "静态output引用切片与返回对象组装",
            "exclude": "Graph launch与GPU执行",
            "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:624-646",
        },
    }
    if stage_id in updates:
        stage.update(updates[stage_id])
        stage["source_summary"] = stage["source_position"]
    if stage_id == 5202:
        stage.update(
            {
                "stage_name": "Debug输入地址验证（V2不适用）",
                "function_scope": "V2 CudaGraphManager.run_fullgraph()无data_ptr地址校验分支",
                "begin": "N/A",
                "end": "N/A",
                "include": "N/A",
                "exclude": "本次V2 Model Runner的Steady Replay不存在该分支",
                "source_position": "vllm/v1/worker/gpu/cudagraph_utils.py:419-445",
                "source_summary": "V2 run_fullgraph源码无debug address-validation分支",
                "applicability": "V2 Model Runner源码证据N/A",
            }
        )


def main() -> int:
    args = parse_args()
    sources = {
        "cuda_graph_py": load_source(
            "vllm/compilation/cuda_graph.py",
            args.vllm_root / "vllm/compilation/cuda_graph.py",
        ),
        "v2_cudagraph": load_source(
            "vllm/v1/worker/gpu/cudagraph_utils.py",
            args.vllm_root / "vllm/v1/worker/gpu/cudagraph_utils.py",
        ),
        "v2_model_runner": load_source(
            "vllm/v1/worker/gpu/model_runner.py",
            args.vllm_root / "vllm/v1/worker/gpu/model_runner.py",
        ),
        "cache": load_source(
            "csrc/libtorch_stable/cache_kernels.cu",
            args.vllm_root / "csrc/libtorch_stable/cache_kernels.cu",
        ),
        "qwen_header": load_source(
            "csrc/read_core_cycle_qwen3.h",
            args.vllm_root / "csrc/read_core_cycle_qwen3.h",
        ),
        "flash_interface": load_source(
            "vllm/vllm_flash_attn/flash_attn_interface.py",
            args.vllm_root
            / "vllm/vllm_flash_attn/flash_attn_interface.py",
        ),
        "dispatcher": load_source(
            "aten/src/ATen/core/dispatch/Dispatcher.h",
            args.torch_root / "aten/src/ATen/core/dispatch/Dispatcher.h",
        ),
        "empty_factory": load_source(
            "aten/src/ATen/native/cuda/TensorFactories.cu",
            args.torch_root / "aten/src/ATen/native/cuda/TensorFactories.cu",
        ),
        "empty_cuda": load_source(
            "aten/src/ATen/cuda/EmptyTensor.cpp",
            args.torch_root / "aten/src/ATen/cuda/EmptyTensor.cpp",
        ),
        "empty_generic": load_source(
            "aten/src/ATen/EmptyTensor.cpp",
            args.torch_root / "aten/src/ATen/EmptyTensor.cpp",
        ),
        "copy": load_source(
            "aten/src/ATen/native/Copy.cpp",
            args.torch_root / "aten/src/ATen/native/Copy.cpp",
        ),
        "copy_cuda": load_source(
            "aten/src/ATen/native/cuda/Copy.cu",
            args.torch_root / "aten/src/ATen/native/cuda/Copy.cu",
        ),
        "loops": load_source(
            "aten/src/ATen/native/cuda/Loops.cuh",
            args.torch_root / "aten/src/ATen/native/cuda/Loops.cuh",
        ),
        "cuda_loops": load_source(
            "aten/src/ATen/native/cuda/CUDALoops.cuh",
            args.torch_root / "aten/src/ATen/native/cuda/CUDALoops.cuh",
        ),
        "fill": load_source(
            "aten/src/ATen/native/cuda/FillKernel.cu",
            args.torch_root / "aten/src/ATen/native/cuda/FillKernel.cu",
        ),
        "allocator": load_source(
            "c10/cuda/CUDACachingAllocator.cpp",
            args.torch_root / "c10/cuda/CUDACachingAllocator.cpp",
        ),
        "blas": load_source(
            "aten/src/ATen/native/cuda/Blas.cpp",
            args.torch_root / "aten/src/ATen/native/cuda/Blas.cpp",
        ),
        "cudablas": load_source(
            "aten/src/ATen/cuda/CUDABlas.cpp",
            args.torch_root / "aten/src/ATen/cuda/CUDABlas.cpp",
        ),
        "cudagraph": load_source(
            "aten/src/ATen/cuda/CUDAGraph.cpp",
            args.torch_root / "aten/src/ATen/cuda/CUDAGraph.cpp",
        ),
        "graph_bridge": load_source(
            "torch/csrc/cuda/Graph.cpp",
            args.torch_root / "torch/csrc/cuda/Graph.cpp",
        ),
        "triton_py": load_source(
            "torch/_inductor/runtime/static_triton_launcher.py",
            args.torch_root / "torch/_inductor/runtime/static_triton_launcher.py",
        ),
        "triton_cpp": load_source(
            "torch/csrc/inductor/static_launcher/cuda.cpp",
            args.torch_root / "torch/csrc/inductor/static_launcher/cuda.cpp",
        ),
        "flash_api": load_source(
            "csrc/flash_attn/flash_api.cpp",
            args.flash_root / "csrc/flash_attn/flash_api.cpp",
        ),
        "flash_launch": load_source(
            "csrc/flash_attn/src/flash_fwd_launch_template.h",
            args.flash_root / "csrc/flash_attn/src/flash_fwd_launch_template.h",
        ),
    }
    all_sources = list(sources.values())
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit_rows: list[dict[str, Any]] = []

    for stage in manifest["stages"]:
        finalize_operator_total_metadata(stage)
        finalize_v2_lifecycle_metadata(stage)
        stage_id = int(stage["stage_id"])
        category = stage["category"]
        evidence: list[str]
        wiring_kind = "direct"
        if stage_id >= 11001:
            evidence, wiring_kind = qwen3_increment_matches(stage_id, sources)
        elif category == "gemm":
            base = (stage_id // 100) * 100
            offset = stage_id - base
            evidence = dynamic_matches(
                base,
                offset,
                [sources["blas"]],
                [sources["blas"], sources["cudablas"]],
                "rcc_stage_base",
            )
            wiring_kind = "gemm-base-plus-offset"
        elif category == "triton":
            base = (stage_id // 100) * 100
            offset = stage_id - base
            evidence = dynamic_matches(
                base,
                offset,
                [sources["triton_py"]],
                [sources["triton_cpp"]],
                "rccStageBase",
            )
            wiring_kind = "triton-base-plus-offset"
        elif stage_id == 5202:
            evidence = [
                "vllm/v1/worker/gpu/cudagraph_utils.py:419-445 "
                "(V2 run_fullgraph has no debug address-validation branch)"
            ]
            wiring_kind = "source-backed-n/a-v2"
        else:
            evidence = direct_matches(stage_id, all_sources)

        status = (
            "source-backed-n/a"
            if stage_id == 5202
            else "source-conditional-runtime-evidence-required"
            if 11901 <= stage_id <= 11903 and evidence
            else "wired-source-audited"
            if evidence
            else "missing-source-wiring"
        )
        stage["implementation_status"] = status
        stage["wiring_kind"] = wiring_kind
        stage["wiring_evidence"] = evidence
        if 11901 <= stage_id <= 11903:
            stage["measurement_status"] = "runtime-evidence-pending"
        audit_rows.append(
            {
                "stage_id": stage_id,
                "stage_code": stage["stage_code"],
                "category": category,
                "status": status,
                "wiring_kind": wiring_kind,
                "evidence": evidence,
            }
        )

    manifest["source_audit"] = {
        "total": len(audit_rows),
        "wired": sum(row["status"] != "missing-source-wiring" for row in audit_rows),
        "missing": sum(row["status"] == "missing-source-wiring" for row in audit_rows),
        "runtime_evidence_pending": sum(
            row["status"] == "source-conditional-runtime-evidence-required"
            for row in audit_rows
        ),
        "method": (
            "Exact stage-ID wiring for direct stages; base selector plus measured "
            "offset wiring for GEMM, static Triton, allocator, copy and fill "
            "families; conditional contiguous leaves remain pending until "
            "runtime stride evidence is attached."
        ),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.audit_json.write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.audit_tsv.open("w", encoding="utf-8") as stream:
        stream.write("stage_id\tstage_code\tcategory\tstatus\twiring_kind\tevidence\n")
        for row in audit_rows:
            stream.write(
                "\t".join(
                    [
                        str(row["stage_id"]),
                        row["stage_code"],
                        row["category"],
                        row["status"],
                        row["wiring_kind"],
                        "; ".join(row["evidence"]),
                    ]
                )
                + "\n"
            )

    missing = [row for row in audit_rows if row["status"] == "missing-source-wiring"]
    print(
        json.dumps(
            {
                "total": len(audit_rows),
                "wired": len(audit_rows) - len(missing),
                "missing": [row["stage_id"] for row in missing],
            },
            ensure_ascii=False,
        )
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
