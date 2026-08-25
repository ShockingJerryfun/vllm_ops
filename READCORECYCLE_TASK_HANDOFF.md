# vllm_ops ReadCoreCycle 打点任务交接

更新日期：2026-08-25

## 1. 当前授权边界

当前任务只维护源码和打点位置，不包含编译、部署、PMU 配置、模型推理、
CUDA Graph 实测、性能对照、结果表格或图形。旧的“Qwen3-8B 全算子
ReadCoreCycle 下发时延实测”目标已经失效，不得创建、恢复或继续执行。

仓库是官方 vLLM 0.26 源码基础上的独立 ReadCoreCycle 打点仓库：

- 本地唯一编辑目录：`/Users/fangjin/Documents/HW/AI机头/vllm_ops`
- GitHub：`ShockingJerryfun/vllm_ops`，维护分支 `main`
- server-84 唯一代码目录：`/home/fj/vllm_ops`
- `/home/fj/qwen3_operator_dispatch_readcorecycle_v1` 已清理，禁止重建
- `/home/fj/vllm_fj` 是旧项目，只保留，不在本任务修改

任何新会话都必须先确认用户当前授权；“完成打点”不等于“授权实测”。

## 2. 最终对齐的两层口径

### L1：FULL CUDA Graph steady replay 下发阶段

L1 只描述一次 FULL graph replay 的全局主机下发边界，不能拆成 434 个
语义算子的逐算子时延。固定为 `CUDAGraph::replay()` 内三个真实阶段：

| site | 阶段 | 被计时代码 |
| --- | --- | --- |
| 340 | R1 | `DeviceGuard` 与 generator `replay_prologue` |
| 341 | R2 | `getCurrentCUDAStream()` |
| 342 | R3 | `cudaGraphLaunch()` 主机 API 返回 |

R3 只能称为 `cudaGraphLaunch` 主机返回时延，不是 GPU 完成时间。

### L2：真实实现/融合组的整体主机下发时延

L2 是不同算子实现之间的主比较口径。一个 L2 区间应从已经确认的真实
C++ 或 generated implementation 入口开始，覆盖参数准备、dispatch 和该
实现相关的全部 CUDA/library host submission 返回。融合算子只记录一次
实现组整体时延；语义算子引用该实测组，不能复制后相加。

需覆盖的实现组至少包括：Linear/GEMM、Attention、KV Cache、RMSNorm 或
对应融合实现、SiLU/门控融合、RoPE，以及源码/运行路径确认后出现的其他
真实组。必须先用实际 import、DSO、generated artifact 或 ELF 确认入口，
不得在 Python forward 计时，也不得在猜测的未运行函数上落点。

L3 是实现内部 3–4 个真实阶段的解释层，本轮没有授权实现。

## 3. 点位控制与测试轮次

L1/L2 是观察层，site ID 是区分不同测量区间的标签，两者不是同一概念。
因此点位编号可以多于两个。

约定的最小扰动方式是：一次编译包含全部必要 L1/L2 点位，每轮只通过
轻量 TLS selector 启用一个 exact site。未选中点位只执行一次 selector
判断；被选中点位执行两次 `ReadCoreCycle()`，并在 end 之后记录。无需为
每个点位重新编译。只有未来获得授权并实际证明 points-on/off 的 TPOT
中位数扰动超过 3% 时，才考虑拆分构建；不能预先扩展多套方案。

## 4. 不可变计时合同

唯一原始计时函数是：

```text
isb sy
mrs PMCCNTR_EL0
isb sy
```

每个区间严格遵循：

```text
begin = ReadCoreCycle()
真实目标代码
end = ReadCoreCycle()
end 之后 record
```

目标线程必须固定 CPU，仅在目标 CPU 启用 PMCCNTR。缓冲区预分配、预触
页、单生产者；热区禁止字符串、日志、锁、动态分配、文件 I/O 和 syscall。
Python 只能做冷路径控制、原始记录校验和明确频率下的 cycles→µs 派生，
不能读时钟或生成原始时延。

## 5. 六个冻结工具文件

只能存在以下六个工具文件，不得新增 Probe、Control、Adapter、第二个
Recorder 或第四个辅助文件：

| 文件 | 唯一职责 |
| --- | --- |
| `csrc/read_core_cycle.h` | 定义唯一 `ReadCoreCycle()` 和 PMCCNTR 汇编 |
| `csrc/read_core_cycle_profile.h` | 编译 profile 与 exact-site TLS selector |
| `csrc/read_core_cycle_recorder.h` | 32-byte event、预分配单生产者缓冲、lost 计数 |
| `csrc/read_core_cycle_runtime.h` | 冷路径 C ABI 与共享提交函数声明 |
| `csrc/read_core_cycle_runtime.cpp` | 唯一 recorder 实例、start/stop/count/停后 CSV |
| `tools/collect_readcorecycle.py` | 同线程冷控制、原始 CSV 校验、停后 µs 派生 |

这六个文件已被用户锁定。后续只在真实目标实现源码中增加/调整 timing
site；除非用户明确批准修改工具 ABI，否则不得改动它们。

静态唯一性核验结果（2026-08-25）：

- `PMCCNTR_EL0` 只出现在 `csrc/read_core_cycle.h`
- event/recorder 定义只出现在 `csrc/read_core_cycle_recorder.h`
- 没有 `ReadCoreCycleProbe`、`DirectPmuProbe`、`OperatorCycleProbe`、
  `ReadCoreCycleControl` 或 Adapter
- Python collector 没有 Python/C/CUDA 计时源
- 六个文件相对提交 `5c2a47143` 无变化

## 6. 当前打点完成度

### 已完成：L1

文件：`pytorch/aten/src/ATen/cuda/CUDAGraph.cpp`。site 340/341/342 的
begin、真实目标代码、end、end 后 record 顺序正确，没有冗余工具实现。

### 未完成：L2

当前只有三处候选 site：

| site | 文件 | 当前边界 | 审计结论 |
| --- | --- | --- | --- |
| 400 | `pytorch/aten/src/ATen/cuda/CUDABlas.cpp` | BF16 `cublasSetMathMode`、`cublasGemmEx`、恢复 math mode | 有效局部 host API 区间，但尚未证明是统一的 L2 实现入口 |
| 450 | `csrc/libtorch_stable/cache_kernels.cu` | `reshape_and_cache_flash_kernel` launch | 只覆盖 launcher，尚未覆盖完整实现入口 |
| 460 | `flash_attention/csrc/flash_attn/flash_api.cpp` | `run_mha_fwd` | 接近实现组边界，但仍需实际 runtime 路径确认 |

RMSNorm/融合 Norm、SiLU/门控融合、RoPE 等尚未落点。因此不得对外称
“L1 和 L2 已完成”，也不得用这三个候选点冒充全 L2 覆盖。

下一会话应先做只读 runtime/import/DSO/ELF 入口确认，再以同一 L2 整体
边界修正或补充最少点位；完成后回看 begin/end/record 顺序和全树重复实现。
未经用户再次授权，不得因此开始构建或实测。

## 7. 固定工作流

1. 只在本地 `vllm_ops` 修改源码。
2. 先运行 `docs/contributing/readcorecycle-instrumentation.md` 中的静态重复
   门禁，核对差异和点位边界。
3. 提交到本地 `main` 并推送 GitHub `origin/main`。
4. server-84 只在 `/home/fj/vllm_ops` 执行干净检查和
   `git pull --ff-only origin main`；不得上传旁路副本、嵌套仓库或新建旧任务目录。
5. 只有用户明确说“开始实测”后，才在拉取完成的服务器源码上先编译，
   再启动推理任务。编译不计入推理时延区间。
6. 若服务器无法访问 GitHub，停止并汇报；未经批准不得改成其他传输流程。

## 8. 交接时的干净状态

- server-84 `/home/fj` 顶层仅有 `project_archive`、`vllm_fj`、`vllm_ops`
- server-84 的 `vllm_ops` 工作树干净，当前停在 `5500c8b7b`；2026-08-25
  执行 HTTPS `git pull --ff-only` 因服务器没有 GitHub 凭据而失败。不得把
  本机密钥复制过去或擅自旁路传文件；需由用户授权服务器认证方式后再拉取
- 旧 `script` 中两个 Topdown 脚本已可恢复地归档到
  `/home/fj/project_archive/script_topdown_v011_20260818`
- 本次错误实测产生的 server 临时构建、seed、source 和 evidence 已删除
- 本机对应临时 seed、bundle、ABI smoke、R3 smoke 已移入废纸篓
- 没有保留本次错误实测结果；历史旧项目产物不属于本仓库，也不得引用
- 没有启动或恢复旧目标

## 9. Git 基线

- 上游基础：vLLM 0.26 源码线
- 工具定稿提交：`5c2a47143`
- 此交接前仓库基线：`5500c8b7b`
- 首次纠偏交接提交：`b6c19a299`
- 最终以 `origin/main` 的最新提交为准；服务器认证恢复后必须 fast-forward
  拉取，拉取前不得构建或运行
