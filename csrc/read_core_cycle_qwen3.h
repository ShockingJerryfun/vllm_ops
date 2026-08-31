#pragma once

#include "read_core_cycle_runtime.h"

namespace vllm::instrumentation {

constexpr std::uint16_t kQwen3GraphSite = 700;

[[gnu::always_inline]] inline bool ReadCoreCycleQwen3FineStage(
    std::uint16_t stage_id) noexcept {
  return stage_id >= 11001 && stage_id <= 11903;
}

[[gnu::always_inline]] inline std::uint16_t ReadCoreCycleQwen3PhysicalRoot(
    std::uint16_t stage_id) noexcept {
  if (stage_id >= 11001 && stage_id <= 11012) {
    return 1016;
  }
  if (stage_id >= 11101 && stage_id <= 11112) {
    return 1038;
  }
  if (stage_id >= 11201 && stage_id <= 11212) {
    return 1047;
  }
  if (stage_id >= 11301 && stage_id <= 11312) {
    return 1048;
  }
  if (stage_id >= 11401 && stage_id <= 11412) {
    return 1049;
  }
  if (stage_id >= 11501 && stage_id <= 11512) {
    return 1050;
  }
  if (stage_id >= 11601 && stage_id <= 11612) {
    return 1051;
  }
  if (stage_id >= 11701 && stage_id <= 11714) {
    return 1057;
  }
  if (stage_id >= 11801 && stage_id <= 11822) {
    return 1039;
  }
  if (stage_id >= 11901 && stage_id <= 11903) {
    return 1017;
  }
  return stage_id;
}

// Fine-stage selection must first pass through the exact semantic leaf in the
// dispatcher.  During that leaf call, published_tail_stage is reserved as the
// physical-root context.  Existing tail probes cannot select concurrently
// because the runtime permits one exact Stage per round.
class ReadCoreCycleQwen3LeafScope final {
 public:
  explicit ReadCoreCycleQwen3LeafScope(bool selected) noexcept {
    const auto exact_stage = read_core_cycle_selection.exact_stage;
    if (!selected || !ReadCoreCycleSiteSelected(kQwen3GraphSite) ||
        !ReadCoreCycleQwen3FineStage(exact_stage)) {
      return;
    }
    previous_root_ = read_core_cycle_selection.published_tail_stage;
    read_core_cycle_selection.published_tail_stage =
        ReadCoreCycleQwen3PhysicalRoot(exact_stage);
    active_ = true;
  }

  ~ReadCoreCycleQwen3LeafScope() {
    if (active_) {
      read_core_cycle_selection.published_tail_stage = previous_root_;
    }
  }

 private:
  std::uint16_t previous_root_{0};
  bool active_{false};
};

[[gnu::always_inline]] inline std::uint16_t
ReadCoreCycleQwen3SelectedLeafRoot() noexcept {
  if (!ReadCoreCycleSiteSelected(kQwen3GraphSite) ||
      !ReadCoreCycleQwen3FineStage(
          read_core_cycle_selection.exact_stage)) {
    return 0;
  }
  return read_core_cycle_selection.published_tail_stage;
}

class ReadCoreCycleQwen3SubleafScope final {
 public:
  explicit ReadCoreCycleQwen3SubleafScope(
      std::uint16_t subleaf_id) noexcept {
    if (ReadCoreCycleQwen3SelectedLeafRoot() != 1039) {
      return;
    }
    previous_subleaf_ =
        read_core_cycle_selection.published_frontend_stage;
    read_core_cycle_selection.published_frontend_stage = subleaf_id;
    active_ = true;
  }

  ~ReadCoreCycleQwen3SubleafScope() {
    if (active_) {
      read_core_cycle_selection.published_frontend_stage =
          previous_subleaf_;
    }
  }

 private:
  std::uint16_t previous_subleaf_{0};
  bool active_{false};
};

[[gnu::always_inline]] inline std::uint16_t
ReadCoreCycleQwen3LeafStageId(std::uint16_t offset) noexcept {
  const auto root = ReadCoreCycleQwen3SelectedLeafRoot();
  switch (root) {
    case 1016: return static_cast<std::uint16_t>(11000 + offset);
    case 1038: return static_cast<std::uint16_t>(11100 + offset);
    case 1047: return static_cast<std::uint16_t>(11200 + offset);
    case 1048: return static_cast<std::uint16_t>(11300 + offset);
    case 1049: return static_cast<std::uint16_t>(11400 + offset);
    case 1050: return static_cast<std::uint16_t>(11500 + offset);
    case 1051: return static_cast<std::uint16_t>(11600 + offset);
    case 1057: return static_cast<std::uint16_t>(11700 + offset);
    default: return 0;
  }
}

// FlashAttention has its own branch/fill Stage namespace.  Do not route it
// through ReadCoreCycleQwen3LeafStageId(): allocator/copy helpers invoked below
// mha_varlen_fwd also use generic leaf offsets and would otherwise alias those
// nested calls onto 11801..11822.
[[gnu::always_inline]] inline std::uint16_t
ReadCoreCycleQwen3FlashStageId(std::uint16_t stage_id) noexcept {
  if (ReadCoreCycleQwen3SelectedLeafRoot() != 1039 ||
      stage_id < 11801 || stage_id > 11822) {
    return 0;
  }
  return stage_id;
}

[[gnu::always_inline]] inline std::uint16_t
ReadCoreCycleQwen3LeafStageId(
    std::uint16_t expected_root, std::uint16_t offset) noexcept {
  if (ReadCoreCycleQwen3SelectedLeafRoot() != expected_root) {
    return 0;
  }
  return ReadCoreCycleQwen3LeafStageId(offset);
}

[[gnu::always_inline]] inline std::uint16_t
ReadCoreCycleQwen3FillStageId(std::uint16_t offset) noexcept {
  if (ReadCoreCycleQwen3SelectedLeafRoot() != 1039 ||
      offset == 0 || offset > 4) {
    return 0;
  }
  switch (read_core_cycle_selection.published_frontend_stage) {
    case 1: return static_cast<std::uint16_t>(11801 + offset);
    case 2: return static_cast<std::uint16_t>(11805 + offset);
    case 3: return static_cast<std::uint16_t>(11809 + offset);
    case 4: return static_cast<std::uint16_t>(11814 + offset);
    case 5: return static_cast<std::uint16_t>(11818 + offset);
    default: return 0;
  }
}

class ReadCoreCycleQwen3StageGuard final {
 public:
  explicit ReadCoreCycleQwen3StageGuard(std::uint16_t stage_id) noexcept
      : stage_id_(stage_id),
        selected_(stage_id != 0 && ReadCoreCycleStageSelected(
            kQwen3GraphSite, stage_id)) {
    if (selected_) {
      begin_ = ReadCoreCycleBoundaryNow();
    }
  }

  ~ReadCoreCycleQwen3StageGuard() {
    Finish();
  }

  void Finish() noexcept {
    if (selected_) {
      const auto end = ReadCoreCycleBoundaryNow();
      CommitReadCoreCyclePairedSample(
          kQwen3GraphSite, stage_id_, begin_, end);
      selected_ = false;
    }
  }

 private:
  std::uint16_t stage_id_{0};
  bool selected_{false};
  ReadCoreCycleBoundary begin_{};
};

}  // namespace vllm::instrumentation
