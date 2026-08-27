#pragma once

#include "read_core_cycle.h"
#include "read_core_cycle_profile.h"
#include "read_core_cycle_recorder.h"

namespace vllm::instrumentation {

extern thread_local std::uint32_t read_core_cycle_frontend_depth;

[[gnu::always_inline]] inline void CommitPublishedReadCoreCycleFrontend(
    std::uint16_t site_id) noexcept;

class ReadCoreCycleFrontendChainGuard final {
 public:
  explicit ReadCoreCycleFrontendChainGuard(bool selected) noexcept
      : selected_(selected) {}

  ~ReadCoreCycleFrontendChainGuard() {
    if (selected_ && read_core_cycle_frontend_depth != 0) {
      if (read_core_cycle_frontend_depth == 1) {
        CommitPublishedReadCoreCycleFrontend(700);
      }
      --read_core_cycle_frontend_depth;
    }
  }

 private:
  bool selected_;
};

VLLM_RCC_PUBLIC void CommitReadCoreCycleSample(
    std::uint16_t site_id,
    std::uint16_t stage_id,
    std::uint64_t begin,
    std::uint64_t end) noexcept;

[[gnu::always_inline]] inline void PublishReadCoreCycleFrontendBegin(
    std::uint16_t site_id, std::uint16_t stage_id) noexcept {
  if (!ReadCoreCycleStageSelected(site_id, stage_id)) {
    return;
  }
  ++read_core_cycle_frontend_depth;
  if (read_core_cycle_frontend_depth != 1) {
    return;
  }
  read_core_cycle_selection.published_frontend_begin = ReadCoreCycle();
  read_core_cycle_selection.published_frontend_stage = stage_id;
}

[[gnu::always_inline]] inline void CommitPublishedReadCoreCycleFrontend(
    std::uint16_t site_id) noexcept {
  if (!ReadCoreCycleSiteSelected(site_id) ||
      read_core_cycle_selection.published_frontend_stage == 0) {
    return;
  }
  const std::uint64_t end = ReadCoreCycle();
  const std::uint16_t stage_id =
      read_core_cycle_selection.published_frontend_stage;
  const std::uint64_t begin =
      read_core_cycle_selection.published_frontend_begin;
  read_core_cycle_selection.published_frontend_stage = 0;
  read_core_cycle_selection.published_frontend_begin = 0;
  CommitReadCoreCycleSample(site_id, stage_id, begin, end);
}

[[gnu::always_inline]] inline void PublishReadCoreCycleTailBegin(
    std::uint16_t site_id, std::uint16_t stage_id) noexcept {
  if (!ReadCoreCycleStageSelected(site_id, stage_id)) {
    return;
  }
  read_core_cycle_selection.published_tail_begin = ReadCoreCycle();
  read_core_cycle_selection.published_tail_stage = stage_id;
}

[[gnu::always_inline]] inline void CommitPublishedReadCoreCycleTail(
    std::uint16_t site_id) noexcept {
  if (!ReadCoreCycleSiteSelected(site_id) ||
      read_core_cycle_selection.published_tail_stage == 0) {
    return;
  }
  const std::uint64_t end = ReadCoreCycle();
  const std::uint16_t stage_id =
      read_core_cycle_selection.published_tail_stage;
  const std::uint64_t begin =
      read_core_cycle_selection.published_tail_begin;
  read_core_cycle_selection.published_tail_stage = 0;
  read_core_cycle_selection.published_tail_begin = 0;
  CommitReadCoreCycleSample(site_id, stage_id, begin, end);
}

}  // namespace vllm::instrumentation

extern "C" {

VLLM_RCC_PUBLIC std::uint32_t rcc_tool_abi_version() noexcept;
VLLM_RCC_PUBLIC int rcc_start(
    std::uint16_t exact_site,
    std::uint32_t context_id,
    std::uint64_t capacity) noexcept;
VLLM_RCC_PUBLIC int rcc_start_selected(
    std::uint16_t exact_site,
    std::uint16_t exact_stage,
    std::uint32_t context_id,
    std::uint64_t capacity) noexcept;
VLLM_RCC_PUBLIC int rcc_stop() noexcept;
VLLM_RCC_PUBLIC std::uint64_t rcc_selected_count() noexcept;
VLLM_RCC_PUBLIC std::uint64_t rcc_event_count() noexcept;
VLLM_RCC_PUBLIC std::uint64_t rcc_lost_count() noexcept;
VLLM_RCC_PUBLIC int rcc_dump_csv(const char* output_path) noexcept;

}
