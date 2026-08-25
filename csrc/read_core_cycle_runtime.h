#pragma once

#include "read_core_cycle.h"
#include "read_core_cycle_profile.h"
#include "read_core_cycle_recorder.h"

namespace vllm::instrumentation {

VLLM_RCC_PUBLIC void CommitReadCoreCycleSample(
    std::uint16_t site_id,
    std::uint16_t stage_id,
    std::uint64_t begin,
    std::uint64_t end) noexcept;

}  // namespace vllm::instrumentation

extern "C" {

VLLM_RCC_PUBLIC std::uint32_t rcc_tool_abi_version() noexcept;
VLLM_RCC_PUBLIC int rcc_start(
    std::uint16_t exact_site,
    std::uint32_t context_id,
    std::uint64_t capacity) noexcept;
VLLM_RCC_PUBLIC int rcc_stop() noexcept;
VLLM_RCC_PUBLIC std::uint64_t rcc_selected_count() noexcept;
VLLM_RCC_PUBLIC std::uint64_t rcc_event_count() noexcept;
VLLM_RCC_PUBLIC std::uint64_t rcc_lost_count() noexcept;
VLLM_RCC_PUBLIC int rcc_dump_csv(const char* output_path) noexcept;

}
