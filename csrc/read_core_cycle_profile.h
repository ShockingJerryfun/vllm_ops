#pragma once

#include <cstdint>

#ifndef VLLM_RCC_PROFILE
#define VLLM_RCC_PROFILE 0
#endif

// Profile 0 compiles every guarded probe out.
#define VLLM_RCC_PROFILE_ENABLED(profile_id) \
  (VLLM_RCC_PROFILE == (profile_id))

#define VLLM_RCC_PROFILE_ALL_SITES 1

#if defined(__GNUC__)
#define VLLM_RCC_PUBLIC __attribute__((visibility("default")))
#else
#define VLLM_RCC_PUBLIC
#endif

namespace vllm::instrumentation {

struct ReadCoreCycleSelection final {
  std::uint16_t exact_site{0};
  std::uint32_t context_id{0};
  bool active{false};
};

extern VLLM_RCC_PUBLIC thread_local ReadCoreCycleSelection
    read_core_cycle_selection;

[[gnu::always_inline]] inline bool ReadCoreCycleSiteSelected(
    std::uint16_t site_id) noexcept {
  return read_core_cycle_selection.active &&
      read_core_cycle_selection.exact_site == site_id;
}

}  // namespace vllm::instrumentation
