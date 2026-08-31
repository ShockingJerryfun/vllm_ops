#pragma once

#include <cstdint>

namespace vllm::instrumentation {

struct ReadCoreCycleBoundary final {
  std::uint64_t cycle{0};
  std::uint64_t monotonic_tick{0};
};

struct ReadCoreCycleBoundaryHistory final {
  ReadCoreCycleBoundary entries[8]{};
  std::uint32_t cursor{0};
};

extern thread_local ReadCoreCycleBoundaryHistory
    read_core_cycle_boundary_history;

#if defined(__aarch64__)
[[gnu::always_inline]] inline ReadCoreCycleBoundary
ReadCoreCycleBoundaryNow() noexcept {
  std::uint64_t cycles = 0;
  std::uint64_t monotonic_tick = 0;
  asm volatile(
      "isb sy\n\t"
      "mrs %0, CNTVCT_EL0\n\t"
      "mrs %1, PMCCNTR_EL0\n\t"
      "isb sy\n\t"
      : "=r"(monotonic_tick), "=r"(cycles)
      :
      : "memory");
  const std::uint32_t index =
      read_core_cycle_boundary_history.cursor++ & 7U;
  read_core_cycle_boundary_history.entries[index] =
      ReadCoreCycleBoundary{cycles, monotonic_tick};
  return ReadCoreCycleBoundary{cycles, monotonic_tick};
}

[[gnu::always_inline]] inline std::uint64_t ReadCoreCycle() noexcept {
  return ReadCoreCycleBoundaryNow().cycle;
}

[[gnu::always_inline]] inline std::uint64_t
ReadMonotonicCounterFrequency() noexcept {
  std::uint64_t frequency = 0;
  asm volatile("mrs %0, CNTFRQ_EL0" : "=r"(frequency));
  return frequency;
}
#else
ReadCoreCycleBoundary ReadCoreCycleBoundaryNow() = delete;
std::uint64_t ReadCoreCycle() = delete;
std::uint64_t ReadMonotonicCounterFrequency() = delete;
#endif

}  // namespace vllm::instrumentation
