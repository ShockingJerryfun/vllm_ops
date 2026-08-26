#pragma once

#include <cstdint>

namespace vllm::instrumentation {

#if defined(__aarch64__)
[[gnu::always_inline]] inline std::uint64_t ReadCoreCycle() noexcept {
  std::uint64_t cycles = 0;
  asm volatile(
      "isb sy\n\t"
      "mrs %0, PMCCNTR_EL0\n\t"
      "isb sy\n\t"
      : "=r"(cycles)
      :
      : "memory");
  return cycles;
}
#else
std::uint64_t ReadCoreCycle() = delete;
#endif

}  // namespace vllm::instrumentation
