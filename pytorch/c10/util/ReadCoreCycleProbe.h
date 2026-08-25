#pragma once

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>

#include <cstdint>

// Exactly one site is compiled into a measurement build. Site 0 removes all
// guarded probes at preprocessing time.
#ifndef RCC_COMPILED_SITE
#define RCC_COMPILED_SITE 0
#endif

#define RCC_SITE_REPLAY_R1 340
#define RCC_SITE_REPLAY_R2 341
#define RCC_SITE_REPLAY_R3 342
#define RCC_SITE_GEMM_CUBLAS_GEMM_EX 400
#define RCC_SITE_KV_CACHE_IMPL 450
#define RCC_SITE_FA2_RUN_MHA_FWD 460

#define RCC_SITE_ENABLED(site_id) (RCC_COMPILED_SITE == (site_id))

namespace c10::rcc {

struct Event final {
  std::uint64_t delta_cycles;
  std::uint32_t context_id;
  std::uint16_t site_id;
  std::uint16_t stage_id;
};
static_assert(sizeof(Event) == 16, "ReadCoreCycle event must stay 16 bytes");

// The required serialized PMCCNTR_EL0 reader. It returns raw processor cycles.
#if defined(__aarch64__)
C10_ALWAYS_INLINE std::uint64_t ReadCoreCycle() noexcept {
  std::uint64_t cycle_num = 0;
  asm volatile(
      "isb sy\n\t"
      "mrs %0, PMCCNTR_EL0\n\t"
      "isb sy\n\t"
      : "=r"(cycle_num)
      :
      : "memory");
  return cycle_num;
}
#else
std::uint64_t ReadCoreCycle() = delete;
#endif

// Only the one compiled site performs this lightweight TLS selector check.
C10_API bool Selected(std::uint16_t site_id) noexcept;

// Called strictly after the ending ReadCoreCycle().
C10_API void RecordAfterEnd(
    std::uint16_t site_id,
    std::uint16_t stage_id,
    std::uint64_t begin,
    std::uint64_t end) noexcept;

} // namespace c10::rcc

extern "C" {

// Start/stop/dump run outside measured regions on the pinned producer thread.
C10_API int rcc_start(
    std::uint16_t exact_site,
    std::uint32_t context_id,
    std::uint64_t capacity) noexcept;
C10_API int rcc_stop() noexcept;
C10_API std::uint64_t rcc_selected_count() noexcept;
C10_API std::uint64_t rcc_event_count() noexcept;
C10_API std::uint64_t rcc_lost_count() noexcept;
C10_API int rcc_dump_csv(const char* output_path) noexcept;

}
