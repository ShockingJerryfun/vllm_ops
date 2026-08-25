#include <c10/util/ReadCoreCycleProbe.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace c10::rcc {
namespace {

constexpr std::uint64_t kMaxEvents = 1ULL << 20;

struct ThreadState final {
  std::uint16_t exact_site{0};
  std::uint32_t context_id{0};
  std::uint64_t capacity{0};
  std::uint64_t cursor{0};
  std::uint64_t selected_count{0};
  std::uint64_t lost_count{0};
  bool active{false};
};

alignas(64) Event g_events[kMaxEvents];
thread_local ThreadState g_state;

bool IsKnownSite(std::uint16_t site_id) noexcept {
  switch (site_id) {
    case RCC_SITE_REPLAY_R1:
    case RCC_SITE_REPLAY_R2:
    case RCC_SITE_REPLAY_R3:
    case RCC_SITE_GEMM_CUBLAS_GEMM_EX:
    case RCC_SITE_KV_CACHE_IMPL:
    case RCC_SITE_FA2_RUN_MHA_FWD:
      return true;
    default:
      return false;
  }
}

const char* SiteName(std::uint16_t site_id) noexcept {
  switch (site_id) {
    case RCC_SITE_REPLAY_R1:
      return "replay_device_guard_and_generator_prologue";
    case RCC_SITE_REPLAY_R2:
      return "replay_get_current_cuda_stream";
    case RCC_SITE_REPLAY_R3:
      return "replay_cuda_graph_launch";
    case RCC_SITE_GEMM_CUBLAS_GEMM_EX:
      return "gemm_cublas_gemm_ex";
    case RCC_SITE_KV_CACHE_IMPL:
      return "reshape_and_cache_flash_impl";
    case RCC_SITE_FA2_RUN_MHA_FWD:
      return "fa2_run_mha_fwd";
    default:
      return "unknown";
  }
}

} // namespace

bool Selected(std::uint16_t site_id) noexcept {
  return g_state.active && g_state.exact_site == site_id;
}

void RecordAfterEnd(
    std::uint16_t site_id,
    std::uint16_t stage_id,
    std::uint64_t begin,
    std::uint64_t end) noexcept {
  ++g_state.selected_count;
  if (g_state.cursor >= g_state.capacity) {
    ++g_state.lost_count;
    return;
  }
  g_events[g_state.cursor++] =
      Event{end - begin, g_state.context_id, site_id, stage_id};
}

} // namespace c10::rcc

extern "C" {

int rcc_start(
    std::uint16_t exact_site,
    std::uint32_t context_id,
    std::uint64_t capacity) noexcept {
  if (c10::rcc::g_state.active) {
    return EBUSY;
  }
  if (!c10::rcc::IsKnownSite(exact_site) || capacity == 0 ||
      capacity > c10::rcc::kMaxEvents) {
    return EINVAL;
  }

  // Pre-touch before activation. The hot path never allocates or faults pages.
  std::memset(
      c10::rcc::g_events,
      0,
      capacity * sizeof(c10::rcc::Event));
  c10::rcc::g_state = c10::rcc::ThreadState{};
  c10::rcc::g_state.exact_site = exact_site;
  c10::rcc::g_state.context_id = context_id;
  c10::rcc::g_state.capacity = capacity;
  c10::rcc::g_state.active = true;
  return 0;
}

int rcc_stop() noexcept {
  c10::rcc::g_state.active = false;
  c10::rcc::g_state.exact_site = 0;
  return 0;
}

std::uint64_t rcc_selected_count() noexcept {
  return c10::rcc::g_state.selected_count;
}

std::uint64_t rcc_event_count() noexcept {
  return std::min(
      c10::rcc::g_state.cursor, c10::rcc::g_state.capacity);
}

std::uint64_t rcc_lost_count() noexcept {
  return c10::rcc::g_state.lost_count;
}

int rcc_dump_csv(const char* output_path) noexcept {
  if (output_path == nullptr || output_path[0] == '\0' ||
      c10::rcc::g_state.active) {
    return EINVAL;
  }
  FILE* output = std::fopen(output_path, "wx");
  if (output == nullptr) {
    return errno;
  }
  std::fputs(
      "sequence,delta_cycles,context_id,site_id,site,stage_id\n", output);
  const std::uint64_t count = rcc_event_count();
  for (std::uint64_t index = 0; index < count; ++index) {
    const auto& event = c10::rcc::g_events[index];
    std::fprintf(
        output,
        "%llu,%llu,%u,%u,%s,%u\n",
        static_cast<unsigned long long>(index),
        static_cast<unsigned long long>(event.delta_cycles),
        static_cast<unsigned int>(event.context_id),
        static_cast<unsigned int>(event.site_id),
        c10::rcc::SiteName(event.site_id),
        static_cast<unsigned int>(event.stage_id));
  }
  if (std::fclose(output) != 0) {
    return errno;
  }
  return 0;
}

} // extern "C"
