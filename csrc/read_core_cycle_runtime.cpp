#include "read_core_cycle_runtime.h"

#include <cerrno>
#include <cstdio>

namespace vllm::instrumentation {
namespace {

constexpr std::uint32_t kToolAbiVersion = 2;
constexpr std::size_t kMaxEvents = 1ULL << 20;
ReadCoreCycleRecorder<kMaxEvents> recorder;

}  // namespace

thread_local ReadCoreCycleSelection read_core_cycle_selection;

void CommitReadCoreCycleSample(
    std::uint16_t site_id,
    std::uint16_t stage_id,
    std::uint64_t begin,
    std::uint64_t end) noexcept {
  if (!ReadCoreCycleSiteSelected(site_id)) {
    return;
  }
  recorder.RecordAfterEnd(
      site_id,
      stage_id,
      read_core_cycle_selection.context_id,
      begin,
      end);
}

}  // namespace vllm::instrumentation

extern "C" {

std::uint32_t rcc_tool_abi_version() noexcept {
  return vllm::instrumentation::kToolAbiVersion;
}

int rcc_start(
    std::uint16_t exact_site,
    std::uint32_t context_id,
    std::uint64_t capacity) noexcept {
  using namespace vllm::instrumentation;
  if (read_core_cycle_selection.active) {
    return EBUSY;
  }
  if (exact_site == 0 || capacity == 0 || capacity > kMaxEvents ||
      !recorder.Prepare(static_cast<std::size_t>(capacity))) {
    return EINVAL;
  }
  read_core_cycle_selection = ReadCoreCycleSelection{};
  read_core_cycle_selection.exact_site = exact_site;
  read_core_cycle_selection.context_id = context_id;
  read_core_cycle_selection.active = true;
  return 0;
}

int rcc_start_selected(
    std::uint16_t exact_site,
    std::uint16_t exact_stage,
    std::uint32_t context_id,
    std::uint64_t capacity) noexcept {
  using namespace vllm::instrumentation;
  if (read_core_cycle_selection.active) {
    return EBUSY;
  }
  if (exact_site == 0 || exact_stage == 0 || capacity == 0 ||
      capacity > kMaxEvents ||
      !recorder.Prepare(static_cast<std::size_t>(capacity))) {
    return EINVAL;
  }
  read_core_cycle_selection = ReadCoreCycleSelection{};
  read_core_cycle_selection.exact_site = exact_site;
  read_core_cycle_selection.exact_stage = exact_stage;
  read_core_cycle_selection.context_id = context_id;
  read_core_cycle_selection.active = true;
  return 0;
}

int rcc_stop() noexcept {
  vllm::instrumentation::read_core_cycle_selection.active = false;
  return 0;
}

std::uint64_t rcc_selected_count() noexcept {
  using namespace vllm::instrumentation;
  return static_cast<std::uint64_t>(
      recorder.event_count() + recorder.lost_count());
}

std::uint64_t rcc_event_count() noexcept {
  return static_cast<std::uint64_t>(
      vllm::instrumentation::recorder.event_count());
}

std::uint64_t rcc_lost_count() noexcept {
  return static_cast<std::uint64_t>(
      vllm::instrumentation::recorder.lost_count());
}

int rcc_dump_csv(const char* output_path) noexcept {
  using namespace vllm::instrumentation;
  if (output_path == nullptr || output_path[0] == '\0' ||
      read_core_cycle_selection.active) {
    return EINVAL;
  }
  FILE* output = std::fopen(output_path, "wx");
  if (output == nullptr) {
    return errno;
  }
  if (std::fputs(
          "sequence,begin_cycle,end_cycle,delta_cycles,context_id,"
          "site_id,stage_id\n",
          output) == EOF) {
    const int error = errno == 0 ? EIO : errno;
    std::fclose(output);
    return error;
  }
  const std::size_t count = recorder.event_count();
  for (std::size_t index = 0; index < count; ++index) {
    const ReadCoreCycleEvent& event = recorder.data()[index];
    if (std::fprintf(
            output,
            "%llu,%llu,%llu,%llu,%u,%u,%u\n",
            static_cast<unsigned long long>(index),
            static_cast<unsigned long long>(event.begin_cycle),
            static_cast<unsigned long long>(event.end_cycle),
            static_cast<unsigned long long>(event.delta_cycles),
            static_cast<unsigned int>(event.context_id),
            static_cast<unsigned int>(event.site_id),
            static_cast<unsigned int>(event.stage_id)) < 0) {
      const int error = errno == 0 ? EIO : errno;
      std::fclose(output);
      return error;
    }
  }
  if (std::fclose(output) != 0) {
    return errno;
  }
  return 0;
}

}  // extern "C"
