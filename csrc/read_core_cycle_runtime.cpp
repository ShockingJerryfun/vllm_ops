#include "read_core_cycle_runtime.h"

#include <cerrno>
#include <cstdio>

namespace vllm::instrumentation {
namespace {

constexpr std::uint32_t kToolAbiVersion = 4;
constexpr std::size_t kMaxEvents = 1ULL << 20;
ReadCoreCycleRecorder<kMaxEvents> recorder;
thread_local bool read_core_cycle_prepared{false};

}  // namespace

thread_local ReadCoreCycleSelection read_core_cycle_selection;
thread_local std::uint32_t read_core_cycle_frontend_depth;
thread_local std::int64_t read_core_cycle_step_id{-1};
thread_local ReadCoreCycleBoundaryHistory read_core_cycle_boundary_history;

namespace {

bool FindMonotonicTick(
    std::uint64_t cycle, std::uint64_t* monotonic_tick) noexcept {
  if (monotonic_tick == nullptr) {
    return false;
  }
  for (const ReadCoreCycleBoundary& boundary :
       read_core_cycle_boundary_history.entries) {
    if (boundary.cycle == cycle && boundary.monotonic_tick != 0) {
      *monotonic_tick = boundary.monotonic_tick;
      return true;
    }
  }
  return false;
}

}  // namespace

void CommitReadCoreCycleSample(
    std::uint16_t site_id,
    std::uint16_t stage_id,
    std::uint64_t begin,
    std::uint64_t end) noexcept {
  if (!ReadCoreCycleStageSelected(site_id, stage_id)) {
    return;
  }
  std::uint64_t begin_monotonic_tick = 0;
  std::uint64_t end_monotonic_tick = 0;
  if (!FindMonotonicTick(begin, &begin_monotonic_tick) ||
      !FindMonotonicTick(end, &end_monotonic_tick) ||
      end_monotonic_tick <= begin_monotonic_tick) {
    recorder.RecordLost();
    return;
  }
  recorder.RecordAfterEnd(
      site_id,
      stage_id,
      read_core_cycle_selection.context_id,
      begin,
      end,
      begin_monotonic_tick,
      end_monotonic_tick,
      ReadMonotonicCounterFrequency(),
      read_core_cycle_step_id);
}

void CommitReadCoreCyclePairedSample(
    std::uint16_t site_id,
    std::uint16_t stage_id,
    const ReadCoreCycleBoundary& begin,
    const ReadCoreCycleBoundary& end) noexcept {
  if (!ReadCoreCycleStageSelected(site_id, stage_id)) {
    return;
  }
  if (end.cycle <= begin.cycle ||
      end.monotonic_tick <= begin.monotonic_tick) {
    recorder.RecordLost();
    return;
  }
  recorder.RecordAfterEnd(
      site_id,
      stage_id,
      read_core_cycle_selection.context_id,
      begin.cycle,
      end.cycle,
      begin.monotonic_tick,
      end.monotonic_tick,
      ReadMonotonicCounterFrequency(),
      read_core_cycle_step_id);
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
  read_core_cycle_frontend_depth = 0;
  read_core_cycle_step_id = -1;
  read_core_cycle_boundary_history = ReadCoreCycleBoundaryHistory{};
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
  const int prepare_result = rcc_prepare_selected(
      exact_site, exact_stage, context_id, capacity);
  if (prepare_result != 0) {
    return prepare_result;
  }
  return rcc_arm_prepared();
}

int rcc_prepare_selected(
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
  read_core_cycle_frontend_depth = 0;
  read_core_cycle_step_id = -1;
  read_core_cycle_boundary_history = ReadCoreCycleBoundaryHistory{};
  read_core_cycle_selection.exact_site = exact_site;
  read_core_cycle_selection.exact_stage = exact_stage;
  read_core_cycle_selection.context_id = context_id;
  read_core_cycle_prepared = true;
  return 0;
}

int rcc_arm_prepared() noexcept {
  using namespace vllm::instrumentation;
  if (read_core_cycle_selection.active || !read_core_cycle_prepared ||
      read_core_cycle_selection.exact_site == 0 ||
      read_core_cycle_selection.exact_stage == 0) {
    return EINVAL;
  }
  recorder.Reset();
  read_core_cycle_frontend_depth = 0;
  read_core_cycle_step_id = -1;
  read_core_cycle_boundary_history = ReadCoreCycleBoundaryHistory{};
  read_core_cycle_selection.published_frontend_begin = 0;
  read_core_cycle_selection.published_tail_begin = 0;
  read_core_cycle_selection.published_frontend_stage = 0;
  read_core_cycle_selection.published_tail_stage = 0;
  read_core_cycle_selection.active = true;
  return 0;
}

int rcc_stop() noexcept {
  vllm::instrumentation::read_core_cycle_selection.active = false;
  vllm::instrumentation::read_core_cycle_frontend_depth = 0;
  return 0;
}

int rcc_set_step(std::int64_t step_id) noexcept {
  using namespace vllm::instrumentation;
  if (!read_core_cycle_selection.active || step_id < 0) {
    return EINVAL;
  }
  read_core_cycle_step_id = step_id;
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
          "site_id,stage_id,begin_monotonic_tick,end_monotonic_tick,"
          "delta_monotonic_ticks,monotonic_frequency_hz,"
          "delta_monotonic_ns,step_id\n",
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
            "%llu,%llu,%llu,%llu,%u,%u,%u,%llu,%llu,%llu,%llu,%.3f,%lld\n",
            static_cast<unsigned long long>(index),
            static_cast<unsigned long long>(event.begin_cycle),
            static_cast<unsigned long long>(event.end_cycle),
            static_cast<unsigned long long>(event.delta_cycles),
            static_cast<unsigned int>(event.context_id),
            static_cast<unsigned int>(event.site_id),
            static_cast<unsigned int>(event.stage_id),
            static_cast<unsigned long long>(event.begin_monotonic_tick),
            static_cast<unsigned long long>(event.end_monotonic_tick),
            static_cast<unsigned long long>(event.delta_monotonic_ticks),
            static_cast<unsigned long long>(event.monotonic_frequency_hz),
            static_cast<double>(event.delta_monotonic_ticks) * 1.0e9 /
                static_cast<double>(event.monotonic_frequency_hz),
            static_cast<long long>(event.step_id)) < 0) {
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
