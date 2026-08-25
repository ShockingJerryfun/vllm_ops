#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vllm::instrumentation {

struct ReadCoreCycleEvent final {
  std::uint64_t delta_cycles;
  std::uint32_t context_id;
  std::uint16_t site_id;
  std::uint16_t stage_id;
};
static_assert(sizeof(ReadCoreCycleEvent) == 16);

template <std::size_t MaxEvents>
class ReadCoreCycleRecorder final {
  static_assert(MaxEvents > 0);

 public:
  // Use static storage. Prepare pre-touches the single-producer buffer.
  bool Prepare(std::size_t capacity) noexcept {
    if (capacity == 0 || capacity > MaxEvents) {
      return false;
    }
    std::memset(events_, 0, capacity * sizeof(ReadCoreCycleEvent));
    capacity_ = capacity;
    cursor_ = 0;
    lost_count_ = 0;
    return true;
  }

  void Reset() noexcept {
    cursor_ = 0;
    lost_count_ = 0;
  }

  [[gnu::always_inline]] inline void RecordAfterEnd(
      std::uint16_t site_id,
      std::uint16_t stage_id,
      std::uint32_t context_id,
      std::uint64_t begin,
      std::uint64_t end) noexcept {
    const std::size_t index = cursor_;
    if (index >= capacity_) {
      ++lost_count_;
      return;
    }
    events_[index] =
        ReadCoreCycleEvent{end - begin, context_id, site_id, stage_id};
    cursor_ = index + 1;
  }

  const ReadCoreCycleEvent* data() const noexcept {
    return events_;
  }

  std::size_t capacity() const noexcept {
    return capacity_;
  }

  std::size_t event_count() const noexcept {
    return cursor_;
  }

  std::size_t lost_count() const noexcept {
    return lost_count_;
  }

 private:
  alignas(64) ReadCoreCycleEvent events_[MaxEvents]{};
  std::size_t capacity_{0};
  std::size_t cursor_{0};
  std::size_t lost_count_{0};
};

}  // namespace vllm::instrumentation
