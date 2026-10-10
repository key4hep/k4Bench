// k4BenchAllocCounter.h
//
// Per-thread heap allocation counters, kept by libk4BenchAllocCounter.so when
// it is preloaded into a job and read by the k4BenchAuditor.

#pragma once

#include <cstdint>

namespace k4bench
{
  struct AllocCounters
  {
    std::uint64_t allocations{0};     // blocks handed out
    std::uint64_t allocated_bytes{0}; // the heap memory they take, malloc's header included
    std::uint64_t freed_bytes{0};     // the heap memory of the blocks freed
    std::int64_t peak_live_bytes{0};           // most live_bytes() since the reader last reset it
    std::uint64_t largest_allocation_bytes{0}; // largest block since the reader last reset it
    bool paused{false};                        // while set, nothing is counted

    // Allocated minus freed on this thread; negative after freeing blocks
    // other threads allocated.
    std::int64_t live_bytes() const { return static_cast<std::int64_t>(allocated_bytes - freed_bytes); }
  };
} // namespace k4bench

// The calling thread's counters.
extern "C" k4bench::AllocCounters *k4bench_thread_alloc_counters() noexcept;
