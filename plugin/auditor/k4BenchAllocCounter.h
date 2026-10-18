// k4BenchAllocCounter.h
//
// Per-thread heap allocation counters, kept by libk4BenchAllocCounter.so when
// it is preloaded into a job and read by the k4BenchAuditor, and the heap each
// owner holds, kept when the job starts with K4BENCH_ALLOC_OWNERS=1.

#pragma once

#include <cstddef>
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
    // other threads allocated, and too high while blocks it allocated were
    // freed by other threads, which then also raise peak_live_bytes.
    std::int64_t live_bytes() const { return static_cast<std::int64_t>(allocated_bytes - freed_bytes); }
  };

  // What the table of owned blocks costs.
  struct OwnerTable
  {
    std::uint64_t entries{0};      // blocks recorded now
    std::uint64_t peak_entries{0}; // most recorded at once
    std::uint64_t mapped_bytes{0}; // memory the table maps
    std::uint32_t owners{0};       // owners the totals hold; a higher owner counts as 0
    bool failed{false};            // the table could not grow, and stopped recording blocks
  };
} // namespace k4bench

extern "C"
{
  // The calling thread's counters.
  k4bench::AllocCounters *k4bench_thread_alloc_counters() noexcept;

  // Makes *owner* the calling thread's owner, charged with the blocks it
  // allocates from now on, and returns the previous one. Owner 0 is no one.
  std::uint32_t k4bench_exchange_alloc_owner(std::uint32_t owner) noexcept;

  // Copies the live heap bytes of the first *capacity* owners, and the most
  // each held since the last k4bench_reset_owner_peaks(), into *held* and
  // *peak*, unless *peak* is null. Returns the number of owners that have held
  // blocks, 0 when ownership is off.
  std::size_t k4bench_read_owners(std::int64_t *held, std::int64_t *peak, std::size_t capacity) noexcept;

  // Starts every owner's peak again from what it holds now.
  void k4bench_reset_owner_peaks() noexcept;

  // Fills *table* and returns true, or returns false when ownership is off.
  bool k4bench_owner_table(k4bench::OwnerTable *table) noexcept;
}
