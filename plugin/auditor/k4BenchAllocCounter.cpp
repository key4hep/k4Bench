// k4BenchAllocCounter.cpp
//
// Counts the heap allocations of every thread for the k4BenchAuditor.
// Preloaded into a job (LD_PRELOAD), it replaces the C allocation functions
// with ones that forward to glibc's allocator and add to the calling thread's
// counters: one allocation and the heap memory it takes for every block handed
// out, and the heap memory of every block freed. Since the auditor last reset
// them, it also keeps the highest the difference of the two has reached and
// the largest block. C++ new and delete allocate through malloc, so they are
// counted too. Memory a library maps from the kernel itself (mmap), as custom
// pool allocators and CPython's small-object arenas do, is not.
//
// glibc 2.34 removed the malloc hooks, so preloading is the only way in. The
// functions below are the ones glibc documents as replaceable, less
// malloc_usable_size, which glibc's own still answers for glibc's blocks.
//
// Reco code allocates millions of small blocks per event, so every instruction
// here shows in its timing: the counting is kept to a few loads and adds.

#include "k4BenchAllocCounter.h"

#include <malloc.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstddef>

// glibc's allocator, which the replacements forward to.
extern "C"
{
  void *__libc_malloc(std::size_t size) noexcept;
  void *__libc_calloc(std::size_t count, std::size_t size) noexcept;
  void *__libc_realloc(void *block, std::size_t size) noexcept;
  void *__libc_memalign(std::size_t alignment, std::size_t size) noexcept;
  void *__libc_valloc(std::size_t size) noexcept;
  void *__libc_pvalloc(std::size_t size) noexcept;
  void __libc_free(void *block) noexcept;
}

namespace
{
  // A preloaded library's thread-local storage is allocated with each thread,
  // so the initial-exec model reads it without calling malloc.
  [[gnu::tls_model("initial-exec")]] thread_local k4bench::AllocCounters counters;

  // The heap memory a glibc block takes: the size of its chunk, malloc's own
  // header included, from the size word glibc keeps just before the block,
  // whose low three bits are flags. malloc and free read that word anyway,
  // while malloc_usable_size would also read the next chunk's header and
  // double the counting's cost.
  std::size_t heap_bytes(const void *block) noexcept
  {
    constexpr std::size_t kFlags = 0x7;
    return static_cast<const std::size_t *>(block)[-1] & ~kFlags;
  }

  [[gnu::always_inline]] inline void *counted(void *block) noexcept
  {
    if (block != nullptr && !counters.paused)
    {
      const std::uint64_t size = heap_bytes(block);
      ++counters.allocations;
      counters.allocated_bytes += size;
      counters.peak_live_bytes = std::max(counters.peak_live_bytes, counters.live_bytes());
      counters.largest_allocation_bytes = std::max(counters.largest_allocation_bytes, size);
    }
    return block;
  }

  [[gnu::always_inline]] inline void count_free(void *block) noexcept
  {
    if (block != nullptr && !counters.paused)
    {
      counters.freed_bytes += heap_bytes(block);
    }
  }
} // namespace

extern "C"
{
  k4bench::AllocCounters *k4bench_thread_alloc_counters() noexcept { return &counters; }

  void *malloc(std::size_t size) noexcept { return counted(__libc_malloc(size)); }
  void *calloc(std::size_t count, std::size_t size) noexcept { return counted(__libc_calloc(count, size)); }
  void *memalign(std::size_t alignment, std::size_t size) noexcept
  {
    return counted(__libc_memalign(alignment, size));
  }
  void *aligned_alloc(std::size_t alignment, std::size_t size) noexcept
  {
    return counted(__libc_memalign(alignment, size));
  }
  void *valloc(std::size_t size) noexcept { return counted(__libc_valloc(size)); }
  void *pvalloc(std::size_t size) noexcept { return counted(__libc_pvalloc(size)); }

  int posix_memalign(void **block, std::size_t alignment, std::size_t size) noexcept
  {
    if (alignment % sizeof(void *) != 0 || !std::has_single_bit(alignment))
    {
      return EINVAL;
    }
    void *aligned = counted(__libc_memalign(alignment, size));
    if (aligned == nullptr)
    {
      return ENOMEM;
    }
    *block = aligned;
    return 0;
  }

  // A moved block is counted as a new block allocated and then the old one
  // freed, as both are held while realloc copies. A block resized in place
  // never coexists with its old size, so that is freed first: counted the
  // other way round, the peak would hold the old size twice.
  void *realloc(void *block, std::size_t size) noexcept
  {
    const std::size_t old_size = block != nullptr ? heap_bytes(block) : 0;
    void *resized = __libc_realloc(block, size);
    if (resized == nullptr && size != 0)
    {
      return nullptr; // failed, and the old block is untouched
    }
    const bool in_place = resized == block;
    if (in_place && !counters.paused)
    {
      counters.freed_bytes += old_size;
    }
    counted(resized);
    if (!in_place && !counters.paused)
    {
      counters.freed_bytes += old_size;
    }
    return resized;
  }

  void free(void *block) noexcept
  {
    count_free(block);
    __libc_free(block);
  }
}
