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
// glibc calls them through its PLT too, so blocks it hands out itself
// (strdup, getline, fopen's buffers, reallocarray) pass through here.
//
// Reco code allocates millions of small blocks per event, so every instruction
// here shows in its timing: the counting is kept to a few loads and adds.
//
// Ownership. A process started with K4BENCH_ALLOC_OWNERS=1 also remembers who
// allocated each block. Every thread has a current owner, a small integer the
// auditor sets to the component running on it (0 is no component), and every
// block handed out is recorded in a table with it. A block freed later, by any
// thread and whatever owner is current then, is given back to the owner that
// allocated it, so an owner's held bytes are the heap it allocated and has not
// freed yet, and its peak the most it held at once since the reader last reset
// it. A resized block keeps its owner. Memory goes to whoever created it, not
// to whoever uses it. A block allocated before ownership started (before this
// library's constructor ran) or while counting was paused is not recorded,
// and freeing it changes nothing: the table only ever forgets blocks it knows.
//
// The table lives inside malloc, so it never calls malloc: its memory is mapped
// with mmap, and a spinlock guards it and the owners' totals. Every allocation
// and free then takes the lock and looks its block up in a table far larger
// than the caches, so ownership is for a run of its own, never a timed one.
// Measured on lxplus (AMD EPYC 7702, one pinned thread) per malloc+free pair of
// 16-256 byte blocks freed in the order they were allocated: glibc alone takes
// 18 ns while a thousand blocks are live and 57 ns while a million are;
// counting adds 4 and 12 ns, ownership another 17 and 127 ns (66 ns at a
// hundred thousand, 175 ns at four million, as the table outgrows the caches).
// With ownership off, the counting runs exactly as without it but for one
// branch on the flag.

#include "k4BenchAllocCounter.h"

#include <malloc.h>
#include <pthread.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <utility>

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
  [[gnu::tls_model("initial-exec")]] thread_local std::uint32_t current_owner = 0;

  // Whether blocks are recorded with their owners; set once at load.
  bool owners_enabled = false;

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

  // Whether glibc mapped the block on its own (mmap), from the same size word.
  bool is_mapped(const void *block) noexcept
  {
    constexpr std::size_t kIsMmapped = 0x2;
    return (static_cast<const std::size_t *>(block)[-1] & kIsMmapped) != 0;
  }

  // Owners the totals hold. The auditor numbers a job's components from 1, and
  // reconstruction jobs audit fewer than a hundred.
  constexpr std::uint32_t kOwners = 1024;

  // Table slots when the first block is recorded; the table doubles before it
  // is half full, which keeps linear probing short.
  constexpr std::size_t kInitialSlots = std::size_t{1} << 16;

  // One recorded block: its address (0 marks an empty slot), its heap memory
  // when recorded and its owner.
  struct Entry
  {
    std::uintptr_t block;
    std::uint64_t bytes : 48;
    std::uint64_t owner : 16;
  };
  static_assert(sizeof(Entry) == 16);
  static_assert(kOwners <= 1U << 16);

  class SpinLock
  {
  public:
    void lock() noexcept
    {
      while (m_flag.test_and_set(std::memory_order_acquire))
      {
        while (m_flag.test(std::memory_order_relaxed))
        {
#if defined(__x86_64__)
          __builtin_ia32_pause();
#endif
        }
      }
    }
    void unlock() noexcept { m_flag.clear(std::memory_order_release); }

  private:
    std::atomic_flag m_flag;
  };

  struct OwnerTotals
  {
    std::int64_t held{0};
    std::int64_t peak{0};
  };

  // The blocks recorded with their owners, by address, and each owner's
  // totals. An open-addressing table with linear probing, removing entries by
  // shifting later ones back, so it holds no tombstones.
  class Ownership
  {
  public:
    // Records the new *block* for *owner*.
    [[gnu::noinline]] void record(const void *block, std::uint32_t owner) noexcept
    {
      const std::lock_guard guard(m_lock);
      insert(entry_for(block, owner));
    }

    // Forgets *block* and gives its memory back to its owner.
    [[gnu::noinline]] void release(const void *block) noexcept
    {
      const std::lock_guard guard(m_lock);
      if (Entry *slot = find(key(block)); slot != nullptr)
      {
        m_totals[slot->owner].held -= static_cast<std::int64_t>(slot->bytes);
        remove(slot);
      }
    }

    // Forgets *block* before realloc may free it, keeping what it was for
    // resized() or, when realloc fails, put_back().
    [[gnu::noinline]] std::optional<Entry> take(const void *block) noexcept
    {
      const std::lock_guard guard(m_lock);
      Entry *slot = find(key(block));
      if (slot == nullptr)
      {
        return std::nullopt;
      }
      const Entry entry = *slot;
      remove(slot);
      return entry;
    }

    void put_back(const Entry &entry) noexcept
    {
      const std::lock_guard guard(m_lock);
      insert(entry, /*charge=*/false);
    }

    // Records *block*, the *old* one resized, for the same owner: after giving
    // the old memory back when the two were never held at once, before when
    // realloc copied from one to the other. A null *block* was freed.
    void resized(const Entry &old, const void *block, bool freed_first) noexcept
    {
      const std::lock_guard guard(m_lock);
      const auto old_bytes = static_cast<std::int64_t>(old.bytes);
      if (freed_first)
      {
        m_totals[old.owner].held -= old_bytes;
      }
      if (block != nullptr)
      {
        insert(entry_for(block, old.owner));
      }
      if (!freed_first)
      {
        m_totals[old.owner].held -= old_bytes;
      }
    }

    std::size_t read(std::int64_t *held, std::int64_t *peak, std::size_t capacity) noexcept
    {
      const std::lock_guard guard(m_lock);
      for (std::size_t owner = 0; owner < std::min<std::size_t>(capacity, m_used); ++owner)
      {
        held[owner] = m_totals[owner].held;
        if (peak != nullptr)
        {
          peak[owner] = m_totals[owner].peak;
        }
      }
      return m_used;
    }

    void reset_peaks() noexcept
    {
      const std::lock_guard guard(m_lock);
      for (std::size_t owner = 0; owner < m_used; ++owner)
      {
        m_totals[owner].peak = m_totals[owner].held;
      }
    }

    void describe(k4bench::OwnerTable &table) noexcept
    {
      const std::lock_guard guard(m_lock);
      table = {m_entries, m_peakEntries, m_slots * sizeof(Entry), kOwners, m_failed};
    }

    // Held across fork(), so the child copies the table whole rather than as
    // another thread left it mid-change.
    void lock_for_fork() noexcept { m_lock.lock(); }
    void unlock_after_fork() noexcept { m_lock.unlock(); }

  private:
    static std::uintptr_t key(const void *block) noexcept { return reinterpret_cast<std::uintptr_t>(block); }

    static Entry entry_for(const void *block, std::uint32_t owner) noexcept
    {
      Entry entry{key(block), 0, 0};
      entry.bytes = heap_bytes(block);
      entry.owner = owner;
      return entry;
    }

    // Fibonacci hashing of the address; glibc blocks are 16-byte aligned, so
    // the low four bits carry nothing.
    std::size_t home(std::uintptr_t block) const noexcept
    {
      return static_cast<std::size_t>(((block >> 4) * 0x9E3779B97F4A7C15ULL) >> m_shift);
    }

    // The slot holding *block*, or the empty slot that ends its probe sequence.
    Entry *slot_for(std::uintptr_t block) noexcept
    {
      for (std::size_t i = home(block);; i = (i + 1) & (m_slots - 1))
      {
        if (m_table[i].block == block || m_table[i].block == 0)
        {
          return &m_table[i];
        }
      }
    }

    Entry *find(std::uintptr_t block) noexcept
    {
      if (m_slots == 0)
      {
        return nullptr;
      }
      Entry *slot = slot_for(block);
      return slot->block != 0 ? slot : nullptr;
    }

    // Charges the entry's memory to its owner unless *charge* is false. A
    // block still in the table at the same address was freed without passing
    // through here, so its owner is given its memory back first.
    void insert(const Entry &entry, bool charge = true) noexcept
    {
      if (m_failed || (2 * (m_entries + 1) > m_slots && !grow()))
      {
        return;
      }
      Entry *slot = slot_for(entry.block);
      if (slot->block != 0)
      {
        m_totals[slot->owner].held -= static_cast<std::int64_t>(slot->bytes);
      }
      else
      {
        m_peakEntries = std::max(m_peakEntries, ++m_entries);
      }
      *slot = entry;
      m_used = std::max<std::size_t>(m_used, entry.owner + 1);
      if (charge)
      {
        OwnerTotals &totals = m_totals[entry.owner];
        totals.held += static_cast<std::int64_t>(entry.bytes);
        totals.peak = std::max(totals.peak, totals.held);
      }
    }

    // Empties *slot*, moving back each later entry of the run whose probe
    // sequence passes the hole.
    void remove(Entry *slot) noexcept
    {
      const std::size_t mask = m_slots - 1;
      auto hole = static_cast<std::size_t>(slot - m_table);
      for (std::size_t i = (hole + 1) & mask; m_table[i].block != 0; i = (i + 1) & mask)
      {
        if (((i - home(m_table[i].block)) & mask) >= ((i - hole) & mask))
        {
          m_table[hole] = m_table[i];
          hole = i;
        }
      }
      m_table[hole].block = 0;
      --m_entries;
    }

    // Moves the entries into a table twice the size, mapped from the kernel.
    // When the kernel refuses, recording stops for good: owners would
    // otherwise miss blocks without anyone knowing.
    bool grow() noexcept
    {
      const int saved_errno = errno; // a malloc that succeeds leaves errno alone
      const std::size_t slots = m_slots == 0 ? kInitialSlots : 2 * m_slots;
      void *region = ::mmap(nullptr, slots * sizeof(Entry), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (region == MAP_FAILED)
      {
        m_failed = true;
        errno = saved_errno;
        return false;
      }
      // Huge pages spare most TLB misses in a table this large.
      ::madvise(region, slots * sizeof(Entry), MADV_HUGEPAGE);
      Entry *old = std::exchange(m_table, static_cast<Entry *>(region));
      const std::size_t old_slots = std::exchange(m_slots, slots);
      m_shift = 64 - std::countr_zero(slots);
      for (std::size_t i = 0; i < old_slots; ++i)
      {
        if (old[i].block != 0)
        {
          *slot_for(old[i].block) = old[i];
        }
      }
      if (old != nullptr)
      {
        ::munmap(old, old_slots * sizeof(Entry));
      }
      errno = saved_errno;
      return true;
    }

    SpinLock m_lock;
    Entry *m_table{nullptr};
    std::size_t m_slots{0};
    int m_shift{64};
    std::size_t m_entries{0};
    std::size_t m_peakEntries{0};
    std::size_t m_used{0}; // one past the highest owner recorded
    bool m_failed{false};
    OwnerTotals m_totals[kOwners]{};
  };

  constinit Ownership ownership;

  // Read once, before the job's own code runs. Preloaded first, this library
  // registers its fork handlers before any other: glibc runs prepare handlers
  // last registered first, the others in order, so the table is locked after
  // and released before every other handler, none of which can find it held.
  [[gnu::constructor]] void start_ownership() noexcept
  {
    const char *enabled = std::getenv("K4BENCH_ALLOC_OWNERS");
    if (enabled != nullptr && std::strcmp(enabled, "1") == 0)
    {
      ::pthread_atfork([] { ownership.lock_for_fork(); }, [] { ownership.unlock_after_fork(); },
                       [] { ownership.unlock_after_fork(); });
      owners_enabled = true;
    }
  }

  // Adds a new block to the calling thread's counters.
  [[gnu::always_inline]] inline void count(const void *block) noexcept
  {
    if (block != nullptr && !counters.paused)
    {
      const std::uint64_t size = heap_bytes(block);
      ++counters.allocations;
      counters.allocated_bytes += size;
      counters.peak_live_bytes = std::max(counters.peak_live_bytes, counters.live_bytes());
      counters.largest_allocation_bytes = std::max(counters.largest_allocation_bytes, size);
    }
  }

  // Counts a new block and records it for the thread's current owner.
  [[gnu::always_inline]] inline void *counted(void *block) noexcept
  {
    count(block);
    if (owners_enabled && block != nullptr && !counters.paused) [[unlikely]]
    {
      ownership.record(block, current_owner);
    }
    return block;
  }

  // Ownership forgets a block even while counting is paused, so no owner
  // keeps memory that was given back.
  [[gnu::always_inline]] inline void count_free(void *block) noexcept
  {
    if (block == nullptr)
    {
      return;
    }
    if (owners_enabled) [[unlikely]]
    {
      ownership.release(block);
    }
    if (!counters.paused)
    {
      counters.freed_bytes += heap_bytes(block);
    }
  }
} // namespace

extern "C"
{
  k4bench::AllocCounters *k4bench_thread_alloc_counters() noexcept { return &counters; }

  std::uint32_t k4bench_exchange_alloc_owner(std::uint32_t owner) noexcept
  {
    return std::exchange(current_owner, owner < kOwners ? owner : 0);
  }

  std::size_t k4bench_read_owners(std::int64_t *held, std::int64_t *peak, std::size_t capacity) noexcept
  {
    return owners_enabled ? ownership.read(held, peak, capacity) : 0;
  }

  void k4bench_reset_owner_peaks() noexcept
  {
    if (owners_enabled)
    {
      ownership.reset_peaks();
    }
  }

  bool k4bench_owner_table(k4bench::OwnerTable *table) noexcept
  {
    if (owners_enabled)
    {
      ownership.describe(*table);
    }
    return owners_enabled;
  }

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

  // A moved heap block is counted as a new block allocated and then the old
  // one freed, as both are held while realloc copies. A block resized in place
  // never coexists with its old size, so that is freed first: counted the
  // other way round, the peak would hold the old size twice. Nor does a block
  // mapped on its own, which glibc moves with mremap rather than copying.
  // Ownership takes a recorded block out of its table before glibc may free
  // it, so a block another thread is handed at the same address meanwhile is
  // never mistaken for it, and records the result for the same owner.
  void *realloc(void *block, std::size_t size) noexcept
  {
    const std::size_t old_size = block != nullptr ? heap_bytes(block) : 0;
    const bool mapped = block != nullptr && is_mapped(block);
    std::optional<Entry> owned;
    if (owners_enabled && block != nullptr) [[unlikely]]
    {
      owned = ownership.take(block);
    }
    void *resized = __libc_realloc(block, size);
    if (resized == nullptr && size != 0)
    {
      if (owned)
      {
        ownership.put_back(*owned);
      }
      return nullptr; // failed, and the old block is untouched
    }
    const bool freed_first = resized == block || mapped;
    if (freed_first && !counters.paused)
    {
      counters.freed_bytes += old_size;
    }
    if (owned)
    {
      count(resized);
      ownership.resized(*owned, resized, freed_first);
    }
    else
    {
      counted(resized);
    }
    if (!freed_first && !counters.paused)
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
