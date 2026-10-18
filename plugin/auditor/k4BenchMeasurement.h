// k4BenchMeasurement.h
//
// What the k4BenchAuditor reads on each side of an audited call, and what it
// makes of two reads: the wall clock, getrusage, the hardware performance
// counters and the allocation counter of the calling thread, and the cost of
// the call between them.

#pragma once

#include "k4BenchAllocCounter.h"

#include <dlfcn.h>
#include <linux/perf_event.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

namespace k4bench
{
  using Clock = std::chrono::steady_clock;

  // The calling thread's allocation counters, or nullptr when the job does not
  // preload libk4BenchAllocCounter.so. Looked up at run time, so the auditor
  // neither links nor needs it.
  inline AllocCounters *alloc_counters()
  {
    using Getter = AllocCounters *(*)() noexcept;
    static const auto getter = reinterpret_cast<Getter>(::dlsym(RTLD_DEFAULT, "k4bench_thread_alloc_counters"));
    return getter != nullptr ? getter() : nullptr;
  }

  // Keeps the calling thread's allocations out of the counts while it lives.
  class PausedAllocCounting
  {
  public:
    PausedAllocCounting() : m_counters(alloc_counters())
    {
      if (m_counters != nullptr)
      {
        m_wasPaused = std::exchange(m_counters->paused, true);
      }
    }
    ~PausedAllocCounting()
    {
      if (m_counters != nullptr)
      {
        m_counters->paused = m_wasPaused;
      }
    }
    PausedAllocCounting(const PausedAllocCounting &) = delete;
    PausedAllocCounting &operator=(const PausedAllocCounting &) = delete;

  private:
    AllocCounters *m_counters;
    bool m_wasPaused{false};
  };

  // The allocation counter's ownership functions (k4BenchAllocCounter.h),
  // looked up at run time like its counters. Empty unless the job preloads the
  // counter and started it with K4BENCH_ALLOC_OWNERS=1.
  struct Owners
  {
    std::uint32_t (*exchange)(std::uint32_t) noexcept{nullptr};
    std::size_t (*read)(std::int64_t *, std::int64_t *, std::size_t) noexcept{nullptr};
    void (*reset_peaks)() noexcept{nullptr};
    bool (*table)(OwnerTable *) noexcept{nullptr};

    static Owners find()
    {
      const auto symbol = [](const char *name) { return ::dlsym(RTLD_DEFAULT, name); };
      Owners owners{reinterpret_cast<decltype(exchange)>(symbol("k4bench_exchange_alloc_owner")),
                    reinterpret_cast<decltype(read)>(symbol("k4bench_read_owners")),
                    reinterpret_cast<decltype(reset_peaks)>(symbol("k4bench_reset_owner_peaks")),
                    reinterpret_cast<decltype(table)>(symbol("k4bench_owner_table"))};
      OwnerTable stats;
      const bool found = owners.exchange && owners.read && owners.reset_peaks && owners.table;
      return found && owners.table(&stats) ? owners : Owners{};
    }

    OwnerTable describe() const
    {
      OwnerTable stats;
      table(&stats);
      return stats;
    }

    explicit operator bool() const { return exchange != nullptr; }
  };

  struct HardwareCounts
  {
    std::uint64_t instructions{0};
    std::uint64_t cycles{0};
  };

  // The user-space instructions and cycles of one thread, counted by a perf
  // event group the kernel switches with the thread. The group is pinned, so it
  // counts all the time or not at all: when the counters cannot be opened (a
  // virtual machine without them, perf_event_paranoid above 2, a container's
  // seccomp policy) or are taken by another user, reads fail and the counts are
  // missing rather than estimated. podman's default seccomp profile allows
  // perf_event_open only to a container granted CAP_PERFMON; one without it
  // gets ENOSYS ("Function not implemented"), or EPERM without CAP_SYS_ADMIN.
  class PerfGroup
  {
  public:
    PerfGroup()
    {
      m_leader = openCounter(PERF_COUNT_HW_INSTRUCTIONS, -1);
      if (m_leader >= 0)
      {
        m_member = openCounter(PERF_COUNT_HW_CPU_CYCLES, m_leader);
      }
      if (m_member < 0)
      {
        m_error = errno;
      }
    }
    ~PerfGroup()
    {
      for (const int fd : {m_member, m_leader})
      {
        if (fd >= 0)
        {
          ::close(fd);
        }
      }
    }
    PerfGroup(const PerfGroup &) = delete;
    PerfGroup &operator=(const PerfGroup &) = delete;

    std::optional<HardwareCounts> read() const
    {
      struct
      {
        std::uint64_t count;
        std::uint64_t values[2];
      } group{};
      if (m_member < 0 || ::read(m_leader, &group, sizeof(group)) != sizeof(group))
      {
        return std::nullopt;
      }
      return HardwareCounts{group.values[0], group.values[1]};
    }

    // Why reads fail: the error opening the counters, or that the kernel did not
    // schedule them.
    std::string failure() const
    {
      return m_error != 0 ? std::string("perf_event_open: ") + std::strerror(m_error) : "counters not scheduled";
    }

  private:
    static int openCounter(std::uint64_t config, int leader)
    {
      perf_event_attr attr{};
      attr.size = sizeof(attr);
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = config;
      attr.read_format = PERF_FORMAT_GROUP;
      attr.pinned = leader < 0; // only a group's leader can be
      attr.exclude_kernel = 1;
      attr.exclude_hv = 1;
      return static_cast<int>(::syscall(SYS_perf_event_open, &attr, 0, -1, leader, PERF_FLAG_FD_CLOEXEC));
    }

    int m_leader{-1};
    int m_member{-1};
    int m_error{0};
  };

  // The calling thread's group, opened on its first use and closed with it.
  inline const PerfGroup &perf_group()
  {
    thread_local const PerfGroup group;
    return group;
  }

  // What one side of a call reads. A single getrusage(RUSAGE_THREAD) gives the
  // thread's CPU time (user + system, microsecond resolution), page faults and
  // context switches, and the process's peak RSS, at a fraction of the cost of
  // reading /proc.
  struct Sample
  {
    Clock::time_point wall;
    rusage usage{};
    AllocCounters alloc;           // zero without the allocation counter
    std::optional<HardwareCounts> hardware; // missing without the hardware counters
  };

  inline Sample sample()
  {
    Sample s;
    s.wall = Clock::now();
    ::getrusage(RUSAGE_THREAD, &s.usage);
    if (const auto *counters = alloc_counters())
    {
      s.alloc = *counters;
    }
    s.hardware = perf_group().read();
    return s;
  }

  inline std::int64_t cpu_us(const rusage &usage)
  {
    return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1'000'000LL + usage.ru_utime.tv_usec +
           usage.ru_stime.tv_usec;
  }

  // What a call cost, or several calls of one component combined.
  struct Cost
  {
    // Self costs, which add up.
    double wall_s{0.0};
    double cpu_s{0.0};
    double peak_rss_increase_mb{0.0};
    std::int64_t minor_page_faults{0};
    std::int64_t major_page_faults{0};
    std::int64_t voluntary_context_switches{0};
    std::int64_t involuntary_context_switches{0};
    std::int64_t instructions{0};
    std::int64_t cycles{0};
    std::int64_t allocations{0};
    std::int64_t allocated_bytes{0};
    std::int64_t net_allocated_bytes{0}; // allocated minus freed
    // Inclusive, the largest of several calls.
    std::int64_t peak_heap_bytes{0};
    std::int64_t largest_allocation_bytes{0};
    // False once a sample of the call or of one nested in it had no hardware counts.
    bool hardware_counted{true};

    Cost &operator+=(const Cost &other)
    {
      wall_s += other.wall_s;
      cpu_s += other.cpu_s;
      peak_rss_increase_mb += other.peak_rss_increase_mb;
      minor_page_faults += other.minor_page_faults;
      major_page_faults += other.major_page_faults;
      voluntary_context_switches += other.voluntary_context_switches;
      involuntary_context_switches += other.involuntary_context_switches;
      instructions += other.instructions;
      cycles += other.cycles;
      allocations += other.allocations;
      allocated_bytes += other.allocated_bytes;
      net_allocated_bytes += other.net_allocated_bytes;
      peak_heap_bytes = std::max(peak_heap_bytes, other.peak_heap_bytes);
      largest_allocation_bytes = std::max(largest_allocation_bytes, other.largest_allocation_bytes);
      hardware_counted = hardware_counted && other.hardware_counted;
      return *this;
    }

    // Leaves the inclusive costs alone: the nested calls ran during this one.
    Cost &operator-=(const Cost &other)
    {
      wall_s -= other.wall_s;
      cpu_s -= other.cpu_s;
      peak_rss_increase_mb -= other.peak_rss_increase_mb;
      minor_page_faults -= other.minor_page_faults;
      major_page_faults -= other.major_page_faults;
      voluntary_context_switches -= other.voluntary_context_switches;
      involuntary_context_switches -= other.involuntary_context_switches;
      instructions -= other.instructions;
      cycles -= other.cycles;
      allocations -= other.allocations;
      allocated_bytes -= other.allocated_bytes;
      net_allocated_bytes -= other.net_allocated_bytes;
      hardware_counted = hardware_counted && other.hardware_counted;
      return *this;
    }
  };

  inline Cost cost_between(const Sample &begin, const Sample &end)
  {
    const auto delta = [](auto from, auto to) { return static_cast<std::int64_t>(to - from); };
    const rusage &b = begin.usage;
    const rusage &e = end.usage;
    Cost cost;
    cost.wall_s = std::chrono::duration<double>(end.wall - begin.wall).count();
    cost.cpu_s = static_cast<double>(cpu_us(e) - cpu_us(b)) * 1e-6;
    cost.peak_rss_increase_mb = static_cast<double>(e.ru_maxrss - b.ru_maxrss) / 1024.0;
    cost.minor_page_faults = delta(b.ru_minflt, e.ru_minflt);
    cost.major_page_faults = delta(b.ru_majflt, e.ru_majflt);
    cost.voluntary_context_switches = delta(b.ru_nvcsw, e.ru_nvcsw);
    cost.involuntary_context_switches = delta(b.ru_nivcsw, e.ru_nivcsw);
    if (begin.hardware && end.hardware)
    {
      cost.instructions = delta(begin.hardware->instructions, end.hardware->instructions);
      cost.cycles = delta(begin.hardware->cycles, end.hardware->cycles);
    }
    else
    {
      cost.hardware_counted = false;
    }
    cost.allocations = delta(begin.alloc.allocations, end.alloc.allocations);
    cost.allocated_bytes = delta(begin.alloc.allocated_bytes, end.alloc.allocated_bytes);
    cost.net_allocated_bytes = cost.allocated_bytes - delta(begin.alloc.freed_bytes, end.alloc.freed_bytes);
    cost.peak_heap_bytes = std::max<std::int64_t>(0, end.alloc.peak_live_bytes - begin.alloc.live_bytes());
    cost.largest_allocation_bytes = static_cast<std::int64_t>(end.alloc.largest_allocation_bytes);
    return cost;
  }

  // The allocation counters a call's window resets: its heap peak and largest
  // block are what the counters keep over the call.
  struct AllocWindow
  {
    std::int64_t peak_live_bytes{0};
    std::uint64_t largest_allocation_bytes{0};
  };

  // Starts the call's window at the live heap and no block yet, and returns
  // the enclosing call's window so far.
  inline AllocWindow open_alloc_window(const Sample &begin)
  {
    auto *counters = alloc_counters();
    if (counters == nullptr)
    {
      return {};
    }
    return {std::exchange(counters->peak_live_bytes, begin.alloc.live_bytes()),
            std::exchange(counters->largest_allocation_bytes, 0)};
  }

  // The enclosing call held whatever this one did, so its window continues
  // from the larger of the two.
  inline void close_alloc_window(const AllocWindow &outer, const Sample &end)
  {
    if (auto *counters = alloc_counters())
    {
      counters->peak_live_bytes = std::max(outer.peak_live_bytes, end.alloc.peak_live_bytes);
      counters->largest_allocation_bytes =
          std::max(outer.largest_allocation_bytes, end.alloc.largest_allocation_bytes);
    }
  }

  // A component called more than once in one event or phase is charged the sum.
  inline void accumulate(std::optional<Cost> &cell, const Cost &cost)
  {
    if (cell)
    {
      *cell += cost;
    }
    else
    {
      cell = cost;
    }
  }
} // namespace k4bench
