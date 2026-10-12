// k4BenchProcStats.h
//
// Process memory readings from /proc/self (Linux only), shared by every
// k4Bench plugin so that sim and reco runs measure memory identically.
//
// A reading that fails is reported as -1 rather than 0, so a reader can tell a
// missing sample from an empty process.

#pragma once

#include <time.h>
#include <unistd.h>

#include <fstream>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>

namespace k4bench
{

  // Ephemeral handoff to the runner, outside every timed region. Hidden files
  // are never uploaded; run_info reduces their union to manifest packages.
  inline void write_loaded_paths() noexcept
  {
    try
    {
      const char *target = std::getenv("K4BENCH_LOADED_PATHS");
      if (!target) return;
      std::ifstream maps("/proc/self/maps");
      if (!maps) return;
      std::set<std::string> paths;
      std::string line;
      while (std::getline(maps, line))
      {
        std::istringstream fields(line);
        std::string field;
        for (int i = 0; i < 5; ++i) fields >> field;
        std::getline(fields >> std::ws, field);
        if (!field.empty() && field.front() == '/') paths.insert(field);
      }
      if (maps.bad()) return;
      std::ofstream out(target);
      out << "# k4bench loaded paths v1\n";
      for (const auto &path : paths) out << path << '\n';
      for (char **variable = ::environ; *variable; ++variable)
      {
        const std::string entry(*variable);
        const auto equals = entry.find('=');
        if (entry.rfind("G4", 0) == 0 && equals >= 6 &&
            entry.substr(equals - 4, 4) == "DATA") out << entry << '\n';
      }
    }
    catch (...) {}
  }

  struct RssValues
  {
    long total{-1};
    long anon{-1};
    long file{-1};
  };

  // Current RSS and its anonymous and file-backed parts, in kB, from
  // /proc/self/status.
  inline RssValues read_rss_kb()
  {
    RssValues values;
    std::ifstream status("/proc/self/status");
    std::string key;
    std::string line;

    while (std::getline(status, line))
    {
      std::istringstream iss(line);
      long kb;
      if (iss >> key >> kb)
      {
        if (key == "VmRSS:")
        {
          values.total = kb;
        }
        else if (key == "RssAnon:")
        {
          values.anon = kb;
        }
        else if (key == "RssFile:")
        {
          values.file = kb;
        }
      }
    }

    return values;
  }

  // Kernel high-water mark of the process's virtual size, in kB.
  inline long read_vmpeak_kb()
  {
    std::ifstream status("/proc/self/status");
    std::string line;

    while (std::getline(status, line))
    {
      if (line.rfind("VmPeak:", 0) == 0)
      {
        std::istringstream iss(line.substr(7));

        long kb = -1;
        return (iss >> kb) ? kb : -1;
      }
    }

    return -1;
  }

  // When this process started, in ns since the Unix epoch, from the start time
  // in /proc/self/stat (clock ticks after boot). It is placed on the epoch by
  // reading the boot-time and real-time clocks together, as the boot time in
  // /proc/stat is truncated to whole seconds. Resolution is one clock tick,
  // typically 10 ms.
  inline long long read_process_start_epoch_ns()
  {
    std::ifstream stat("/proc/self/stat");
    std::string line;
    if (!std::getline(stat, line))
    {
      return -1;
    }
    // The command name in field 2 may contain spaces; fields after it don't.
    const auto close = line.rfind(')');
    if (close == std::string::npos)
    {
      return -1;
    }
    std::istringstream fields(line.substr(close + 1));
    std::string field;
    long long start_ticks = -1;
    // starttime is field 22; the stream starts at field 3.
    for (int i = 3; i <= 22 && fields >> field; ++i)
    {
      if (i == 22)
      {
        start_ticks = std::stoll(field);
      }
    }

    // The start time counts from boot on the boot-time clock, so the boot
    // time on the epoch is the real-time clock less the boot-time clock.
    timespec since_boot{};
    timespec since_epoch{};
    const long ticks_per_s = sysconf(_SC_CLK_TCK);
    if (start_ticks < 0 || ticks_per_s <= 0 || clock_gettime(CLOCK_BOOTTIME, &since_boot) != 0 ||
        clock_gettime(CLOCK_REALTIME, &since_epoch) != 0)
    {
      return -1;
    }
    const auto ns = [](const timespec &t) { return t.tv_sec * 1'000'000'000LL + t.tv_nsec; };
    return ns(since_epoch) - ns(since_boot) + start_ticks * (1'000'000'000LL / ticks_per_s);
  }

} // namespace k4bench
