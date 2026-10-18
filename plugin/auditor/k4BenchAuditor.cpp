// k4BenchAuditor.cpp
//
// Gaudi auditor that measures every audited call of every component — each
// algorithm execution, and each initialize, start, stop and finalize of
// algorithms and services:
//
//   - its wall time, the CPU time of its thread and how much it raised the
//     process's peak RSS;
//   - the page faults of its thread, minor (a page mapped in from memory) and
//     major (a page read from disk or CVMFS first), and its context switches,
//     voluntary (it blocked, on I/O or a lock) and involuntary (it was
//     preempted, a sign the host was busy);
//   - the user-space instructions and cycles it ran, where the hardware
//     performance counters can be read. Instructions depend on the work done,
//     not on the host's load, so they show changes too small for time to;
//   - when the job preloads libk4BenchAllocCounter.so, the heap blocks it
//     allocated, the bytes it allocated and kept, the most heap it held at
//     once and its largest single block.
//
// It knows nothing about the job it audits; components are discovered from the
// calls and described from the Gaudi managers.
//
// Every cost but the heap peak and the largest block is a *self* cost: calls
// nest (a sequencer runs its children, the event loop manager initializes the
// algorithms), and a call is charged only what was not spent in the audited
// calls nested inside it. Self costs therefore add up: summed over the
// components of an event they give the cost of its audited calls, without
// double counting at any depth. The heap peak (the most heap the call held at
// once beyond what its thread held when it started) and the largest block
// include nested calls, whose memory was held during the call too; they do not
// add up, and a component called several times in one event or phase is
// charged its largest.
//
// A non-execute call made during an execution (a service initialized on first
// use, a custom audited section) stays in the executing component's event cost
// and is also listed under its own phase, without widening that phase's span.
//
// An event's time in EventsOutput runs from its first top-level call to its
// last, so with several top-level algorithms it also covers what happens
// between them: the framework's own work and the auditor's memory read at the
// end of each top-level call. No component is charged for that; it is the
// difference between the event's time and its components' summed wall time.
//
// Nesting is tracked per thread. In a job that runs calls concurrently
// ("threads" > 1 in the output), the summed CPU time is still the event's work,
// but wall times of calls that overlapped on different threads add up to more
// than the event's elapsed time, which is EventsOutput's event time. The peak
// RSS is the process's, so a call is then also charged increases caused by
// whatever ran on other threads meanwhile; only in a serial job does a nonzero
// increase name the component that raised the high-water mark. A thread's heap
// is what it allocated less what it freed, so a block one thread allocated and
// another freed stays on the first thread's heap, and a call's heap peak may
// then count blocks no longer held. Everything else is counted per thread and
// stays exact.
//
// The auditor's own bookkeeping allocates too; counting is paused while it
// runs, so no component is charged for it.
//
// A job that also starts with K4BENCH_ALLOC_OWNERS=1 has the counter record
// who allocated each heap block. While an audited call runs, its component is
// its thread's owner, so whatever the call allocates, at any depth, is charged
// to the innermost running component until it is freed, wherever and whenever
// that happens; a service initialized from within an algorithm's call is a
// call of its own and keeps its own memory. What no component allocated (the
// Python configuration, ROOT's start-up, the framework between calls) belongs
// to no component. The heap file says what each component held before the
// first event (what initialize and start left), at the start and end of each
// event and at most during it, and still held at the auditor's finalize,
// beside malloc's own count of the heap in use and the anonymous RSS.
// What it does not say:
//
//   - who uses memory: a block is charged to the component that created it,
//     so the event data the input reader creates stays the reader's until the
//     output writer frees it;
//   - memory outside malloc's heap: what a library maps itself (pool
//     allocators, CPython's arenas, large buffers) and file-backed memory;
//   - the timed run's values: recording owners makes every allocation several
//     times dearer, so the file comes from a run of its own;
//   - a service first used during an event is initialized then, and what it
//     keeps is held from that event on rather than before the first one.
//
// In a job running calls concurrently, attribution still follows each
// thread's running component, but a snapshot taken while other threads run
// catches them mid-call, and a job running several events at once has no
// per-event peaks: the owners' peaks are the process's.
//
// Files written at finalize:
//
//   ComponentsOutput  per-component self costs per event and per lifecycle
//                     phase; format versioned as COMPONENT_SCHEMA_VERSION in
//                     k4bench/plugin/schema.py.
//   EventsOutput      per-event wall time and memory in the format of the DDG4
//                     k4BenchTimingAction (EVENT_SCHEMA_VERSION), so a k4run
//                     job feeds the same event-level views and regression
//                     metrics as a ddsim job.
//   HeapOutput        only with K4BENCH_ALLOC_OWNERS=1: the heap each
//                     component held (HEAP_SCHEMA_VERSION).
//
// What each side of a call reads and what a call costs are in
// k4BenchMeasurement.h, what the auditor finds out about a component's classes
// in k4BenchComponentInfo.h.
//
// Enabled by appending plugin/auditor/k4BenchAuditorOptions.py to the job's options.

#include "k4BenchComponentInfo.h"
#include "k4BenchMeasurement.h"
#include "k4BenchProcStats.h"

#include <Gaudi/Algorithm.h>
#include <Gaudi/Auditor.h>
#include <Gaudi/Sequence.h>
#include <GaudiKernel/DataObjID.h>
#include <GaudiKernel/IAlgManager.h>
#include <GaudiKernel/IDataHandleHolder.h>
#include <GaudiKernel/IService.h>
#include <GaudiKernel/ISvcLocator.h>
#include <GaudiKernel/System.h>

#include <nlohmann/json.hpp>

#include <malloc.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
  // Versions of the JSON formats written below, not of the software. Must
  // match COMPONENT_SCHEMA_VERSION, EVENT_SCHEMA_VERSION and
  // HEAP_SCHEMA_VERSION in k4bench/plugin/schema.py.
  constexpr int kComponentSchemaVersion = 1;
  constexpr int kEventSchemaVersion = 1;
  constexpr int kHeapSchemaVersion = 1;

  // The heap file's name for owner 0, the memory no component allocated.
  constexpr const char *kNoComponent = "(no component)";

  // Measurement pairs timed at initialize to report the auditor's own cost.
  constexpr int kCalibrationPairs = 1000;

  // Phase name of an algorithm execution, Gaudi::IAuditor::Execute lowercased.
  constexpr const char *kExecutePhase = "execute";

  using Json = nlohmann::ordered_json;

  // From k4BenchMeasurement.h and k4BenchComponentInfo.h.
  using k4bench::accumulate, k4bench::alloc_counters, k4bench::close_alloc_window, k4bench::cost_between,
      k4bench::open_alloc_window, k4bench::perf_group, k4bench::sample;
  using k4bench::AllocWindow, k4bench::Clock, k4bench::Cost, k4bench::Owners, k4bench::PausedAllocCounting,
      k4bench::Sample;
  using k4bench::implementation, k4bench::library_of, k4bench::library_of_marlin_processor;

  // The heap each owner held at one moment, owner 0 (no component) first,
  // and the process's heap as malloc and the kernel saw it then.
  struct HeapSnapshot
  {
    std::vector<std::int64_t> held;
    std::vector<std::int64_t> peak; // the most held since peaks were last reset
    std::size_t malloc_in_use_bytes{0};
    long rss_anon_kb{-1};
  };

  // Values rounded to what the measurement resolves keep the files small.
  // Dividing by the scale yields the double nearest the decimal, which prints
  // as that decimal.
  double rounded(double value, double scale) { return std::round(value * scale) / scale; }
  Json seconds(double value) { return rounded(value, 1e6); }
  Json megabytes(double value) { return rounded(value, 1e3); }

  std::string lowercase(std::string text)
  {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
  }

  std::string from_env(const char *variable, const char *fallback)
  {
    const char *value = std::getenv(variable);
    return value != nullptr && *value != '\0' ? value : fallback;
  }

  // What a component is, as far as the framework can tell.
  struct Component
  {
    explicit Component(std::string component_name) : name(std::move(component_name)) {}

    std::string name;
    std::string category{"other"}; // algorithm | service | other
    std::string type;
    std::string impl;
    std::string library;               // defines type
    std::string impl_library;          // defines impl; empty when it cannot be found
    std::optional<std::string> parent; // enclosing sequencer
    // Event data keys an algorithm declared through its data handles; unset
    // for a component that holds none.
    std::optional<std::vector<std::string>> inputs;
    std::optional<std::vector<std::string>> outputs;
    bool shared{false};                // member of several sequencers
    long execute_calls{0};
    bool described{false};
  };

  // A call in flight on one thread, collecting the inclusive cost of the calls
  // nested inside it.
  struct Frame
  {
    std::size_t component;
    std::string phase;
    Sample begin;
    bool top_level_execute;
    AllocWindow outer_alloc_window; // the enclosing call's, while this one runs
    Cost nested{};
    std::uint32_t outer_owner{0}; // the thread's heap owner before this call
  };

  // One event's top-level execution and the self costs of its components.
  struct EventRecord
  {
    EventContext::ContextEvt_t number;
    Sample begin;
    std::optional<Sample> end;
    k4bench::RssValues rss_begin;
    k4bench::RssValues rss_end;
    std::vector<std::optional<Cost>> costs;
    // When owners are recorded: what each held when the event opened, and
    // the snapshot at its end.
    std::vector<std::int64_t> start_held;
    std::optional<HeapSnapshot> heap;
  };

  // Span of one phase across all components, in ns since the Unix epoch.
  struct Span
  {
    long long begin_epoch_ns{-1};
    long long end_epoch_ns{-1};
  };

  // A column of the components file: its name and how it prints a cost.
  struct Metric
  {
    const char *name;
    Json (*value)(const Cost &);
  };

  constexpr std::array kThreadMetrics{
      Metric{"wall_s", [](const Cost &c) { return seconds(c.wall_s); }},
      Metric{"cpu_s", [](const Cost &c) { return seconds(c.cpu_s); }},
      Metric{"peak_rss_increase_mb", [](const Cost &c) { return megabytes(c.peak_rss_increase_mb); }},
      Metric{"minor_page_faults", [](const Cost &c) { return Json(c.minor_page_faults); }},
      Metric{"major_page_faults", [](const Cost &c) { return Json(c.major_page_faults); }},
      Metric{"voluntary_context_switches", [](const Cost &c) { return Json(c.voluntary_context_switches); }},
      Metric{"involuntary_context_switches", [](const Cost &c) { return Json(c.involuntary_context_switches); }},
  };

  constexpr std::array kHardwareMetrics{
      Metric{"instructions", [](const Cost &c) { return c.hardware_counted ? Json(c.instructions) : Json(nullptr); }},
      Metric{"cycles", [](const Cost &c) { return c.hardware_counted ? Json(c.cycles) : Json(nullptr); }},
  };

  constexpr std::array kAllocationMetrics{
      Metric{"allocations", [](const Cost &c) { return Json(c.allocations); }},
      Metric{"allocated_bytes", [](const Cost &c) { return Json(c.allocated_bytes); }},
      Metric{"net_allocated_bytes", [](const Cost &c) { return Json(c.net_allocated_bytes); }},
      Metric{"peak_heap_bytes", [](const Cost &c) { return Json(c.peak_heap_bytes); }},
      Metric{"largest_allocation_bytes", [](const Cost &c) { return Json(c.largest_allocation_bytes); }},
  };

  Json cost_column(const std::vector<std::optional<Cost>> &costs, std::size_t n, const Metric &metric)
  {
    Json column = Json::array();
    for (std::size_t i = 0; i < n; ++i)
    {
      column.push_back(i < costs.size() && costs[i] ? metric.value(*costs[i]) : Json(nullptr));
    }
    return column;
  }
} // namespace

class k4BenchAuditor final : public Gaudi::Auditor
{
public:
  using Gaudi::Auditor::Auditor;

  StatusCode initialize() override
  {
    const PausedAllocCounting paused;
    m_processStartEpochNs = k4bench::read_process_start_epoch_ns();
    m_steadyOrigin = Clock::now();
    m_epochOrigin = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    m_metrics.assign(kThreadMetrics.begin(), kThreadMetrics.end());
    // A missing metric is reported as a warning, which jobs that quieten their
    // output to warnings (as CLDReconstruction.py does) still print.
    if (perf_group().read())
    {
      m_metrics.insert(m_metrics.end(), kHardwareMetrics.begin(), kHardwareMetrics.end());
      info() << "Counting instructions and cycles" << endmsg;
    }
    else
    {
      m_hardwareCountersError = perf_group().failure();
      warning() << "Hardware counters unavailable (" << *m_hardwareCountersError
                << "): no instruction or cycle counts" << endmsg;
    }
    if (alloc_counters() != nullptr)
    {
      m_metrics.insert(m_metrics.end(), kAllocationMetrics.begin(), kAllocationMetrics.end());
      info() << "Counting heap allocations" << endmsg;
    }
    else
    {
      warning() << "libk4BenchAllocCounter.so is not preloaded: no allocation counts" << endmsg;
    }
    m_owners = Owners::find();
    if (m_owners)
    {
      m_ownerCapacity = m_owners.describe().owners;
      // The peaks before the first event span initialize and start from here.
      m_owners.reset_peaks();
      info() << "Charging heap blocks to the components allocating them" << endmsg;
    }
    calibrate();
    return StatusCode::SUCCESS;
  }

  StatusCode finalize() override
  {
    const PausedAllocCounting paused;
    std::lock_guard lock(m_mutex);
    // Read before the outputs are built, so the job's peak and what it left
    // on the heap exclude the auditor's own serialisation.
    const long peak_vmem_kb = k4bench::read_vmpeak_kb();
    std::optional<HeapSnapshot> at_end;
    if (m_owners)
    {
      at_end = heapSnapshot(k4bench::read_rss_kb().anon);
    }
    describeComponents();
    writeComponents();
    writeEvents(peak_vmem_kb);
    if (at_end)
    {
      writeHeap(*at_end);
    }
    k4bench::write_loaded_paths();
    return StatusCode::SUCCESS;
  }

  void before(std::string const &event, std::string const &caller, EventContext const &ctx) override
  {
    const PausedAllocCounting paused;
    const bool execute = event == Gaudi::IAuditor::Execute;
    std::optional<k4bench::RssValues> event_rss;
    if (execute)
    {
      // Every algorithm is initialized by the first event; describe them now,
      // while they are certain to exist, and before the first measurement so
      // the description is charged to no component.
      std::call_once(m_describedOnce, [this] {
        std::lock_guard lock(m_mutex);
        describeComponents();
      });
      // An event's starting memory is read before its clock starts, as in the
      // DDG4 plugin, so the read is not part of the event's time. It is read
      // only for the call opening the event: for a later top-level call the
      // read would fall inside the event and be charged to no component.
      if (startsEvent(ctx.evt()))
      {
        event_rss = k4bench::read_rss_kb();
        // What initialize and start left on the heap, also read before the
        // first event's clock starts.
        if (m_owners)
        {
          std::lock_guard lock(m_mutex);
          if (!m_beforeFirstEvent)
          {
            m_beforeFirstEvent = heapSnapshot(event_rss->anon);
          }
        }
      }
    }
    const Sample begin = sample();
    const AllocWindow outer_alloc_window = open_alloc_window(begin);
    std::lock_guard lock(m_mutex);
    const std::size_t index = componentIndex(caller);
    auto &stack = m_stacks[std::this_thread::get_id()];
    const bool top_level = execute && stack.empty();

    if (execute)
    {
      m_threads.insert(std::this_thread::get_id());
    }
    if (top_level)
    {
      openEvent(ctx.evt(), begin, event_rss);
    }
    stack.push_back(Frame{index, lowercase(event), begin, top_level, outer_alloc_window});
    // Until after(), what this thread allocates is the component's.
    if (m_owners)
    {
      if (index + 1 >= m_ownerCapacity && !std::exchange(m_ownersExhausted, true))
      {
        warning() << "More components than the allocation counter's " << m_ownerCapacity - 1
                  << " owners: what " << caller << " and later ones allocate counts as no component's" << endmsg;
      }
      stack.back().outer_owner = m_owners.exchange(static_cast<std::uint32_t>(index + 1));
    }
  }

  void after(std::string const &event, std::string const &caller, EventContext const &ctx,
             StatusCode const & /* sc */) override
  {
    const PausedAllocCounting paused;
    const Sample end = sample();
    std::lock_guard lock(m_mutex);
    auto &stack = m_stacks[std::this_thread::get_id()];
    const std::size_t index = componentIndex(caller);
    // Unwind to this component's frame; a frame with no before() (the auditor
    // started mid-call) has nothing to pair with.
    const auto frame = std::find_if(stack.rbegin(), stack.rend(),
                                    [index](const Frame &f) { return f.component == index; });
    if (frame == stack.rend())
    {
      return;
    }
    const Frame done = *frame;
    stack.erase(std::next(frame).base(), stack.end());
    if (m_owners)
    {
      m_owners.exchange(done.outer_owner);
    }
    close_alloc_window(done.outer_alloc_window, end);

    const bool execute = event == Gaudi::IAuditor::Execute;
    const Cost inclusive = cost_between(done.begin, end);
    // A call is subtracted only from an enclosing call recorded in the same
    // table: a non-execute call during an execution (a service initialized on
    // first use, a custom audited section) stays in the executing component's
    // event cost, so an event's costs still add up, and is also listed under
    // its own phase.
    if (!stack.empty() && (stack.back().phase == kExecutePhase) == execute)
    {
      stack.back().nested += inclusive;
    }
    Cost self = inclusive;
    self -= done.nested;

    if (execute)
    {
      recordExecute(ctx.evt(), index, self, done, end);
    }
    else
    {
      // Only calls outside event processing delimit their phase, so a late
      // initialize does not stretch the initialize phase over the event loop.
      const bool during_execute =
          std::any_of(stack.begin(), stack.end(), [](const Frame &f) { return f.phase == kExecutePhase; });
      recordLifecycle(done.phase, index, self, done.begin, end, !during_execute);
    }
  }

private:
  // Output paths default to the environment, as for the DDG4 plugins, so the
  // options file can enable the auditor without a generated Configurable.
  Gaudi::Property<std::string> m_componentsOutput{
      this, "ComponentsOutput", from_env("K4BENCH_COMPONENTS_JSON", "k4bench_components.json"),
      "per-component JSON written at finalize"};
  Gaudi::Property<std::string> m_eventsOutput{
      this, "EventsOutput", from_env("K4BENCH_EVENT_JSON", "k4bench_events.json"),
      "per-event JSON written at finalize"};
  Gaudi::Property<std::string> m_heapOutput{
      this, "HeapOutput", from_env("K4BENCH_HEAP_JSON", "k4bench_heap.json"),
      "per-component heap JSON written at finalize when the job records heap owners"};

  std::mutex m_mutex;
  long long m_processStartEpochNs{-1};
  Clock::time_point m_steadyOrigin;
  long long m_epochOrigin{0};
  double m_overheadNsPerCall{-1.0};
  std::optional<std::string> m_hardwareCountersError; // why there are no instruction or cycle counts
  std::once_flag m_describedOnce;
  std::vector<Metric> m_metrics; // the columns written, allocations when counted
  Owners m_owners;                // empty unless the job records heap owners
  std::size_t m_ownerCapacity{0};
  bool m_ownersExhausted{false};
  std::optional<HeapSnapshot> m_beforeFirstEvent;
  bool m_eventsOverlap{false}; // several events were in flight at once

  std::vector<Component> m_components;
  std::unordered_map<std::string, std::size_t> m_index;
  std::unordered_map<std::thread::id, std::vector<Frame>> m_stacks;
  std::set<std::thread::id> m_threads;

  std::vector<EventRecord> m_events;
  std::unordered_map<EventContext::ContextEvt_t, std::size_t> m_eventRows;
  std::map<std::string, std::vector<std::optional<Cost>>> m_lifecycle;
  std::map<std::string, Span> m_phases;

  // Time of one before/after measurement pair, which lands in the self cost of
  // the enclosing call once per nested call.
  void calibrate()
  {
    long sink = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < kCalibrationPairs; ++i)
    {
      sink += sample().usage.ru_maxrss;
      sink += sample().usage.ru_maxrss;
    }
    const auto elapsed = Clock::now() - begin;
    m_overheadNsPerCall = sink < 0 ? -1.0 : std::chrono::duration<double, std::nano>(elapsed).count() / kCalibrationPairs;
  }

  long long epochNs(const Sample &s) const
  {
    return m_epochOrigin + std::chrono::duration_cast<std::chrono::nanoseconds>(s.wall - m_steadyOrigin).count();
  }

  // Whether any thread is inside an algorithm's execution (callers hold m_mutex).
  bool executing() const
  {
    return std::any_of(m_stacks.begin(), m_stacks.end(), [](const auto &thread) {
      return std::any_of(thread.second.begin(), thread.second.end(),
                         [](const Frame &f) { return f.phase == kExecutePhase; });
    });
  }

  // Whether this thread's next call is top-level and opens event *number*.
  bool startsEvent(EventContext::ContextEvt_t number)
  {
    std::lock_guard lock(m_mutex);
    return m_stacks[std::this_thread::get_id()].empty() && !m_eventRows.contains(number);
  }

  // -- bookkeeping (callers hold m_mutex) ------------------------------------

  std::size_t componentIndex(const std::string &name)
  {
    const auto [it, inserted] = m_index.try_emplace(name, m_components.size());
    if (inserted)
    {
      m_components.emplace_back(name);
    }
    return it->second;
  }

  // An event opens at its earliest top-level call; later top-level calls of the
  // same event (several top algorithms) extend it. Calls sample before taking
  // the lock, so concurrent ones may arrive out of order; a call that found the
  // event already open sampled after its opener and read no memory.
  void openEvent(EventContext::ContextEvt_t number, const Sample &begin, const std::optional<k4bench::RssValues> &rss)
  {
    const auto [row, inserted] = m_eventRows.try_emplace(number, m_events.size());
    // Events overlap when one opens while another still executes, or an
    // earlier one's next top-level call comes after a later one opened.
    if (inserted ? executing() : number != m_events.back().number)
    {
      m_eventsOverlap = true;
    }
    if (inserted)
    {
      m_events.push_back(EventRecord{number, begin, std::nullopt, rss.value_or(k4bench::RssValues{}), {}, {}, {}, {}});
      // The event's heap peaks start here, from what each component holds now.
      if (m_owners)
      {
        m_owners.reset_peaks();
        std::vector<std::int64_t> &held = m_events.back().start_held;
        held.resize(m_components.size() + 1);
        m_owners.read(held.data(), nullptr, held.size());
      }
      return;
    }
    EventRecord &record = m_events[row->second];
    if (rss && begin.wall < record.begin.wall)
    {
      record.begin = begin;
      record.rss_begin = *rss;
    }
  }

  void recordExecute(EventContext::ContextEvt_t number, std::size_t index, const Cost &self, const Frame &frame,
                     const Sample &end)
  {
    const auto row = m_eventRows.find(number);
    if (row == m_eventRows.end())
    {
      return;
    }
    EventRecord &record = m_events[row->second];
    if (record.costs.size() <= index)
    {
      record.costs.resize(index + 1);
    }
    accumulate(record.costs[index], self);
    ++m_components[index].execute_calls;

    if (frame.top_level_execute && (!record.end || record.end->wall < end.wall))
    {
      record.end = end;
      record.rss_end = k4bench::read_rss_kb();
      if (m_owners)
      {
        record.heap = heapSnapshot(record.rss_end.anon);
      }
    }
    extendSpan("execute", frame.begin, end);
  }

  void recordLifecycle(const std::string &phase, std::size_t index, const Cost &self, const Sample &begin,
                       const Sample &end, bool delimits_phase)
  {
    auto &costs = m_lifecycle[phase];
    if (costs.size() <= index)
    {
      costs.resize(index + 1);
    }
    accumulate(costs[index], self);
    if (delimits_phase)
    {
      extendSpan(phase, begin, end);
    }
  }

  // What every component holds now. mallinfo2 walks malloc's free lists,
  // about 0.1 µs per free chunk: milliseconds at most, once per event.
  HeapSnapshot heapSnapshot(long rss_anon_kb) const
  {
    HeapSnapshot snapshot;
    snapshot.held.resize(m_components.size() + 1);
    snapshot.peak.resize(snapshot.held.size());
    m_owners.read(snapshot.held.data(), snapshot.peak.data(), snapshot.held.size());
    // uordblks leaves out the blocks glibc mapped on their own, hblkhd.
    const struct mallinfo2 info = ::mallinfo2();
    snapshot.malloc_in_use_bytes = info.uordblks + info.hblkhd;
    snapshot.rss_anon_kb = rss_anon_kb;
    return snapshot;
  }

  void extendSpan(const std::string &phase, const Sample &begin, const Sample &end)
  {
    Span &span = m_phases[phase];
    const long long b = epochNs(begin);
    if (span.begin_epoch_ns < 0 || b < span.begin_epoch_ns)
    {
      span.begin_epoch_ns = b;
    }
    span.end_epoch_ns = std::max(span.end_epoch_ns, epochNs(end));
  }

  // An algorithm listed by several sequencers runs once per event, under
  // whichever reaches it first, so no single one encloses it: it gets no
  // parent, and no sequencer's inclusive cost claims it.
  static void setParent(Component &component, const std::string &sequence)
  {
    if (component.shared)
    {
      return;
    }
    if (component.parent && *component.parent != sequence)
    {
      component.parent.reset();
      component.shared = true;
      return;
    }
    component.parent = sequence;
  }

  // Shared object defining the class *component* delegates to: its own type's,
  // unless it wraps a Marlin processor (implementation()).
  static std::string implementation_library(const Component &component)
  {
    return component.impl == component.type ? component.library
                                            : library_of_marlin_processor(component.impl);
  }

  // The keys of *ids*, sorted, so a file does not depend on hashing order.
  static std::vector<std::string> data_keys(const DataObjIDColl &ids)
  {
    std::vector<std::string> keys;
    keys.reserve(ids.size());
    for (const DataObjID &id : ids)
    {
      keys.push_back(id.key());
    }
    std::sort(keys.begin(), keys.end());
    return keys;
  }

  // Fill in category, type, implementation, libraries, data and parent for every
  // component the managers know. Algorithms are walked in full, so a
  // sequencer's children are registered even before their first call.
  void describeComponents()
  {
    SmartIF<IAlgManager> algorithms(serviceLocator());
    if (algorithms)
    {
      for (IAlgorithm *ialg : algorithms->getAlgorithms())
      {
        if (auto *sequence = dynamic_cast<Gaudi::Sequence *>(ialg))
        {
          if (const auto *children = sequence->subAlgorithms())
          {
            for (const Gaudi::Algorithm *child : *children)
            {
              setParent(m_components[componentIndex(child->name())], sequence->name());
            }
          }
        }
        Component &component = m_components[componentIndex(ialg->name())];
        if (component.described)
        {
          continue;
        }
        component.category = "algorithm";
        component.type = ialg->type();
        component.impl = implementation(ialg, component.type);
        component.library = library_of(dynamic_cast<const void *>(ialg));
        component.impl_library = implementation_library(component);
        // Handles are declared by initialize at the latest (the k4FWCore
        // Reader declares one per collection of its input file then), and
        // components are described once the first event starts.
        if (const auto *holder = dynamic_cast<const IDataHandleHolder *>(ialg))
        {
          component.inputs = data_keys(holder->inputDataObjs());
          component.outputs = data_keys(holder->outputDataObjs());
        }
        component.described = true;
      }
    }

    for (Component &component : m_components)
    {
      if (component.described || !serviceLocator()->existsService(component.name))
      {
        continue;
      }
      SmartIF<IService> &service = serviceLocator()->service(component.name, /*createIf=*/false);
      if (!service)
      {
        continue;
      }
      component.category = "service";
      component.type = System::typeinfoName(typeid(*service.get()));
      component.impl = implementation(service.get(), component.type);
      component.library = library_of(dynamic_cast<const void *>(service.get()));
      component.impl_library = implementation_library(component);
      component.described = true;
    }
  }

  // -- output ----------------------------------------------------------------

  Json costTable(const std::vector<std::optional<Cost>> &costs, std::size_t n) const
  {
    Json table = Json::object();
    for (const Metric &metric : m_metrics)
    {
      table[metric.name] = cost_column(costs, n, metric);
    }
    return table;
  }

  std::vector<const EventRecord *> completedEvents() const
  {
    std::vector<const EventRecord *> done;
    for (const auto &record : m_events)
    {
      if (record.end)
      {
        done.push_back(&record);
      }
    }
    return done;
  }

  void writeComponents() const
  {
    const std::size_t n = m_components.size();
    Json out = Json::object();
    out["schema_version"] = kComponentSchemaVersion;
    out["producer"] = "k4BenchAuditor";
    out["measurement_overhead_ns"] = std::round(m_overheadNsPerCall);
    out["threads"] = m_threads.size();
    out["process_start_epoch_ns"] = m_processStartEpochNs;
    out["hardware_counters_error"] = m_hardwareCountersError ? Json(*m_hardwareCountersError) : Json(nullptr);

    Json phases = Json::object();
    for (const auto &[phase, span] : m_phases)
    {
      phases[phase] = {{"begin_epoch_ns", span.begin_epoch_ns}, {"end_epoch_ns", span.end_epoch_ns}};
    }
    out["phases"] = phases;

    Json components = Json::array();
    for (const Component &c : m_components)
    {
      components.push_back({{"name", c.name},
                            {"category", c.category},
                            {"type", c.type},
                            {"impl", c.impl},
                            {"library", c.library},
                            {"impl_library", c.impl_library},
                            {"inputs", c.inputs ? Json(*c.inputs) : Json(nullptr)},
                            {"outputs", c.outputs ? Json(*c.outputs) : Json(nullptr)},
                            {"parent", c.parent ? Json(m_index.at(*c.parent)) : Json(nullptr)},
                            {"execute_calls", c.execute_calls}});
    }
    out["components"] = components;

    Json lifecycle = Json::object();
    for (const auto &[phase, costs] : m_lifecycle)
    {
      lifecycle[phase] = costTable(costs, n);
    }
    out["lifecycle"] = lifecycle;

    Json numbers = Json::array();
    Json execute = Json::object();
    for (const Metric &metric : m_metrics)
    {
      execute[metric.name] = Json::array();
    }
    for (const EventRecord *record : completedEvents())
    {
      numbers.push_back(record->number);
      for (const Metric &metric : m_metrics)
      {
        execute[metric.name].push_back(cost_column(record->costs, n, metric));
      }
    }
    out["event_numbers"] = numbers;
    out["execute"] = execute;

    write(m_componentsOutput.value(), out);
  }

  void writeEvents(long peak_vmem_kb) const
  {
    const auto mb = [](long kb) { return kb < 0 ? Json(-1.0) : megabytes(static_cast<double>(kb) / 1024.0); };

    Json numbers = Json::array();
    Json times = Json::array();
    Json rss_begin = Json::array();
    Json rss_end = Json::array();
    Json anon_begin = Json::array();
    Json anon_end = Json::array();
    Json file_end = Json::array();
    for (const EventRecord *record : completedEvents())
    {
      numbers.push_back(record->number);
      times.push_back(seconds(std::chrono::duration<double>(record->end->wall - record->begin.wall).count()));
      rss_begin.push_back(mb(record->rss_begin.total));
      rss_end.push_back(mb(record->rss_end.total));
      anon_begin.push_back(mb(record->rss_begin.anon));
      anon_end.push_back(mb(record->rss_end.anon));
      file_end.push_back(mb(record->rss_end.file));
    }

    Json out = Json::object();
    out["schema_version"] = kEventSchemaVersion;
    out["peak_vmem_mb"] = mb(peak_vmem_kb);
    out["event_numbers"] = numbers;
    out["event_times_s"] = times;
    out["event_rss_begin_mb"] = rss_begin;
    out["event_rss_end_mb"] = rss_end;
    out["event_rss_anon_begin_mb"] = anon_begin;
    out["event_rss_anon_end_mb"] = anon_end;
    out["event_rss_file_end_mb"] = file_end;

    write(m_eventsOutput.value(), out);
  }

  void writeHeap(const HeapSnapshot &at_end) const
  {
    const k4bench::OwnerTable table = m_owners.describe();
    if (table.failed)
    {
      error() << "The allocation counter could not map a larger table of owned blocks: no heap file" << endmsg;
      return;
    }
    // The owners' peaks are the process's, reset as each event opens: with
    // several events in flight, an opening one reset the peaks of those still
    // running, and no event has a peak of its own.
    if (m_eventsOverlap)
    {
      warning() << "Events ran concurrently: no per-event heap peaks" << endmsg;
    }
    const std::size_t n = m_components.size() + 1;
    // A component first seen after a snapshot held nothing then.
    const auto per_owner = [n](const std::vector<std::int64_t> &values)
    {
      Json column = Json::array();
      for (std::size_t owner = 0; owner < n; ++owner)
      {
        column.push_back(owner < values.size() ? values[owner] : 0);
      }
      return column;
    };
    const auto moment = [&per_owner](const HeapSnapshot &snapshot, bool peaks)
    {
      Json out = Json::object();
      out["held_bytes"] = per_owner(snapshot.held);
      if (peaks)
      {
        out["peak_held_bytes"] = per_owner(snapshot.peak);
      }
      out["malloc_in_use_bytes"] = snapshot.malloc_in_use_bytes;
      out["rss_anon_mb"] = snapshot.rss_anon_kb < 0 ? Json(-1.0)
                                                    : megabytes(static_cast<double>(snapshot.rss_anon_kb) / 1024.0);
      return out;
    };

    Json names = Json::array();
    names.push_back(kNoComponent);
    for (const Component &c : m_components)
    {
      names.push_back(c.name);
    }
    // A row per event: what each component held as it opened, then its
    // moment at the end.
    Json numbers = Json::array();
    Json execute = Json::object();
    for (const char *key : {"start_held_bytes", "held_bytes", "peak_held_bytes", "malloc_in_use_bytes", "rss_anon_mb"})
    {
      execute[key] = Json::array();
    }
    for (const EventRecord *record : completedEvents())
    {
      numbers.push_back(record->number);
      execute["start_held_bytes"].push_back(per_owner(record->start_held));
      Json row = moment(*record->heap, true);
      if (m_eventsOverlap)
      {
        row["peak_held_bytes"] = Json(n, Json(nullptr));
      }
      for (const auto &item : row.items())
      {
        execute[item.key()].push_back(item.value());
      }
    }

    Json out = Json::object();
    out["schema_version"] = kHeapSchemaVersion;
    out["producer"] = "k4BenchAuditor";
    out["components"] = names;
    out["before_first_event"] = m_beforeFirstEvent ? moment(*m_beforeFirstEvent, true) : Json(nullptr);
    out["event_numbers"] = numbers;
    out["execute"] = execute;
    out["at_end"] = moment(at_end, false);
    out["table"] = {{"peak_entries", table.peak_entries}, {"mapped_bytes", table.mapped_bytes}};

    write(m_heapOutput.value(), out);
  }

  void write(const std::string &path, const Json &content) const
  {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out)
    {
      error() << "Could not open " << path << " for writing" << endmsg;
      return;
    }
    out << content.dump() << '\n';
    info() << "Wrote " << path << endmsg;
  }
};

DECLARE_COMPONENT(k4BenchAuditor)
