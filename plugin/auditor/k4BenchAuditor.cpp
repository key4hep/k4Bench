// k4BenchAuditor.cpp
//
// Gaudi auditor that measures every audited call of every component — each
// algorithm execution, and each initialize, start, stop and finalize of
// algorithms and services: its wall time, the CPU time of its thread, and how
// much it raised the process's peak RSS. It knows nothing about the job it
// audits; components are discovered from the calls and described from the
// Gaudi managers.
//
// Every cost is a *self* cost: calls nest (a sequencer runs its children, the
// event loop manager initializes the algorithms), and a call is charged only
// what was not spent in the audited calls nested inside it. Self costs
// therefore add up: summed over the components of an event they give the cost
// of its audited calls, without double counting at any depth. A non-execute
// call made during an execution (a service initialized on first use, a custom
// audited section) stays in the executing component's event cost and is also
// listed under its own phase, without widening that phase's span.
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
// increase name the component that raised the high-water mark.
//
// Two files are written at finalize:
//
//   ComponentsOutput  per-component self costs per event and per lifecycle
//                     phase; format versioned as COMPONENT_SCHEMA_VERSION in
//                     k4bench/plugin/schema.py.
//   EventsOutput      per-event wall time and memory in the format of the DDG4
//                     k4BenchTimingAction (EVENT_SCHEMA_VERSION), so a k4run
//                     job feeds the same event-level views and regression
//                     metrics as a ddsim job.
//
// Enabled by appending plugin/auditor/k4BenchAuditorOptions.py to the job's options.

#include "k4BenchProcStats.h"

#include <Gaudi/Algorithm.h>
#include <Gaudi/Auditor.h>
#include <Gaudi/Sequence.h>
#include <GaudiKernel/IAlgManager.h>
#include <GaudiKernel/IProperty.h>
#include <GaudiKernel/IService.h>
#include <GaudiKernel/ISvcLocator.h>
#include <GaudiKernel/System.h>

#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <sys/resource.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
  // Versions of the two JSON formats written below, not of the software. Must
  // match COMPONENT_SCHEMA_VERSION and EVENT_SCHEMA_VERSION in
  // k4bench/plugin/schema.py.
  constexpr int kComponentSchemaVersion = 1;
  constexpr int kEventSchemaVersion = 1;

  // Measurement pairs timed at initialize to report the auditor's own cost.
  constexpr int kCalibrationPairs = 1000;

  // Phase name of an algorithm execution, Gaudi::IAuditor::Execute lowercased.
  constexpr const char *kExecutePhase = "execute";

  using Clock = std::chrono::steady_clock;
  using Json = nlohmann::ordered_json;

  // What one side of a call reads. A single getrusage(RUSAGE_THREAD) gives both
  // the thread's CPU time (user + system, microsecond resolution) and the
  // process's peak RSS, at a fraction of the cost of reading /proc.
  struct Sample
  {
    Clock::time_point wall;
    std::int64_t cpu_us{0};
    long peak_rss_kb{0};
  };

  Sample sample()
  {
    Sample s;
    s.wall = Clock::now();
    rusage usage{};
    if (::getrusage(RUSAGE_THREAD, &usage) == 0)
    {
      s.cpu_us = (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1'000'000LL +
                 usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
      s.peak_rss_kb = usage.ru_maxrss;
    }
    return s;
  }

  // What a call cost, or several calls of one component summed.
  struct Cost
  {
    double wall_s{0.0};
    double cpu_s{0.0};
    double peak_rss_increase_mb{0.0};

    Cost &operator+=(const Cost &other)
    {
      wall_s += other.wall_s;
      cpu_s += other.cpu_s;
      peak_rss_increase_mb += other.peak_rss_increase_mb;
      return *this;
    }

    Cost &operator-=(const Cost &other)
    {
      wall_s -= other.wall_s;
      cpu_s -= other.cpu_s;
      peak_rss_increase_mb -= other.peak_rss_increase_mb;
      return *this;
    }
  };

  Cost cost_between(const Sample &begin, const Sample &end)
  {
    Cost cost;
    cost.wall_s = std::chrono::duration<double>(end.wall - begin.wall).count();
    cost.cpu_s = static_cast<double>(end.cpu_us - begin.cpu_us) * 1e-6;
    cost.peak_rss_increase_mb = static_cast<double>(end.peak_rss_kb - begin.peak_rss_kb) / 1024.0;
    return cost;
  }

  // A component called more than once in one event or phase is charged the sum.
  void accumulate(std::optional<Cost> &cell, const Cost &cost)
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

  std::string unquoted(std::string text)
  {
    if (text.size() >= 2 && (text.front() == '\'' || text.front() == '"') && text.back() == text.front())
    {
      return text.substr(1, text.size() - 2);
    }
    return text;
  }

  // Shared object the dynamic type of *object* is defined in, resolved through
  // symlinks so a path into an LCG view names the package install it points to.
  std::string library_of(const void *object)
  {
    // Under the Itanium C++ ABI a polymorphic object starts with its vtable
    // pointer, and the vtable lives in the library that defines the class.
    const void *vtable = *static_cast<const void *const *>(object);
    Dl_info info{};
    if (::dladdr(vtable, &info) == 0 || info.dli_fname == nullptr)
    {
      return {};
    }
    std::error_code error;
    const auto path = std::filesystem::canonical(info.dli_fname, error);
    return error ? std::string(info.dli_fname) : path.string();
  }

  // The class doing the work: the component's own type, unless a wrapper names
  // the class it delegates to in a ProcessorType property (k4MarlinWrapper).
  std::string implementation(IInterface *component, const std::string &type)
  {
    auto *properties = dynamic_cast<IProperty *>(component);
    std::string wrapped;
    if (properties && properties->hasProperty("ProcessorType") &&
        properties->getProperty("ProcessorType", wrapped).isSuccess() && !wrapped.empty())
    {
      return unquoted(wrapped);
    }
    return type;
  }

  // What a component is, as far as the framework can tell.
  struct Component
  {
    explicit Component(std::string component_name) : name(std::move(component_name)) {}

    std::string name;
    std::string category{"other"}; // algorithm | service | other
    std::string type;
    std::string impl;
    std::string library;
    std::optional<std::string> parent; // enclosing sequencer
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
    Cost nested{};
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
  };

  // Span of one phase across all components, in ns since the Unix epoch.
  struct Span
  {
    long long begin_epoch_ns{-1};
    long long end_epoch_ns{-1};
  };

  Json cost_column(const std::vector<std::optional<Cost>> &costs, std::size_t n, double Cost::*field,
                   Json (*format)(double))
  {
    Json column = Json::array();
    for (std::size_t i = 0; i < n; ++i)
    {
      column.push_back(i < costs.size() && costs[i] ? format((*costs[i]).*field) : Json(nullptr));
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
    m_processStartEpochNs = k4bench::read_process_start_epoch_ns();
    m_steadyOrigin = Clock::now();
    m_epochOrigin = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    calibrate();
    return StatusCode::SUCCESS;
  }

  StatusCode finalize() override
  {
    std::lock_guard lock(m_mutex);
    // Read before the outputs are built, so the job's peak excludes the
    // auditor's own serialisation.
    const long peak_vmem_kb = k4bench::read_vmpeak_kb();
    describeComponents();
    writeComponents();
    writeEvents(peak_vmem_kb);
    return StatusCode::SUCCESS;
  }

  void before(std::string const &event, std::string const &caller, EventContext const &ctx) override
  {
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
      }
    }
    const Sample begin = sample();
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
    stack.push_back(Frame{index, lowercase(event), begin, top_level});
  }

  void after(std::string const &event, std::string const &caller, EventContext const &ctx,
             StatusCode const & /* sc */) override
  {
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

  std::mutex m_mutex;
  long long m_processStartEpochNs{-1};
  Clock::time_point m_steadyOrigin;
  long long m_epochOrigin{0};
  double m_overheadNsPerCall{-1.0};
  std::once_flag m_describedOnce;

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
      sink += sample().peak_rss_kb;
      sink += sample().peak_rss_kb;
    }
    const auto elapsed = Clock::now() - begin;
    m_overheadNsPerCall = sink < 0 ? -1.0 : std::chrono::duration<double, std::nano>(elapsed).count() / kCalibrationPairs;
  }

  long long epochNs(const Sample &s) const
  {
    return m_epochOrigin + std::chrono::duration_cast<std::chrono::nanoseconds>(s.wall - m_steadyOrigin).count();
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
    if (inserted)
    {
      m_events.push_back(EventRecord{number, begin, std::nullopt, rss.value_or(k4bench::RssValues{}), {}, {}});
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

  // Fill in category, type, implementation, library and parent for every
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
      component.described = true;
    }
  }

  // -- output ----------------------------------------------------------------

  static Json costTable(const std::vector<std::optional<Cost>> &costs, std::size_t n)
  {
    return {{"wall_s", cost_column(costs, n, &Cost::wall_s, seconds)},
            {"cpu_s", cost_column(costs, n, &Cost::cpu_s, seconds)},
            {"peak_rss_increase_mb", cost_column(costs, n, &Cost::peak_rss_increase_mb, megabytes)}};
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
    Json wall = Json::array();
    Json cpu = Json::array();
    Json peak = Json::array();
    for (const EventRecord *record : completedEvents())
    {
      numbers.push_back(record->number);
      wall.push_back(cost_column(record->costs, n, &Cost::wall_s, seconds));
      cpu.push_back(cost_column(record->costs, n, &Cost::cpu_s, seconds));
      peak.push_back(cost_column(record->costs, n, &Cost::peak_rss_increase_mb, megabytes));
    }
    out["event_numbers"] = numbers;
    out["execute"] = {{"wall_s", wall}, {"cpu_s", cpu}, {"peak_rss_increase_mb", peak}};

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
