// DD4benchRegionTimingAction.cpp
//
// DDG4 stepping + tracking + event actions that record per-event, per-region
// wall time spent inside Geant4 stepping. Two complementary views are
// produced:
//
//   - "at_location": time charged to the G4Region the step is currently in.
//   - "by_birth":    time charged to the G4Region in which the track was
//                    created. Charges secondaries back to where they were
//                    born, regardless of where they end up being tracked.
//
// The gap between the two views is informative: it tells you how much of a
// region's cost is intrinsic vs. imported from upstream showers.
//
// Output path is controlled via:
//
//   DD4BENCH_REGION_JSON=/path/to/output.json
//
// If unset, defaults to:
//
//   dd4bench_regions.json
//
// Overhead notes:
// - Uses __rdtscp where available (x86_64 Linux), falling back to
//   std::chrono::steady_clock otherwise. __rdtscp is serializing on its
//   read, which prevents the CPU from reordering the timing fences around
//   the step itself; this matters for short (~hundreds of ns) intervals.
// - TSC frequency is calibrated against steady_clock at first event begin
//   (three rounds, median), so calibration happens once the process is
//   warm but before any timing data is recorded.
// - Per-step overhead is measured at the same time and reported in the
//   JSON. It is NOT subtracted automatically; overhead is uniform per
//   step, so ratios between regions are not biased.
//
// Multithreading:
// - This plugin assumes single-threaded simulation. At first event begin it
//   checks G4Threading::IsMultithreadedApplication() and disables itself
//   (with an ERROR printout) if MT is detected, rather than silently
//   producing garbage from racing accumulators.
//
// Output schema version: 1.

#include <DDG4/Geant4SteppingAction.h>
#include <DDG4/Geant4TrackingAction.h>
#include <DDG4/Geant4EventAction.h>
#include <DDG4/Geant4Context.h>
#include <DDG4/Geant4Kernel.h>
#include <DD4hep/Printout.h>

#include <G4Step.hh>
#include <G4Track.hh>
#include <G4Event.hh>
#include <G4Region.hh>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>
#include <G4Threading.hh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#define DD4BENCH_HAVE_RDTSCP 1
#else
#define DD4BENCH_HAVE_RDTSCP 0
#endif

namespace dd4hep
{
  namespace sim
  {

    // -----------------------------------------------------------------------
    // Low-overhead timer
    //
    // On x86_64 Linux we use __rdtscp (read TSC + serialize). The TSC is
    // calibrated against std::chrono::steady_clock with three short rounds,
    // taking the median to reduce sensitivity to scheduling jitter and
    // transient frequency events. On other platforms we fall back to
    // steady_clock::now() directly.
    // -----------------------------------------------------------------------

    struct TscTimer
    {
      double nsPerTick{1.0};
      bool useRdtscp{false};

      static inline std::uint64_t rdtscpNow()
      {
#if DD4BENCH_HAVE_RDTSCP
        unsigned int aux;
        return __rdtscp(&aux);
#else
        return 0;
#endif
      }

      static inline std::uint64_t steadyNs()
      {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(
                   steady_clock::now().time_since_epoch())
            .count();
      }

      void calibrate()
      {
#if DD4BENCH_HAVE_RDTSCP
        // Three calibration rounds of ~15 ms each, median taken. Short
        // enough not to delay startup noticeably; multiple rounds let us
        // discard a transient frequency excursion.
        constexpr int kRounds = 3;
        constexpr auto kWindow = std::chrono::milliseconds(15);

        std::vector<double> samples;
        samples.reserve(kRounds);

        for (int r = 0; r < kRounds; ++r)
        {
          const std::uint64_t tsc0 = rdtscpNow();
          const std::uint64_t ns0 = steadyNs();

          const auto deadline = std::chrono::steady_clock::now() + kWindow;
          while (std::chrono::steady_clock::now() < deadline)
          {
            // Spin.
          }

          const std::uint64_t tsc1 = rdtscpNow();
          const std::uint64_t ns1 = steadyNs();

          const double dTsc = static_cast<double>(tsc1 - tsc0);
          const double dNs = static_cast<double>(ns1 - ns0);

          if (dTsc > 0.0 && dNs > 0.0)
          {
            samples.push_back(dNs / dTsc);
          }
        }

        if (samples.empty())
        {
          nsPerTick = 1.0;
          useRdtscp = false;
          return;
        }

        std::sort(samples.begin(), samples.end());
        nsPerTick = samples[samples.size() / 2];
        useRdtscp = true;
#else
        nsPerTick = 1.0;
        useRdtscp = false;
#endif
      }

      inline std::uint64_t now() const
      {
#if DD4BENCH_HAVE_RDTSCP
        if (useRdtscp)
        {
          return rdtscpNow();
        }
#endif
        return steadyNs();
      }

      inline double toSeconds(std::uint64_t ticks) const
      {
#if DD4BENCH_HAVE_RDTSCP
        if (useRdtscp)
        {
          return static_cast<double>(ticks) * nsPerTick * 1e-9;
        }
#endif
        return static_cast<double>(ticks) * 1e-9;
      }
    };

    // -----------------------------------------------------------------------
    // Shared state between stepping action, tracking action, and event
    // action.
    //
    // Single-threaded by design (see file header). Held as a function-local
    // static so all three plugins see the same instance regardless of
    // construction order.
    // -----------------------------------------------------------------------

    struct RegionTimingState
    {
      TscTimer timer;
      bool calibrated{false};

      // Set to false at first event begin if multithreaded mode is detected.
      // When false, all hot-path callbacks return immediately.
      bool enabled{true};

      // Track ID -> birth region name. Cleared at event begin.
      std::unordered_map<int, std::string> birthRegion;

      // Per-event tick accumulators.
      std::map<std::string, std::uint64_t> atLocationTicks;
      std::map<std::string, std::uint64_t> byBirthTicks;

      // "Intervals attributed to region" -- i.e. number of timer-to-timer
      // intervals charged to this region. This is the previous step's
      // region at each callback, not the step itself, so it is off by one
      // from "steps in region" at the boundaries. For shower-heavy events
      // the distinction is negligible, but the name reflects what it
      // actually measures.
      std::map<std::string, std::uint64_t> intervalCounts;

      // Diagnostic: how often the stepping action had to fall back to
      // computing the birth region from the track because the tracking
      // action had not stamped it yet. Should be zero in healthy runs.
      std::uint64_t birthFallbackCount{0};

      // Per-event history.
      std::vector<int> eventNumbers;
      std::vector<std::map<std::string, double>> atLocationHistory;
      std::vector<std::map<std::string, double>> byBirthHistory;
      std::vector<std::map<std::string, std::uint64_t>> intervalCountHistory;
      std::vector<double> eventWallSeconds;
      std::vector<double> eventRegionSumSeconds;
      std::vector<double> eventUnaccountedSeconds;
      std::vector<std::uint64_t> eventBirthFallbacks;

      // Measured per-step overhead (seconds), set at first event begin.
      double perStepOverheadSeconds{0.0};

      // Output file path (set by stepping action constructor).
      std::string outputFile{"dd4bench_regions.json"};

      // True once results have been flushed to disk. Prevents double-write
      // when both the run-end hook and the destructor try to finalize.
      bool finalized{false};

      static RegionTimingState &instance()
      {
        static RegionTimingState s;
        return s;
      }

      // Called at the start of the first event. We do calibration and the
      // MT check here, not in constructors, because:
      //   - At construction time the Geant4 kernel may not yet have
      //     decided on MT vs sequential mode.
      //   - Calibrating at first event means the CPU is warm and the
      //     scheduler is settled, which gives more stable numbers than
      //     calibrating at plugin load.
      void initOnFirstEvent()
      {
        if (calibrated)
        {
          return;
        }
        calibrated = true;

        if (G4Threading::IsMultithreadedApplication())
        {
          enabled = false;
          printout(
              ERROR,
              "DD4benchRegionTimingAction",
              "Multithreaded Geant4 detected; disabling per-region timing. "
              "This plugin is single-threaded only. Re-run with "
              "/run/numberOfThreads 1 (or equivalent) to use it.");
          return;
        }

        timer.calibrate();
        measureOverhead();
      }

      void measureOverhead()
      {
        constexpr int kIterations = 200000;
        std::map<std::string, std::uint64_t> dummy;
        const std::string key = "__overhead_probe__";
        dummy[key] = 0;

        const std::uint64_t wallStart = TscTimer::steadyNs();

        std::uint64_t prev = timer.now();
        for (int i = 0; i < kIterations; ++i)
        {
          const std::uint64_t cur = timer.now();
          dummy[key] += (cur - prev);
          prev = cur;
        }

        const std::uint64_t wallEnd = TscTimer::steadyNs();
        const double totalSeconds = (wallEnd - wallStart) * 1e-9;

        perStepOverheadSeconds = totalSeconds / static_cast<double>(kIterations);
      }
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    static inline const G4Region *stepRegion(const G4Step *step)
    {
      if (!step)
      {
        return nullptr;
      }
      const G4StepPoint *pre = step->GetPreStepPoint();
      if (!pre)
      {
        return nullptr;
      }
      const G4VPhysicalVolume *phys = pre->GetPhysicalVolume();
      if (!phys)
      {
        return nullptr;
      }
      const G4LogicalVolume *log = phys->GetLogicalVolume();
      if (!log)
      {
        return nullptr;
      }
      return log->GetRegion();
    }

    static inline std::string regionName(const G4Region *region)
    {
      return region ? std::string(region->GetName()) : std::string("world");
    }

    static inline std::string trackBirthRegionName(const G4Track *track)
    {
      if (!track)
      {
        return "world";
      }
      const G4LogicalVolume *log = track->GetLogicalVolumeAtVertex();
      if (!log)
      {
        return "world";
      }
      const G4Region *region = log->GetRegion();
      return regionName(region);
    }

    // -----------------------------------------------------------------------
    // DD4benchRegionTrackingAction
    //
    // Records the birth region of every track at creation time so that the
    // stepping action can charge time to the originating region (View B).
    // -----------------------------------------------------------------------

    class DD4benchRegionTrackingAction : public Geant4TrackingAction
    {
    public:
      DD4benchRegionTrackingAction(
          Geant4Context *ctx,
          const std::string &name)
          : Geant4TrackingAction(ctx, name)
      {
      }

      virtual ~DD4benchRegionTrackingAction() = default;

      void begin(const G4Track *track) override
      {
        auto &state = RegionTimingState::instance();
        if (!state.enabled || !track)
        {
          return;
        }
        state.birthRegion[track->GetTrackID()] = trackBirthRegionName(track);
      }

      void end(const G4Track *track) override
      {
        auto &state = RegionTimingState::instance();
        if (!state.enabled || !track)
        {
          return;
        }
        // Reclaim memory once the track is done. The stepping action only
        // ever queries birthRegion[currentTrackID], so erasing here is safe
        // and keeps the map size bounded for shower-heavy events.
        state.birthRegion.erase(track->GetTrackID());
      }
    };

    // Forward declaration so the event action can call into the writer.
    static void finalizeAndWrite();

    // -----------------------------------------------------------------------
    // DD4benchRegionTimingAction
    //
    // The stepping action. Hot path: two timer reads, two map increments,
    // one map lookup, one counter increment.
    //
    // Event lifecycle (reset, snapshot, write) lives entirely in the
    // companion event action below.
    // -----------------------------------------------------------------------

    class DD4benchRegionTimingAction : public Geant4SteppingAction
    {
    private:
      // Last timestamp seen (ticks if rdtscp, ns if steady fallback).
      std::uint64_t m_lastTick{0};
      bool m_haveLastTick{false};

      // Attribution for the *previous* step. Time between tick_i and
      // tick_{i+1} is charged to the region the step at tick_i was in.
      std::string m_lastAtLocation;
      std::string m_lastByBirth;

    public:
      DD4benchRegionTimingAction(
          Geant4Context *ctx,
          const std::string &name)
          : Geant4SteppingAction(ctx, name)
      {
        const char *env = std::getenv("DD4BENCH_REGION_JSON");
        auto &state = RegionTimingState::instance();
        state.outputFile = env ? env : "dd4bench_regions.json";

        constexpr std::size_t reserveSize = 10000;
        state.eventNumbers.reserve(reserveSize);
        state.atLocationHistory.reserve(reserveSize);
        state.byBirthHistory.reserve(reserveSize);
        state.intervalCountHistory.reserve(reserveSize);
        state.eventWallSeconds.reserve(reserveSize);
        state.eventRegionSumSeconds.reserve(reserveSize);
        state.eventUnaccountedSeconds.reserve(reserveSize);
        state.eventBirthFallbacks.reserve(reserveSize);

        printout(
            INFO,
            "DD4benchRegionTimingAction",
            "Will write per-region metrics to %s at end of run",
            state.outputFile.c_str());
      }

      virtual ~DD4benchRegionTimingAction()
      {
        // Belt-and-suspenders write in case the event action's destructor
        // did not fire (e.g. crash mid-shutdown). finalizeAndWrite() is
        // idempotent.
        finalizeAndWrite();
      }

      // Allow event action to reset the per-step state at event begin.
      void resetStepState()
      {
        m_haveLastTick = false;
        m_lastAtLocation.clear();
        m_lastByBirth.clear();
      }

      // Called on every step. Hot path.
      void operator()(const G4Step *step, G4SteppingManager * /*mgr*/) override
      {
        auto &state = RegionTimingState::instance();
        if (!state.enabled)
        {
          return;
        }

        const std::uint64_t nowTick = state.timer.now();

        if (m_haveLastTick)
        {
          const std::uint64_t delta = nowTick - m_lastTick;
          state.atLocationTicks[m_lastAtLocation] += delta;
          state.byBirthTicks[m_lastByBirth] += delta;
          state.intervalCounts[m_lastAtLocation] += 1;
        }

        // Update attribution for the next interval.
        m_lastAtLocation = regionName(stepRegion(step));

        const G4Track *track = step ? step->GetTrack() : nullptr;
        if (track)
        {
          auto it = state.birthRegion.find(track->GetTrackID());
          if (it != state.birthRegion.end())
          {
            m_lastByBirth = it->second;
          }
          else
          {
            // Tracking action did not stamp this track. Should not happen
            // in steady state, but can occur at the very first step of an
            // event depending on Geant4 callback ordering. Stamp it now
            // and bump a diagnostic counter so the user can spot a wiring
            // problem (counter shows up in the JSON per event).
            const std::string birth = trackBirthRegionName(track);
            state.birthRegion[track->GetTrackID()] = birth;
            m_lastByBirth = birth;
            state.birthFallbackCount += 1;
          }
        }
        else
        {
          m_lastByBirth = m_lastAtLocation;
        }

        m_lastTick = nowTick;
        m_haveLastTick = true;
      }
    };

    // -----------------------------------------------------------------------
    // DD4benchRegionEventAction
    //
    // Owns the event lifecycle:
    //   - First-event-only: trigger calibration + MT check.
    //   - Begin: clear per-event accumulators, record event number, start
    //     wall clock, reset the stepping action's per-step state.
    //   - End: convert accumulated ticks to seconds, snapshot into history,
    //     record wall time and unaccounted-time delta.
    //   - On destruction: trigger final JSON write (idempotent).
    // -----------------------------------------------------------------------

    class DD4benchRegionEventAction : public Geant4EventAction
    {
    private:
      using Clock = std::chrono::steady_clock;
      Clock::time_point m_eventWallStart;
      std::uint64_t m_eventBeginFallbackBaseline{0};

    public:
      DD4benchRegionEventAction(
          Geant4Context *ctx,
          const std::string &name)
          : Geant4EventAction(ctx, name)
      {
      }

      virtual ~DD4benchRegionEventAction()
      {
        finalizeAndWrite();
      }

      void begin(const G4Event *event) override
      {
        auto &state = RegionTimingState::instance();

        state.initOnFirstEvent();

        if (!state.enabled)
        {
          return;
        }

        state.atLocationTicks.clear();
        state.byBirthTicks.clear();
        state.intervalCounts.clear();
        state.birthRegion.clear();

        m_eventBeginFallbackBaseline = state.birthFallbackCount;

        state.eventNumbers.push_back(event ? event->GetEventID() : -1);

        m_eventWallStart = Clock::now();

        // Reset the stepping action's local "previous tick" so the first
        // step of the new event does not get a stale delta from the end
        // of the last event.
        Geant4Action *act = context()->kernel().steppingAction().get(
            "DD4benchRegionTimingAction");
        if (auto *step = dynamic_cast<DD4benchRegionTimingAction *>(act))
        {
          step->resetStepState();
        }
      }

      void end(const G4Event * /*event*/) override
      {
        auto &state = RegionTimingState::instance();
        if (!state.enabled)
        {
          return;
        }

        const double wallSeconds =
            std::chrono::duration<double>(Clock::now() - m_eventWallStart).count();

        std::map<std::string, double> atLocSec;
        std::map<std::string, double> byBirthSec;
        double regionSum = 0.0;

        for (const auto &kv : state.atLocationTicks)
        {
          const double s = state.timer.toSeconds(kv.second);
          atLocSec[kv.first] = s;
          regionSum += s;
        }
        for (const auto &kv : state.byBirthTicks)
        {
          byBirthSec[kv.first] = state.timer.toSeconds(kv.second);
        }

        state.atLocationHistory.push_back(std::move(atLocSec));
        state.byBirthHistory.push_back(std::move(byBirthSec));
        state.intervalCountHistory.push_back(state.intervalCounts);
        state.eventWallSeconds.push_back(wallSeconds);
        state.eventRegionSumSeconds.push_back(regionSum);
        state.eventUnaccountedSeconds.push_back(wallSeconds - regionSum);
        state.eventBirthFallbacks.push_back(
            state.birthFallbackCount - m_eventBeginFallbackBaseline);
      }
    };

    // -----------------------------------------------------------------------
    // JSON writer
    //
    // Free function so both action destructors can call it. Guarded by a
    // `finalized` flag so multiple calls are safe.
    // -----------------------------------------------------------------------

    static void writeStringQuoted(std::ofstream &out, const std::string &s)
    {
      out << "\"";
      for (char c : s)
      {
        switch (c)
        {
        case '"':
          out << "\\\"";
          break;
        case '\\':
          out << "\\\\";
          break;
        case '\n':
          out << "\\n";
          break;
        case '\r':
          out << "\\r";
          break;
        case '\t':
          out << "\\t";
          break;
        default:
          out << c;
        }
      }
      out << "\"";
    }

    template <typename T>
    static void writeRegionMap(
        std::ofstream &out,
        const std::map<std::string, T> &m,
        int precision)
    {
      out << "{";
      bool first = true;
      for (const auto &kv : m)
      {
        if (!first)
        {
          out << ", ";
        }
        first = false;
        writeStringQuoted(out, kv.first);
        out << ": ";
        if constexpr (std::is_floating_point_v<T>)
        {
          out << std::fixed << std::setprecision(precision) << kv.second;
        }
        else
        {
          out << kv.second;
        }
      }
      out << "}";
    }

    template <typename T>
    static void writeRegionHistory(
        std::ofstream &out,
        const std::string &key,
        const std::vector<std::map<std::string, T>> &history,
        int precision,
        bool last)
    {
      out << "  \"" << key << "\": [";
      for (std::size_t i = 0; i < history.size(); ++i)
      {
        if (i > 0)
        {
          out << ", ";
        }
        writeRegionMap(out, history[i], precision);
      }
      out << (last ? "]\n" : "],\n");
    }

    template <typename T>
    static void writeScalarArray(
        std::ofstream &out,
        const std::string &key,
        const std::vector<T> &values,
        int precision,
        bool last)
    {
      out << "  \"" << key << "\": [";
      for (std::size_t i = 0; i < values.size(); ++i)
      {
        if (i > 0)
        {
          out << ", ";
        }
        if constexpr (std::is_floating_point_v<T>)
        {
          out << std::fixed << std::setprecision(precision) << values[i];
        }
        else
        {
          out << values[i];
        }
      }
      out << (last ? "]\n" : "],\n");
    }

    static void finalizeAndWrite()
    {
      auto &state = RegionTimingState::instance();

      if (state.finalized)
      {
        return;
      }
      if (!state.enabled)
      {
        state.finalized = true;
        return;
      }
      if (state.atLocationHistory.empty())
      {
        state.finalized = true;
        return;
      }

      std::ofstream out(
          state.outputFile,
          std::ios::out | std::ios::trunc);

      if (!out.is_open())
      {
        printout(
            ERROR,
            "DD4benchRegionTimingAction",
            "Could not open output file: %s",
            state.outputFile.c_str());
        state.finalized = true;
        return;
      }

      out << "{\n";

      out << "  \"schema_version\": 1,\n";
      out << "  \"timer\": \""
          << (state.timer.useRdtscp ? "rdtscp" : "steady_clock")
          << "\",\n";
      out << "  \"per_step_overhead_ns\": "
          << std::fixed << std::setprecision(2)
          << (state.perStepOverheadSeconds * 1e9) << ",\n";
      out << "  \"interval_counts_note\": \"Counts of timer intervals "
          << "attributed to each region, not strict step counts. Off by "
          << "one from 'steps in region' at event boundaries.\",\n";

      writeScalarArray(out, "event_numbers", state.eventNumbers, 0, false);
      writeScalarArray(out, "event_wall_seconds", state.eventWallSeconds, 6, false);
      writeScalarArray(out, "event_region_sum_seconds", state.eventRegionSumSeconds, 6, false);
      writeScalarArray(out, "event_unaccounted_seconds", state.eventUnaccountedSeconds, 6, false);
      writeScalarArray(out, "event_birth_fallbacks", state.eventBirthFallbacks, 0, false);

      writeRegionHistory(out, "at_location_seconds",
                         state.atLocationHistory, 6, false);
      writeRegionHistory(out, "by_birth_seconds",
                         state.byBirthHistory, 6, false);
      writeRegionHistory(out, "interval_counts",
                         state.intervalCountHistory, 0, true);

      out << "}\n";

      printout(
          INFO,
          "DD4benchRegionTimingAction",
          "Per-region metrics written to %s (%zu events, %llu total birth-region fallbacks)",
          state.outputFile.c_str(),
          state.atLocationHistory.size(),
          static_cast<unsigned long long>(state.birthFallbackCount));

      state.finalized = true;
    }

  } // namespace sim
} // namespace dd4hep

#include <DDG4/Factories.h>

DECLARE_GEANT4ACTION(DD4benchRegionTimingAction)
DECLARE_GEANT4ACTION(DD4benchRegionTrackingAction)
DECLARE_GEANT4ACTION(DD4benchRegionEventAction)