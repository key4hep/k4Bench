// k4BenchTimingAction.cpp
//
// DDG4 event action that records per-event wall time and RSS memory.
//
// The plugin is intentionally lightweight:
// - Measures per-event wall time using a monotonic clock
// - Samples RSS memory before/after each event
// - Writes JSON metrics at shutdown
//
// Output path is controlled via:
//
//   K4BENCH_EVENT_JSON=/path/to/output.json
//
// If unset, defaults to:
//
//   k4bench_events.json
//
// NOTE:
// This implementation currently assumes sequential event processing.
// The internal vectors are not protected for multithreaded Geant4 runs.

#include <DDG4/Geant4EventAction.h>
#include <DDG4/Geant4Context.h>
#include <DD4hep/Printout.h>
#include <G4Event.hh>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

namespace dd4hep
{
  namespace sim
  {

    // ---------------------------------------------------------------------------
    // Read current RSS from /proc/self/status (Linux only)
    // ---------------------------------------------------------------------------

    static long read_rss_kb()
    {
      std::ifstream status("/proc/self/status");

      if (!status.is_open())
      {
        return -1;
      }

      std::string line;

      while (std::getline(status, line))
      {
        if (line.rfind("VmRSS:", 0) == 0)
        {
          std::istringstream iss(line.substr(6));

          long kb = -1;
          iss >> kb;

          return kb;
        }
      }

      return -1;
    }

    // ---------------------------------------------------------------------------
    // k4BenchTimingAction
    // ---------------------------------------------------------------------------

    class k4BenchTimingAction : public Geant4EventAction
    {
    public:
      std::string m_outputFile;

    private:
      using Clock = std::chrono::steady_clock;
      using TimePoint = std::chrono::time_point<Clock>;

      TimePoint m_eventStart;
      long m_rssBegin{-1};

      std::vector<int> m_eventNumbers;
      std::vector<double> m_eventTimes;
      std::vector<long> m_rssBeginValues;
      std::vector<long> m_rssEndValues;

    public:
      k4BenchTimingAction(
          Geant4Context *ctx,
          const std::string &name)
          : Geant4EventAction(ctx, name)
      {
        const char *env = std::getenv("K4BENCH_EVENT_JSON");

        m_outputFile = env ? env : "k4bench_events.json";

        constexpr std::size_t reserveSize = 10000;

        m_eventNumbers.reserve(reserveSize);
        m_eventTimes.reserve(reserveSize);
        m_rssBeginValues.reserve(reserveSize);
        m_rssEndValues.reserve(reserveSize);

        printout(
            INFO,
            "k4BenchTimingAction",
            "Writing per-event metrics to %s",
            m_outputFile.c_str());
      }

      virtual ~k4BenchTimingAction()
      {
        writeResults();
      }

      void begin(const G4Event *event) override
      {
        m_rssBegin = read_rss_kb();
        m_eventStart = Clock::now();
        m_eventNumbers.push_back(event->GetEventID());
      }

      void end(const G4Event * /* event */) override
      {
        auto elapsed = Clock::now() - m_eventStart;

        m_eventTimes.push_back(
            std::chrono::duration<double>(elapsed).count());

        m_rssBeginValues.push_back(m_rssBegin);
        m_rssEndValues.push_back(read_rss_kb());
      }

    private:
      template <typename T>
      void writeArray(
          std::ofstream &out,
          const std::string &key,
          const std::vector<T> &values,
          int precision,
          double scale = 1.0,
          bool last = false)
      {
        out << "  \"" << key << "\": [";

        for (std::size_t i = 0; i < values.size(); ++i)
        {
          if (i > 0)
          {
            out << ", ";
          }

          out << std::fixed
              << std::setprecision(precision)
              << (static_cast<double>(values[i]) * scale);
        }

        out << (last ? "]\n" : "],\n");
      }

      void writeResults()
      {
        if (m_eventTimes.empty())
        {
          return;
        }

        std::ofstream out(
            m_outputFile,
            std::ios::out | std::ios::trunc);

        if (!out.is_open())
        {
          printout(
              ERROR,
              "k4BenchTimingAction",
              "Could not open output file: %s",
              m_outputFile.c_str());
          return;
        }

        out << "{\n";

        writeArray(out, "event_numbers", m_eventNumbers, 0);
        writeArray(out, "event_times_s", m_eventTimes, 6);
        writeArray(out, "event_rss_begin_mb", m_rssBeginValues, 3, 1.0 / 1024.0);
        writeArray(out, "event_rss_end_mb", m_rssEndValues, 3, 1.0 / 1024.0, /*last=*/true);

        out << "}\n";

        printout(
            INFO,
            "k4BenchTimingAction",
            "Per-event metrics written to %s (%zu events)",
            m_outputFile.c_str(),
            m_eventTimes.size());
      }
    };

  } // namespace sim
} // namespace dd4hep

#include <DDG4/Factories.h>

DECLARE_GEANT4ACTION(k4BenchTimingAction)