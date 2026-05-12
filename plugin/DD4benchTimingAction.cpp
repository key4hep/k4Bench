// DD4benchTimingAction.cpp
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
//   DD4BENCH_EVENT_JSON=/path/to/output.json
//
// If unset, defaults to:
//
//   dd4bench_events.json
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
#include <algorithm>
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
    // DD4benchTimingAction
    // ---------------------------------------------------------------------------

    class DD4benchTimingAction : public Geant4EventAction
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
      std::vector<long> m_rssDeltas;
      std::vector<long> m_rssPeaks;

    public:
      DD4benchTimingAction(
          Geant4Context *ctx,
          const std::string &name)
          : Geant4EventAction(ctx, name)
      {

        const char *env = std::getenv("DD4BENCH_EVENT_JSON");

        m_outputFile = env ? env : "dd4bench_events.json";

        // Reduce vector reallocations during benchmarking
        constexpr std::size_t reserveSize = 10000;

        m_eventNumbers.reserve(reserveSize);
        m_eventTimes.reserve(reserveSize);
        m_rssDeltas.reserve(reserveSize);
        m_rssPeaks.reserve(reserveSize);

        printout(
            INFO,
            "DD4benchTimingAction",
            "Writing per-event metrics to %s",
            m_outputFile.c_str());
      }

      virtual ~DD4benchTimingAction()
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

        long rssEnd = read_rss_kb();

        m_rssPeaks.push_back(
            (m_rssBegin >= 0 && rssEnd >= 0)
                ? std::max(m_rssBegin, rssEnd)
                : 0);

        m_rssDeltas.push_back(
            (m_rssBegin >= 0 && rssEnd >= 0)
                ? (rssEnd - m_rssBegin)
                : 0);
      }

    private:
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
              "DD4benchTimingAction",
              "Could not open output file: %s",
              m_outputFile.c_str());
          return;
        }

        out << "{\n";

        // ---------------------------------------------------------------------
        // Event numbers
        // ---------------------------------------------------------------------

        out << "  \"event_numbers\": [";

        for (std::size_t i = 0; i < m_eventNumbers.size(); ++i)
        {
          if (i > 0)
          {
            out << ", ";
          }

          out << m_eventNumbers[i];
        }

        out << "],\n";

        // ---------------------------------------------------------------------
        // Event wall times
        // ---------------------------------------------------------------------

        out << "  \"event_times_s\": [";

        for (std::size_t i = 0; i < m_eventTimes.size(); ++i)
        {
          if (i > 0)
          {
            out << ", ";
          }

          out << std::fixed
              << std::setprecision(6)
              << m_eventTimes[i];
        }

        out << "],\n";

        // ---------------------------------------------------------------------
        // RSS peaks
        // ---------------------------------------------------------------------

        out << "  \"event_rss_peak_mb\": [";

        for (std::size_t i = 0; i < m_rssPeaks.size(); ++i)
        {
          if (i > 0)
          {
            out << ", ";
          }

          out << std::fixed
              << std::setprecision(3)
              << (static_cast<double>(m_rssPeaks[i]) / 1024.0);
        }

        out << "],\n";

        // ---------------------------------------------------------------------
        // RSS deltas
        // ---------------------------------------------------------------------

        out << "  \"event_rss_delta_mb\": [";

        for (std::size_t i = 0; i < m_rssDeltas.size(); ++i)
        {
          if (i > 0)
          {
            out << ", ";
          }

          out << std::fixed
              << std::setprecision(3)
              << (static_cast<double>(m_rssDeltas[i]) / 1024.0);
        }

        out << "]\n";

        out << "}\n";

        printout(
            INFO,
            "DD4benchTimingAction",
            "Per-event metrics written to %s (%zu events)",
            m_outputFile.c_str(),
            m_eventTimes.size());
      }
    };

  } // namespace sim
} // namespace dd4hep

#include <DDG4/Factories.h>

DECLARE_GEANT4ACTION(DD4benchTimingAction)