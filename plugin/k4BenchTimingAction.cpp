// k4BenchTimingAction.cpp
//
// DDG4 event action that records per-event wall time and RSS memory.
//
// The plugin is intentionally lightweight:
// - Measures per-event wall time using a monotonic clock
// - Samples RSS memory before/after each event
// - Reads the kernel virtual-size high-water mark at shutdown
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
    // Read current RSS and its components from /proc/self/status (Linux only)
    // ---------------------------------------------------------------------------

    struct RssValues
    {
      long total{-1};
      long anon{-1};
      long file{-1};
    };

    static RssValues read_rss_kb()
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

    static long read_vmpeak_kb()
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
      long m_rssAnonBegin{-1};

      std::vector<int> m_eventNumbers;
      std::vector<double> m_eventTimes;
      std::vector<long> m_rssBeginValues;
      std::vector<long> m_rssEndValues;
      std::vector<long> m_rssAnonBeginValues;
      std::vector<long> m_rssAnonEndValues;
      std::vector<long> m_rssFileEndValues;

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
        m_rssAnonBeginValues.reserve(reserveSize);
        m_rssAnonEndValues.reserve(reserveSize);
        m_rssFileEndValues.reserve(reserveSize);

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

      void begin(const G4Event * /* event */) override
      {
        const auto rss = read_rss_kb();
        m_rssBegin = rss.total;
        m_rssAnonBegin = rss.anon;
        m_eventStart = Clock::now();
      }

      void end(const G4Event *event) override
      {
        auto elapsed = Clock::now() - m_eventStart;

        m_eventNumbers.push_back(event->GetEventID());
        m_eventTimes.push_back(
            std::chrono::duration<double>(elapsed).count());

        const auto rss = read_rss_kb();
        m_rssBeginValues.push_back(m_rssBegin);
        m_rssEndValues.push_back(rss.total);
        m_rssAnonBeginValues.push_back(m_rssAnonBegin);
        m_rssAnonEndValues.push_back(rss.anon);
        m_rssFileEndValues.push_back(rss.file);
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
        const long vmpeak = read_vmpeak_kb();
        out << "  \"peak_vmem_mb\": " << std::fixed << std::setprecision(3)
            << (vmpeak < 0 ? -1.0 : vmpeak / 1024.0) << ",\n";

        writeArray(out, "event_numbers", m_eventNumbers, 0);
        writeArray(out, "event_times_s", m_eventTimes, 6);
        writeArray(out, "event_rss_begin_mb", m_rssBeginValues, 3, 1.0 / 1024.0);
        writeArray(out, "event_rss_end_mb", m_rssEndValues, 3, 1.0 / 1024.0);
        writeArray(out, "event_rss_anon_begin_mb", m_rssAnonBeginValues, 3, 1.0 / 1024.0);
        writeArray(out, "event_rss_anon_end_mb", m_rssAnonEndValues, 3, 1.0 / 1024.0);
        writeArray(out, "event_rss_file_end_mb", m_rssFileEndValues, 3, 1.0 / 1024.0, /*last=*/true);

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
