// DD4benchRegionTimingAction.cpp
//
// DDG4 stepping + tracking + event actions that record per-event,
// per-(top-level-)detector wall time spent inside Geant4 stepping. The
// attribution is based on DD4hep DetElement children of the world (the
// same notion of "subdetector" used by ablation tools), not on Geant4
// G4Region. This matters because in many production geometries
// (e.g. FCCee ALLEGRO o1_v03) only a subset of detectors have explicit
// regions, so a region-based view dumps most of the simulation time into
// "DefaultRegionForTheWorld".
//
// Two complementary views are produced:
//
//   - "at_location": time charged to the detector the step is currently in.
//   - "by_birth":    time charged to the detector in which the track was
//                    created. Charges secondaries back to where they were
//                    born, regardless of where they end up being tracked.
//
// The gap between the two views is informative: it tells you how much of a
// detector's cost is intrinsic vs. imported from upstream showers.
//
// Steps that fall outside any DetElement (typically vacuum transport
// through the world volume, or beampipe-like structures that are not
// modelled as their own DetElement) are bucketed as "unattributed".
//
// Output path is controlled via:
//
//   DD4BENCH_REGION_JSON=/path/to/output.json
//
// If unset, defaults to:
//
//   dd4bench_regions.json
//
// (The env var name and class names keep the word "Region" for backward
// compatibility with the build system and steering scripts. The output
// schema is keyed by DD4hep DetElement name; see "attribution" field.)
//
// Overhead notes:
// - Uses __rdtscp where available (x86_64 Linux), falling back to
//   std::chrono::steady_clock otherwise.
// - TSC frequency is calibrated against steady_clock at first event begin.
// - LogicalVolume -> detector-name lookups are cached in an unordered_map
//   so each LV is walked at most once per process. After warmup the hot
//   path is one map lookup + two timer reads + two map increments.
//
// Multithreading:
// - This plugin assumes single-threaded simulation. At first event begin
//   it checks G4Threading::IsMultithreadedApplication() and disables
//   itself (with an ERROR printout) if MT is detected.
//
// Output schema version: 1.

#include <DDG4/Geant4SteppingAction.h>
#include <DDG4/Geant4TrackingAction.h>
#include <DDG4/Geant4EventAction.h>
#include <DDG4/Geant4Context.h>
#include <DDG4/Geant4Kernel.h>
#include <DD4hep/Detector.h>
#include <DD4hep/DetElement.h>
#include <DD4hep/Volumes.h>
#include <DD4hep/Printout.h>

#include <TGeoNode.h>
#include <TGeoVolume.h>

#include <G4Step.hh>
#include <G4Track.hh>
#include <G4Event.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4VPhysicalVolume.hh>
#include <G4TouchableHandle.hh>
#include <G4Threading.hh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
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

    static const std::string kUnattributed = "unattributed";

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
    // Detector-volume index
    //
    // Built once at first event begin from the DD4hep Detector singleton.
    // Holds two structures:
    //
    //   - topLogVols:  set of G4LogicalVolume* corresponding to the
    //                  placement of each top-level DetElement.
    //                  (i.e. detector.world().children())
    //   - lvToDetector: cache of "ancestor walk" results. For a given
    //                   LogicalVolume L, lvToDetector[L] is the name of
    //                   the top-level detector that contains L (or
    //                   "unattributed" if none does).
    //
    // The cache is populated lazily by the stepping action: on first
    // encounter of an L, we walk up its placement hierarchy to the world
    // and check at each level whether the volume is in topLogVols. The
    // result is memoized so subsequent steps in the same L are O(1).
    // -----------------------------------------------------------------------

    struct DetectorVolumeIndex
    {
      // top-level placement LV -> detector name
      std::unordered_map<const G4LogicalVolume *, std::string> topLogVols;

      // any LV -> resolved detector name (after walk)
      std::unordered_map<const G4LogicalVolume *, std::string> lvToDetector;

      bool built{false};

      // Walk the DD4hep DetElement tree starting from the world's children
      // and record the G4 logical volumes that should attribute time to
      // each top-level detector.
      //
      // Two cases:
      //
      //   1. Solid top-level volume (e.g. ECalBarrel's envelope): record
      //      its G4LogicalVolume directly.
      //
      //   2. Assembly top-level volume (e.g. BeBeampipe_assembly):
      //      Geant4 dissolves assemblies at translation, so the assembly
      //      LV itself does not exist in G4LogicalVolumeStore. Instead we
      //      index every G4LogicalVolume reachable from the assembly's
      //      child placements, all mapped to the top-level DetElement
      //      name.
      //
      // For assemblies we walk the TGeo daughter tree recursively because
      // assemblies can nest other assemblies. Each "real" (non-assembly)
      // daughter volume found contributes its G4LV to topLogVols.
      void buildFromDD4hep()
      {
        if (built)
        {
          return;
        }
        built = true;

        Detector &detector = Detector::getInstance();

        const auto children = detector.world().children();

        std::size_t solidCount = 0;
        std::size_t assemblyCount = 0;
        std::size_t assemblyLeafCount = 0;
        std::size_t failedCount = 0;

        for (const auto &kv : children)
        {
          const std::string &name = kv.first;
          DetElement de = kv.second;

          PlacedVolume pv = de.placement();
          if (!pv.isValid())
          {
            continue;
          }
          Volume vol = pv.volume();
          if (!vol.isValid())
          {
            continue;
          }

          if (vol.isAssembly())
          {
            // Walk the assembly's TGeo node tree and index every concrete
            // (non-assembly) descendant's G4LV under this detector's name.
            const std::size_t before = topLogVols.size();
            const std::size_t daughterCount = countAssemblyDaughters(vol);
            indexAssemblyDescendants(vol, name);
            const std::size_t added = topLogVols.size() - before;

            if (added == 0)
            {
              if (daughterCount == 0)
              {
                // Empty assembly with no daughter nodes at all. This is a
                // legitimate geometry pattern: DetElements registered
                // purely for field maps, alignment frames, or other
                // bookkeeping have no placed volumes. Nothing to
                // attribute, and that's correct.
                printout(
                    INFO,
                    "DD4benchRegionTimingAction",
                    "Top-level detector '%s' is an empty assembly "
                    "(no daughter volumes). Skipping; no time can be "
                    "attributed to it.",
                    name.c_str());
              }
              else
              {
                // Assembly had daughters but none resolved to a
                // G4LogicalVolume. This IS a problem worth warning
                // about -- the geometry has real volumes here that we
                // can't attribute.
                printout(
                    WARNING,
                    "DD4benchRegionTimingAction",
                    "Top-level detector '%s' is an assembly with %zu "
                    "daughter node(s) but none resolved to a "
                    "G4LogicalVolume; steps inside it will be "
                    "'unattributed'.",
                    name.c_str(),
                    daughterCount);
              }
              failedCount += 1;
            }
            else
            {
              assemblyCount += 1;
              assemblyLeafCount += added;
            }
            continue;
          }

          // Regular solid volume: index it directly.
          const std::string volName = vol.name();
          const G4LogicalVolume *lv = findG4LogicalVolumeByName(volName);
          if (lv == nullptr)
          {
            printout(
                WARNING,
                "DD4benchRegionTimingAction",
                "Could not find G4LogicalVolume for top-level detector "
                "'%s' (volume '%s'); steps inside it will be "
                "'unattributed'.",
                name.c_str(),
                volName.c_str());
            failedCount += 1;
            continue;
          }

          topLogVols[lv] = name;
          lvToDetector[lv] = name;
          solidCount += 1;
        }

        printout(
            INFO,
            "DD4benchRegionTimingAction",
            "Indexed %zu top-level detectors for timing attribution "
            "(%zu solid + %zu assemblies covering %zu child LVs; %zu failed).",
            solidCount + assemblyCount,
            solidCount,
            assemblyCount,
            assemblyLeafCount,
            failedCount);
      }

      // Count direct daughters of an assembly without descending. Used to
      // distinguish "empty assembly" (legitimate, e.g., field-only
      // DetElement) from "assembly with daughters we couldn't resolve"
      // (real problem).
      static std::size_t countAssemblyDaughters(const Volume &assemblyVol)
      {
        TGeoVolume *tvol = assemblyVol.ptr();
        if (tvol == nullptr)
        {
          return 0;
        }
        return static_cast<std::size_t>(tvol->GetNdaughters());
      }

      // Recursively walk an assembly volume's daughter tree, registering
      // every concrete (non-assembly) G4LogicalVolume found under the
      // given detector name. Nested assemblies are descended into.
      void indexAssemblyDescendants(
          const Volume &assemblyVol,
          const std::string &detectorName)
      {
        TGeoVolume *tvol = assemblyVol.ptr();
        if (tvol == nullptr)
        {
          return;
        }
        const Int_t nDaughters = tvol->GetNdaughters();
        for (Int_t i = 0; i < nDaughters; ++i)
        {
          TGeoNode *node = tvol->GetNode(i);
          if (node == nullptr)
          {
            continue;
          }
          TGeoVolume *daughterTVol = node->GetVolume();
          if (daughterTVol == nullptr)
          {
            continue;
          }
          Volume daughter(daughterTVol);
          if (daughter.isAssembly())
          {
            // Nested assembly: recurse without registering this level.
            indexAssemblyDescendants(daughter, detectorName);
            continue;
          }
          // Concrete volume: find its G4LV and register it.
          const std::string daughterName = daughterTVol->GetName();
          const G4LogicalVolume *lv = findG4LogicalVolumeByName(daughterName);
          if (lv == nullptr)
          {
            // Some daughters may have name decorations (replicas, etc.).
            // Skip silently; the assembly-level warning will fire only
            // if NOTHING is indexed.
            continue;
          }
          // First registration wins. If an LV is somehow reachable from
          // two top-level detectors (shouldn't happen for direct world
          // children, but defensive), the first one named owns it. We
          // log the collision so it's not silent.
          auto existing = topLogVols.find(lv);
          if (existing == topLogVols.end())
          {
            topLogVols[lv] = detectorName;
            lvToDetector[lv] = detectorName;
          }
          else if (existing->second != detectorName)
          {
            printout(
                WARNING,
                "DD4benchRegionTimingAction",
                "G4LogicalVolume '%s' is reachable from both top-level "
                "detector '%s' (first claim) and '%s' (ignored). Steps "
                "in this LV will be attributed to the first claimer.",
                daughterName.c_str(),
                existing->second.c_str(),
                detectorName.c_str());
          }
        }
      }

      // Resolve a LogicalVolume to its containing top-level detector name.
      //
      // Caching policy: we ONLY cache positive results that depend solely
      // on the LogicalVolume identity (i.e., the LV is itself a registered
      // top-level placement). We do NOT cache results obtained by walking
      // the touchable ancestry, because the same LV can be placed under
      // different parents in principle, reaching different top-level
      // detectors. Caching the walk result would force one answer
      // globally and silently misattribute on the second placement.
      //
      // In practice for ALLEGRO each detector's LVs are unique to that
      // detector, so the touchable walk gives the same answer every time,
      // but we don't rely on that assumption.
      const std::string &resolveByTouchable(
          const G4LogicalVolume *currentLv,
          const G4VTouchable *touch)
      {
        if (currentLv == nullptr)
        {
          return kUnattributed;
        }

        // Direct hit: currentLv is itself a registered top-level placement.
        // This case is LV-only and safe to cache (and was pre-cached at
        // index time).
        auto direct = lvToDetector.find(currentLv);
        if (direct != lvToDetector.end())
        {
          return direct->second;
        }

        // Walk up the touchable hierarchy looking for an ancestor that IS
        // registered. Do NOT cache the result against currentLv, because
        // the answer depended on the specific placement chain, not on the
        // LV identity alone.
        if (touch != nullptr)
        {
          const int depth = touch->GetHistoryDepth();
          for (int d = 0; d <= depth; ++d)
          {
            const G4VPhysicalVolume *pv = touch->GetVolume(d);
            if (pv == nullptr)
            {
              continue;
            }
            const G4LogicalVolume *lv = pv->GetLogicalVolume();
            auto it = topLogVols.find(lv);
            if (it != topLogVols.end())
            {
              return it->second;
            }
          }
        }

        // No ancestor matched. Don't cache "unattributed": the same LV
        // may be visited again under a different placement that resolves
        // to a registered detector.
        return kUnattributed;
      }

    private:
      // Look up a G4LogicalVolume by name from the global LV store. Slow
      // (linear scan), called only once per top-level detector at init.
      static const G4LogicalVolume *findG4LogicalVolumeByName(
          const std::string &name)
      {
        const G4LogicalVolumeStore *store = G4LogicalVolumeStore::GetInstance();
        if (store == nullptr)
        {
          return nullptr;
        }
        for (const G4LogicalVolume *lv : *store)
        {
          if (lv != nullptr && lv->GetName() == name)
          {
            return lv;
          }
        }
        return nullptr;
      }
    };

    // -----------------------------------------------------------------------
    // Shared state between stepping action, tracking action, and event
    // action.
    //
    // Single-threaded by design. Held as a function-local static so all
    // three plugins see the same instance regardless of construction order.
    // -----------------------------------------------------------------------

    struct RegionTimingState
    {
      TscTimer timer;
      DetectorVolumeIndex detIndex;
      bool calibrated{false};
      bool enabled{true};

      // Track ID -> birth detector name. Cleared at event begin.
      std::unordered_map<int, std::string> birthDetector;

      // Per-event accumulators.
      std::map<std::string, std::uint64_t> atLocationTicks;
      std::map<std::string, std::uint64_t> byBirthTicks;
      std::map<std::string, std::uint64_t> intervalCounts;

      // Diagnostic counter for tracking-action / stepping-action ordering.
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

      double perStepOverheadSeconds{0.0};
      std::string outputFile{"dd4bench_regions.json"};
      bool finalized{false};

      static RegionTimingState &instance()
      {
        static RegionTimingState s;
        return s;
      }

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
              "Multithreaded Geant4 detected; disabling per-detector "
              "timing. This plugin is single-threaded only.");
          return;
        }

        timer.calibrate();
        measureOverhead();
        detIndex.buildFromDD4hep();
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

    // Resolve a G4Step to a detector name using the cached index.
    static inline const std::string &stepDetectorName(
        const G4Step *step,
        DetectorVolumeIndex &index)
    {
      if (step == nullptr)
      {
        return kUnattributed;
      }
      const G4StepPoint *pre = step->GetPreStepPoint();
      if (pre == nullptr)
      {
        return kUnattributed;
      }
      const G4VPhysicalVolume *phys = pre->GetPhysicalVolume();
      if (phys == nullptr)
      {
        return kUnattributed;
      }
      const G4LogicalVolume *lv = phys->GetLogicalVolume();
      const G4VTouchable *touch = pre->GetTouchable();
      return index.resolveByTouchable(lv, touch);
    }

    // Resolve a track's birth detector. We can't ask a touchable here
    // because the track only stores its production-vertex LogicalVolume
    // (GetLogicalVolumeAtVertex), not the placement hierarchy. We fall
    // back to a direct LV cache check; if it's a known top-level detector
    // we return that, else "unattributed". For showers this is usually
    // correct because secondaries are born inside the same LV as their
    // parent's step, which has already been resolved.
    static inline const std::string &trackBirthDetectorName(
        const G4Track *track,
        DetectorVolumeIndex &index)
    {
      if (track == nullptr)
      {
        return kUnattributed;
      }
      const G4LogicalVolume *lv = track->GetLogicalVolumeAtVertex();
      if (lv == nullptr)
      {
        return kUnattributed;
      }
      auto it = index.lvToDetector.find(lv);
      if (it != index.lvToDetector.end())
      {
        return it->second;
      }
      // Not yet cached. We don't have a touchable here; the best we can
      // do is check direct top-level membership.
      auto top = index.topLogVols.find(lv);
      if (top != index.topLogVols.end())
      {
        index.lvToDetector[lv] = top->second;
        return index.lvToDetector[lv];
      }
      // Don't pollute the cache with an "unattributed" verdict for this
      // LV based on incomplete info; the stepping action will likely
      // visit it with a touchable soon and resolve it properly.
      return kUnattributed;
    }

    // -----------------------------------------------------------------------
    // DD4benchRegionTrackingAction
    //
    // Records the birth detector of every track at creation time so the
    // stepping action can charge time to the originating detector.
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
        if (!state.enabled || track == nullptr)
        {
          return;
        }
        state.birthDetector[track->GetTrackID()] =
            trackBirthDetectorName(track, state.detIndex);
      }

      void end(const G4Track *track) override
      {
        auto &state = RegionTimingState::instance();
        if (!state.enabled || track == nullptr)
        {
          return;
        }
        state.birthDetector.erase(track->GetTrackID());
      }
    };

    // Forward declaration.
    static void finalizeAndWrite();

    // -----------------------------------------------------------------------
    // DD4benchRegionTimingAction
    //
    // The stepping action. Hot path: two timer reads, one cache lookup,
    // two map increments, one counter increment.
    // -----------------------------------------------------------------------

    class DD4benchRegionTimingAction : public Geant4SteppingAction
    {
    private:
      std::uint64_t m_lastTick{0};
      bool m_haveLastTick{false};
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
            "Will write per-detector metrics to %s at end of run",
            state.outputFile.c_str());
      }

      virtual ~DD4benchRegionTimingAction()
      {
        finalizeAndWrite();
      }

      void resetStepState()
      {
        m_haveLastTick = false;
        m_lastAtLocation.clear();
        m_lastByBirth.clear();
      }

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
          // Guard against non-monotonic ticks. With __rdtscp this is rare
          // but possible across core migrations or clock resync events.
          // Without the guard, unsigned subtraction would produce a huge
          // bogus delta that corrupts all timing for the event.
          std::uint64_t delta = 0;
          if (nowTick >= m_lastTick)
          {
            delta = nowTick - m_lastTick;
          }
          state.atLocationTicks[m_lastAtLocation] += delta;
          state.byBirthTicks[m_lastByBirth] += delta;
          state.intervalCounts[m_lastAtLocation] += 1;
        }

        // Update attribution for the next interval.
        m_lastAtLocation = stepDetectorName(step, state.detIndex);

        const G4Track *track = step ? step->GetTrack() : nullptr;
        if (track != nullptr)
        {
          auto it = state.birthDetector.find(track->GetTrackID());
          if (it != state.birthDetector.end())
          {
            m_lastByBirth = it->second;
          }
          else
          {
            // Tracking action did not stamp this track. Stamp it now
            // using the current step's location as a best-effort
            // approximation (the track is, after all, currently here)
            // and bump the diagnostic counter.
            const std::string &birth = m_lastAtLocation;
            state.birthDetector[track->GetTrackID()] = birth;
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
        state.birthDetector.clear();

        m_eventBeginFallbackBaseline = state.birthFallbackCount;

        state.eventNumbers.push_back(event ? event->GetEventID() : -1);

        m_eventWallStart = Clock::now();

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

      std::ofstream out(state.outputFile, std::ios::out | std::ios::trunc);
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
      out << "  \"attribution\": \"dd4hep_top_level_detelement\",\n";
      out << "  \"timer\": \""
          << (state.timer.useRdtscp ? "rdtscp" : "steady_clock")
          << "\",\n";
      out << "  \"per_step_timer_overhead_ns\": "
          << std::fixed << std::setprecision(2)
          << (state.perStepOverheadSeconds * 1e9) << ",\n";
      out << "  \"per_step_timer_overhead_note\": \"Cost of two timer "
          << "reads plus a map increment, measured at startup. Does NOT "
          << "include the per-step touchable walk or hash lookup costs "
          << "for detector attribution; those are typically 50-100 ns "
          << "additional but vary with placement depth.\",\n";
      out << "  \"interval_counts_note\": \"Counts of timer intervals "
          << "attributed to each detector, not strict step counts. Off "
          << "by one from 'steps in detector' at event boundaries.\",\n";
      // Deduplicate detector names from topLogVols (which has one entry
      // per indexed G4LogicalVolume, so assemblies contribute multiple
      // entries pointing to the same name) and emit both:
      //   - a unique sorted name list (what users actually want to see)
      //   - a per-detector LV count (diagnostic: shows assembly fan-out)
      std::map<std::string, std::size_t> lvCountByDetector;
      for (const auto &kv : state.detIndex.topLogVols)
      {
        lvCountByDetector[kv.second] += 1;
      }

      out << "  \"indexed_top_level_detectors\": [";
      {
        bool first = true;
        for (const auto &kv : lvCountByDetector)
        {
          if (!first)
          {
            out << ", ";
          }
          first = false;
          writeStringQuoted(out, kv.first);
        }
      }
      out << "],\n";

      out << "  \"indexed_top_level_detector_lv_counts\": {";
      {
        bool first = true;
        for (const auto &kv : lvCountByDetector)
        {
          if (!first)
          {
            out << ", ";
          }
          first = false;
          writeStringQuoted(out, kv.first);
          out << ": " << kv.second;
        }
      }
      out << "},\n";

      writeScalarArray(out, "event_numbers", state.eventNumbers, 0, false);
      writeScalarArray(out, "event_wall_seconds", state.eventWallSeconds, 6, false);
      writeScalarArray(out, "event_region_sum_seconds", state.eventRegionSumSeconds, 6, false);
      writeScalarArray(out, "event_unaccounted_seconds", state.eventUnaccountedSeconds, 6, false);
      writeScalarArray(out, "event_birth_fallbacks", state.eventBirthFallbacks, 0, false);

      writeRegionHistory(out, "at_location_seconds", state.atLocationHistory, 6, false);
      writeRegionHistory(out, "by_birth_seconds", state.byBirthHistory, 6, false);
      writeRegionHistory(out, "interval_counts", state.intervalCountHistory, 0, true);

      out << "}\n";

      printout(
          INFO,
          "DD4benchRegionTimingAction",
          "Per-detector metrics written to %s (%zu events, %llu total birth fallbacks)",
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