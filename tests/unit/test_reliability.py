"""Unit tests for k4bench.results.reliability."""

from __future__ import annotations

from k4bench.results.reliability import (
    MIN_CPU_EFFICIENCY,
    ReliabilityVerdict,
    Status,
    evaluate_reliability,
)


def _criterion(verdict: ReliabilityVerdict, name: str):
    return next(c for c in verdict.criteria if c.name == name)


class TestCpuEfficiency:
    def test_clean_single_thread_passes(self):
        v = evaluate_reliability(cpu_efficiency=0.99)
        assert _criterion(v, "CPU efficiency").status is Status.PASS

    def test_below_floor_fails(self):
        v = evaluate_reliability(cpu_efficiency=0.80)
        c = _criterion(v, "CPU efficiency")
        assert c.status is Status.FAIL
        assert v.reliable is False

    def test_floor_scales_with_threads(self):
        # 3.6/4 = 0.90 < 0.95 ideal-per-thread -> fail at 4 threads.
        v = evaluate_reliability(cpu_efficiency=3.6, n_threads=4)
        assert _criterion(v, "CPU efficiency").status is Status.FAIL
        # 3.9/4 = 0.975 >= 0.95 -> pass.
        v2 = evaluate_reliability(cpu_efficiency=3.9, n_threads=4)
        assert _criterion(v2, "CPU efficiency").status is Status.PASS

    def test_missing_is_unknown(self):
        v = evaluate_reliability(cpu_efficiency=None)
        assert _criterion(v, "CPU efficiency").status is Status.UNKNOWN


class TestLoad:
    def test_within_cores_passes(self):
        v = evaluate_reliability(load_avg_1m_pre=4.0, physical_cores=8)
        assert _criterion(v, "System load").status is Status.PASS

    def test_oversubscribed_fails(self):
        v = evaluate_reliability(load_avg_1m_pre=9.0, physical_cores=8)
        assert _criterion(v, "System load").status is Status.FAIL
        assert v.reliable is False

    def test_uses_peak_of_pre_and_post(self):
        v = evaluate_reliability(load_avg_1m_pre=1.0, load_avg_1m_post=20.0, physical_cores=8)
        assert _criterion(v, "System load").status is Status.FAIL

    def test_missing_cores_is_unknown(self):
        v = evaluate_reliability(load_avg_1m_pre=4.0, physical_cores=None)
        assert _criterion(v, "System load").status is Status.UNKNOWN


class TestSwap:
    def test_no_activity_passes(self):
        v = evaluate_reliability(swap_in_pages=0, swap_out_pages=0)
        assert _criterion(v, "Swap activity").status is Status.PASS

    def test_any_activity_fails(self):
        v = evaluate_reliability(swap_in_pages=0, swap_out_pages=128)
        assert _criterion(v, "Swap activity").status is Status.FAIL
        assert v.reliable is False

    def test_missing_is_unknown(self):
        v = evaluate_reliability(swap_in_pages=None, swap_out_pages=None)
        assert _criterion(v, "Swap activity").status is Status.UNKNOWN

    def test_partial_missing_is_unknown(self):
        v = evaluate_reliability(swap_in_pages=0, swap_out_pages=None)
        assert _criterion(v, "Swap activity").status is Status.UNKNOWN


class TestThermal:
    def test_zero_passes(self):
        assert _criterion(evaluate_reliability(thermal_throttle_events=0), "Thermal throttling").status is Status.PASS

    def test_nonzero_fails(self):
        v = evaluate_reliability(thermal_throttle_events=3)
        assert _criterion(v, "Thermal throttling").status is Status.FAIL
        assert v.reliable is False

    def test_missing_is_unknown(self):
        assert _criterion(evaluate_reliability(thermal_throttle_events=None), "Thermal throttling").status is Status.UNKNOWN


class TestContextSwitches:
    def test_warns_without_baseline(self):
        v = evaluate_reliability(involuntary_ctx_switches=50_000, total_cpu_s=100.0)
        c = _criterion(v, "Involuntary context switches")
        assert c.status is Status.WARN
        assert c.hard is False

    def test_within_baseline_passes(self):
        # 2000/100 = 20/CPU-s, baseline 5/CPU-s, limit 50 -> pass.
        v = evaluate_reliability(
            involuntary_ctx_switches=2_000, total_cpu_s=100.0,
            ctx_switch_baseline_per_cpu_s=5.0,
        )
        assert _criterion(v, "Involuntary context switches").status is Status.PASS

    def test_above_baseline_warns_but_does_not_reject(self):
        # 100000/100 = 1000/CPU-s, baseline 5 -> limit 50, exceeded.
        v = evaluate_reliability(
            cpu_efficiency=0.99,
            involuntary_ctx_switches=100_000, total_cpu_s=100.0,
            ctx_switch_baseline_per_cpu_s=5.0,
        )
        c = _criterion(v, "Involuntary context switches")
        assert c.status is Status.WARN
        # Advisory only: a clean run is still reliable despite the warning.
        assert v.reliable is True
        assert c in v.warnings


class TestRam:
    def test_below_threshold_passes(self):
        assert _criterion(evaluate_reliability(ram_used_fraction=0.5), "RAM utilisation").status is Status.PASS

    def test_above_threshold_warns_only(self):
        v = evaluate_reliability(cpu_efficiency=0.99, ram_used_fraction=0.95)
        c = _criterion(v, "RAM utilisation")
        assert c.status is Status.WARN
        assert v.reliable is True  # warning never rejects


class TestOverallVerdict:
    def test_all_clean_is_reliable(self):
        v = evaluate_reliability(
            cpu_efficiency=0.99,
            load_avg_1m_pre=1.0, physical_cores=64,
            swap_in_pages=0, swap_out_pages=0,
            thermal_throttle_events=0,
            ram_used_fraction=0.4,
        )
        assert v.reliable is True
        assert not v.failures

    def test_no_data_is_unknown(self):
        v = evaluate_reliability()
        assert v.reliable is None

    def test_single_hard_failure_rejects(self):
        v = evaluate_reliability(
            cpu_efficiency=0.99,
            load_avg_1m_pre=1.0, physical_cores=64,
            swap_in_pages=512, swap_out_pages=0,  # the one bad signal
            thermal_throttle_events=0,
        )
        assert v.reliable is False
        assert [c.name for c in v.failures] == ["Swap activity"]

    def test_efficiency_constant_matches_module(self):
        assert MIN_CPU_EFFICIENCY == 0.95
