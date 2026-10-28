"""Nightly performance-regression detection over the benchmark trend history.

Layout mirrors :mod:`k4bench.results`:

- :mod:`k4bench.regression.models` — verdict/report dataclasses and enums.
- :mod:`k4bench.regression.engine` — the pure statistical detector.
- :mod:`k4bench.regression.report_builder` — walks one suite's EOS run tree
  and assembles a :class:`~k4bench.regression.models.NightlyReport` covering
  every detector in it.
- :mod:`k4bench.regression.render` — markdown / HTML / JSON rendering.
- :mod:`k4bench.regression.notify` — e-group email delivery.
"""
