"""Unit tests for the event JSON version contract (:mod:`k4bench.plugin.event_schema`)."""

from __future__ import annotations

import pytest

from k4bench.plugin.event_schema import EVENT_SCHEMA_VERSION, validate_event_schema


def test_current_version_is_one():
    assert EVENT_SCHEMA_VERSION == 1


@pytest.mark.parametrize("raw", [{}, {"event_numbers": []}, {"schema_version": 1}])
def test_legacy_unversioned_and_current_files_are_accepted(raw):
    validate_event_schema(raw)


@pytest.mark.parametrize(
    "version",
    # bool is an int subclass in Python, so True would slip past isinstance.
    ["1", True, False, 1.0, None, [1], {"v": 1}, 0, -1],
)
def test_malformed_versions_are_refused(version):
    with pytest.raises(ValueError, match="malformed schema_version"):
        validate_event_schema({"schema_version": version})


def test_future_version_is_refused_rather_than_misread():
    with pytest.raises(ValueError, match="unsupported future schema_version 2"):
        validate_event_schema({"schema_version": EVENT_SCHEMA_VERSION + 1})


def test_source_names_the_file():
    with pytest.raises(ValueError, match="^/runs/x_events.json: "):
        validate_event_schema({"schema_version": 99}, source="/runs/x_events.json")
