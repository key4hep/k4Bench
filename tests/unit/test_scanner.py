"""Unit tests for dd4bench.geometry.scanner.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from dd4bench.geometry.scanner import (
    _detector_names_in_file,
    get_detector_names,
    resolve_includes,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures" / "minimal_geometry"
MINIMAL_XML = FIXTURES / "minimal.xml"
TRACKER_XML = FIXTURES / "subdetectors" / "tracker.xml"
CALORIMETER_XML = FIXTURES / "subdetectors" / "calorimeter.xml"
MATERIALS_XML = FIXTURES / "materials.xml"


# ---------------------------------------------------------------------------
# resolve_includes
# ---------------------------------------------------------------------------


class TestResolveIncludes:
    def test_top_level_file_is_first(self):
        files = resolve_includes(MINIMAL_XML)
        assert files[0] == MINIMAL_XML.resolve()

    def test_all_included_files_present(self):
        files = resolve_includes(MINIMAL_XML)
        resolved = {f.resolve() for f in files}
        assert MATERIALS_XML.resolve() in resolved
        assert TRACKER_XML.resolve() in resolved
        assert CALORIMETER_XML.resolve() in resolved

    def test_no_duplicates(self):
        files = resolve_includes(MINIMAL_XML)
        assert len(files) == len(set(files))

    def test_returns_only_paths(self):
        files = resolve_includes(MINIMAL_XML)
        assert all(isinstance(f, Path) for f in files)

    def test_leaf_file_returns_only_itself(self):
        # tracker.xml has no includes of its own
        files = resolve_includes(TRACKER_XML)
        assert files == [TRACKER_XML.resolve()]

    def test_missing_include_warns(self, tmp_path):
        xml = tmp_path / "missing_include.xml"
        xml.write_text(
            '<?xml version="1.0"?>'
            '<lccdd><include ref="does_not_exist.xml"/></lccdd>'
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            files = resolve_includes(xml)

        assert any("not found" in str(w.message) for w in caught)
        # top-level file is still returned
        assert xml.resolve() in files

    def test_unparseable_file_warns(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("this is not xml <<<<")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            files = resolve_includes(bad)

        assert any("Could not parse" in str(w.message) for w in caught)
        assert bad.resolve() in files

    def test_cycle_does_not_loop(self, tmp_path):
        # a.xml includes b.xml, b.xml includes a.xml
        a = tmp_path / "a.xml"
        b = tmp_path / "b.xml"
        a.write_text(
            '<?xml version="1.0"?>'
            '<lccdd><include ref="b.xml"/></lccdd>'
        )
        b.write_text(
            '<?xml version="1.0"?>'
            '<lccdd><include ref="a.xml"/></lccdd>'
        )
        files = resolve_includes(a)
        assert files.count(a.resolve()) == 1
        assert files.count(b.resolve()) == 1

    def test_unresolved_env_var_is_skipped(self, tmp_path):
        xml = tmp_path / "env_include.xml"
        xml.write_text(
            '<?xml version="1.0"?>'
            '<lccdd><include ref="${DD4hepINSTALL}/some/file.xml"/></lccdd>'
        )
        # Must not raise, must not warn about missing file
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            files = resolve_includes(xml)

        assert xml.resolve() in files
        assert not any("not found" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# _detector_names_in_file
# ---------------------------------------------------------------------------


class TestDetectorNamesInFile:
    def test_tracker_has_two_detectors(self):
        names = _detector_names_in_file(TRACKER_XML)
        assert names == ["InnerTracker", "OuterTracker"]

    def test_calorimeter_has_two_detectors(self):
        names = _detector_names_in_file(CALORIMETER_XML)
        assert names == ["EcalBarrel", "HcalBarrel"]

    def test_materials_has_no_detectors(self):
        names = _detector_names_in_file(MATERIALS_XML)
        assert names == []

    def test_bad_xml_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("not xml at all <<<")
        assert _detector_names_in_file(bad) == []

    def test_detector_without_name_is_skipped(self, tmp_path):
        xml = tmp_path / "nameless.xml"
        xml.write_text(
            '<?xml version="1.0"?>'
            '<lccdd><detectors>'
            '<detector type="Foo"/>'          # no name attribute
            '<detector name="Named" type="Bar"/>'
            '</detectors></lccdd>'
        )
        assert _detector_names_in_file(xml) == ["Named"]


# ---------------------------------------------------------------------------
# get_detector_names — integration across full fixture tree
# ---------------------------------------------------------------------------


class TestGetDetectorNames:
    @pytest.fixture(scope="class")
    def names(self):
        return get_detector_names(MINIMAL_XML)

    def test_returns_all_four_detectors(self, names):
        assert set(names) == {"InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel"}

    def test_count_is_four(self, names):
        assert len(names) == 4

    def test_no_duplicates(self, names):
        assert len(names) == len(set(names))

    def test_tracker_before_calorimeter(self, names):
        # minimal.xml includes tracker.xml before calorimeter.xml
        assert names.index("InnerTracker") < names.index("EcalBarrel")

    def test_returns_list_of_strings(self, names):
        assert all(isinstance(n, str) for n in names)

    def test_empty_geometry_returns_empty_list(self, tmp_path):
        xml = tmp_path / "empty.xml"
        xml.write_text('<?xml version="1.0"?><lccdd></lccdd>')
        assert get_detector_names(xml) == []
