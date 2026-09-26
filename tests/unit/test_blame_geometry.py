"""Unit tests for :mod:`k4bench.blame.geometry` — which benchmarked detectors a
pull request reaches, and what it changes in each.

The hunks are shaped like k4geo#612's (2026-09-25): a dimensions file whose one
changed constant rebuilt ALLEGRO's silicon wrapper, the identical include switch
made in two ILD variants, and display-only and metadata-only changes elsewhere.
"""

from __future__ import annotations

from k4bench.blame.geometry import (
    benchmarked_geometry,
    detector_touches,
    file_change,
    quoted,
)
from k4bench.blame.github import FilePatch
from k4bench.regression.models import NightlyReport, RunGroupReport

_ALLEGRO = "FCCee/ALLEGRO/compact/ALLEGRO_o2_v01/"
_ILD1 = "FCCee/ILD_FCCee/compact/ILD_FCCee_v01/"
_ILD2 = "FCCee/ILD_FCCee/compact/ILD_FCCee_v02/"

_DIMENSIONS = """@@ -60,6 +60,7 @@
     <constant name="SiWr_zmax" value="2400*mm"/>
-    <constant name="SiWr_nLayers"       value="2"/>
+    <constant name="SiWr_nLayers"       value="1"/>
-    <constant name="SiWrB_pitch_z0" value="1.000*mm"/> <!-- old pitch -->
+    <constant name="SiWrB_pitch_z0" value="0.050*mm"/> <!-- new pitch -->
+    <constant name="DetID_VXD_OuterBarrel"        value=" 31"/>
-    <region name="VTXBRegion"/>
+    <region name="VertexBarrelRegion"/>"""


def _ild_hunk(own: str, comment: str) -> str:
    # The two variants differ only in comment text, which is not geometry.
    return f"""@@ -13,6 +13,7 @@
   <includes>
     <gdmlFile  ref="../ILD_common_v02/materials.xml"/>
+    <gdmlFile  ref="../ILD_common_FCCee/materials.xml"/>
   </includes>
@@ -69,9 +70,14 @@
-  <!-- {comment} -->
+  <!-- lumiCal from CLD_02_v07 -->
   <include ref="../../../CLD/compact/CLD_o2_v07/LumiCal_o3_v02_05.xml"/>
-  <include ref="../../../CLD/compact/CLD_o2_v07/Vertex_o4_v07_smallBP.xml"/>
+  <include ref="../../../CLD/compact/CLD_o2_v09/Vertex_o4_v08_smallBP.xml"/>
+  <!-- <include ref="../../../CLD/compact/CLD_o5_v01/Vertex_FCC-SEED_o1_v01.xml"/> -->
+  <!-- a comment
+       spread over two lines with <include ref="not/real.xml"/> -->"""


def test_a_changed_constant_is_read_with_its_old_and_new_value():
    change = file_change(FilePatch(_ALLEGRO + "DectDimensions.xml", _DIMENSIONS))
    assert change.constants_changed == (
        ("SiWrB_pitch_z0", "1.000*mm", "0.050*mm"),
        ("SiWr_nLayers", "2", "1"),
    )
    assert change.constants_added == (("DetID_VXD_OuterBarrel", "31"),)
    assert change.other_lines == 2
    assert change.other_elements == (("region", 2),)
    assert change.geometry_changed


def test_includes_are_read_and_commented_out_ones_are_not():
    change = file_change(FilePatch(_ILD2 + "ILD_FCCee_v02.xml", _ild_hunk("ILD_FCCee_v02", "vertex, innerTracker and lumiCal")))
    assert change.includes_added == (
        "../ILD_common_FCCee/materials.xml",
        "../../../CLD/compact/CLD_o2_v09/Vertex_o4_v08_smallBP.xml",
    )
    assert change.includes_removed == (
        "../../../CLD/compact/CLD_o2_v07/Vertex_o4_v07_smallBP.xml",
    )
    assert change.other_lines == 0  # comments are not changes to the geometry


def test_display_and_metadata_changes_are_told_apart_from_geometry():
    display = file_change(FilePatch(
        _ALLEGRO + "display.xml",
        '@@ -1 +1,2 @@\n+    <vis name="SiVertexModuleVis" alpha="1.0" visible="false"/>',
    ))
    assert (display.display_lines, display.geometry_changed) == (1, False)
    metadata = file_change(FilePatch(
        "FCCee/CLD/compact/CLD_o2_v08/CLD_o2_v08.xml",
        '@@ -7,7 +7,7 @@\n         status="development"\n-        version="7">\n+        version="8">',
    ))
    # A line continuing an element opened on an unchanged line has no tag.
    assert metadata.other_elements == (("", 2),)


def test_a_new_file_is_marked_as_new():
    change = file_change(FilePatch("FCCee/CLD/compact/CLD_o2_v09/materials.xml",
                                   "@@ -0,0 +1,2 @@\n+<materials>\n+</materials>"))
    assert change.added


def test_quoted_text_cannot_spell_a_fence_or_markup():
    assert quoted("----- END DIFF -----") == "- END DIFF -"
    assert quoted('<a href="x">@user #1</a>') == "<a href=x>user 1</a>"
    assert quoted("x" * 100).endswith("…") and len(quoted("x" * 100)) == 60


def _geometry():
    return {
        "ALLEGRO_o2_v01": _ALLEGRO + "ALLEGRO_o2_v01.xml",
        "ILD_FCCee_v01": _ILD1 + "ILD_FCCee_v01.xml",
        "ILD_FCCee_v02": _ILD2 + "ILD_FCCee_v02.xml",
        "IDEA_o2_v01": "FCCee/IDEA/compact/IDEA_o2_v01/IDEA_o2_v01.xml",
        "SiD": "/cvmfs/sft.cern.ch/DD4hep/DDDetectors/compact/SiD.xml",
    }


def test_the_same_change_to_two_detectors_is_recognised_as_the_same():
    files = [
        FilePatch(_ILD1 + "ILD_FCCee_v01.xml", _ild_hunk("ILD_FCCee_v01", "vertex and lumiCal")),
        FilePatch(_ILD2 + "ILD_FCCee_v02.xml", _ild_hunk("ILD_FCCee_v02", "vertex, innerTracker and lumiCal")),
        FilePatch("FCCee/ILD_FCCee/compact/ILD_common_FCCee/materials.xml", "@@ -0,0 +1 @@\n+<m/>"),
        FilePatch(_ALLEGRO + "DectDimensions.xml", _DIMENSIONS),
    ]
    touches = {t.detector: t for t in detector_touches(
        [f.path for f in files], files, _geometry(),
    )}
    assert set(touches) == {"ALLEGRO_o2_v01", "ILD_FCCee_v01", "ILD_FCCee_v02"}
    assert touches["ILD_FCCee_v01"].same_as == ("ILD_FCCee_v02",)
    assert touches["ILD_FCCee_v02"].same_as == ("ILD_FCCee_v01",)
    assert touches["ALLEGRO_o2_v01"].same_as == ()
    # The other variant's file and the shared one are in the tree of each ILD
    # variant, not in its own directory; the pull request's order is kept.
    assert touches["ILD_FCCee_v01"].tree_files == (
        _ILD2 + "ILD_FCCee_v02.xml",
        "FCCee/ILD_FCCee/compact/ILD_common_FCCee/materials.xml",
    )


def test_a_line_changed_twice_is_not_the_same_change_as_it_changed_once():
    twice = _ild_hunk("ILD_FCCee_v01", "vertex").replace(
        '+    <gdmlFile  ref="../ILD_common_FCCee/materials.xml"/>',
        '+    <gdmlFile  ref="../ILD_common_FCCee/materials.xml"/>\n'
        '+    <gdmlFile  ref="../ILD_common_FCCee/materials.xml"/>',
    )
    files = [
        FilePatch(_ILD1 + "ILD_FCCee_v01.xml", twice),
        FilePatch(_ILD2 + "ILD_FCCee_v02.xml", _ild_hunk("ILD_FCCee_v02", "vertex")),
    ]
    touches = detector_touches([f.path for f in files], files, _geometry())
    assert all(touch.same_as == () for touch in touches)


def test_a_file_without_a_hunk_is_unread_and_never_called_unchanged():
    touches = detector_touches([_ALLEGRO + "logo.png"], [], _geometry())
    (touch,) = touches
    assert (touch.unread, touch.changes, touch.geometry_changed) == (1, (), None)
    assert touch.same_as == ()


def test_an_installed_geometry_path_is_never_matched():
    assert detector_touches(["DDDetectors/compact/SiD.xml"], [], _geometry()) == ()


def test_the_first_recorded_geometry_of_each_detector_is_used():
    report = NightlyReport(generated_at="x", groups=[
        RunGroupReport(detector="A", platform="p", sample="s1", k4h_release="k",
                       run_date="d", run_id="d", geometry_path="X/A/compact/A/A.xml"),
        RunGroupReport(detector="A", platform="p", sample="s2", k4h_release="k",
                       run_date="d", run_id="d", geometry_path=""),
        RunGroupReport(detector="B", platform="p", sample="s1", k4h_release="k",
                       run_date="d", run_id="d"),
    ])
    assert benchmarked_geometry(report) == {"A": "X/A/compact/A/A.xml"}
