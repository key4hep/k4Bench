"""Where a reconstruction component's code comes from: the stack package whose
install holds its class's library, and links to the class and to the
package's repository at the revision the run's stack built. Shared by the
sections that list components by package."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import cached_load_run_packages
from k4bench.analysis.components import ComponentTiming
from k4bench.provenance.sources import package_of

#: How the tables write and explain :func:`source_columns`.
SOURCE_COLUMN_CONFIG = {
    "Package": st.column_config.TextColumn(
        "Package", help="The stack package whose install holds the class's "
                        "library, at its built commit or release version.",
    ),
    "Code": st.column_config.LinkColumn(
        "Code", display_text="↗ find class", width="small",
        help="Where the class is defined: a GitHub code search on the default "
             "branch (needs a signed-in GitHub user), or a GitLab search at the "
             "built commit.",
    ),
    "Repository": st.column_config.LinkColumn(
        "Repository", display_text="↗ at revision", width="small",
        help="The package's repository at the commit this run's stack built. "
             "Absent for a package built from a release whose repository is "
             "unknown (Gaudi's is known).",
    ),
}


def package_labels(
    data_dir: str, timing: ComponentTiming, names, *, unknown: str = "Unknown",
) -> pd.Series:
    """The stack package each of *names* comes from, by its class's library;
    *unknown* where no package holds it."""
    packages = cached_load_run_packages(data_dir)
    libraries = timing.components["impl_library"]
    return pd.Series(
        [getattr(package_of(libraries.get(name, ""), packages), "label", unknown) for name in names],
        index=names, dtype=object,
    )


def source_columns(data_dir: str, timing: ComponentTiming, names) -> pd.DataFrame:
    """The ``Package``, ``Code`` and ``Repository`` of each of *names* (as
    :data:`SOURCE_COLUMN_CONFIG` writes them), indexed by name: "—" and no
    links for one no package holds, or that the run did not record."""
    packages = cached_load_run_packages(data_dir)
    rows = {}
    for name in names:
        source = (
            package_of(timing.components.at[name, "impl_library"], packages)
            if name in timing.components.index else None
        )
        impl = timing.components.at[name, "impl"] if source else None
        rows[name] = {
            "Package": source.label if source else "—",
            "Code": source.search_url(impl) if source else None,
            "Repository": source.tree_url if source else None,
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(SOURCE_COLUMN_CONFIG))
