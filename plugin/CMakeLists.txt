cmake_minimum_required(VERSION 3.14)
project(k4BenchPlugin VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DD4hep REQUIRED COMPONENTS DDG4 DDCore)
dd4hep_set_compiler_flags()

# -----------------------------------------------------------------------------
# Per-event wall time + RSS plugin
# -----------------------------------------------------------------------------

add_dd4hep_plugin(
  k4BenchTimingAction
  SHARED
  k4BenchTimingAction.cpp
)

target_link_libraries(
  k4BenchTimingAction
  PUBLIC
    DD4hep::DDCore
    DD4hep::DDG4
)

# -----------------------------------------------------------------------------
# Per-region timing plugin (stepping + tracking + event actions in one .so)
# -----------------------------------------------------------------------------

add_dd4hep_plugin(
  k4BenchRegionTimingAction
  SHARED
  k4BenchRegionTimingAction.cpp
)

target_link_libraries(
  k4BenchRegionTimingAction
  PUBLIC
    DD4hep::DDCore
    DD4hep::DDG4
)

# rdtscp lives in <x86intrin.h>; on x86_64 GCC/Clang it is available without
# extra flags, but be explicit so a non-x86 build cleanly falls back to the
# steady_clock path inside the source.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_definitions(k4BenchRegionTimingAction PRIVATE K4BENCH_X86_64=1)
endif()

# -----------------------------------------------------------------------------
# Package + install
# -----------------------------------------------------------------------------

dd4hep_instantiate_package(k4BenchTimingAction)

# ${CMAKE_INSTALL_LIBDIR} is "lib" on Debian/Ubuntu and "lib64" on RHEL/Fedora
# (set by GNUInstallDirs based on the platform multiarch convention).
# k4bench/plugin/runtime.py::find_plugin_lib_dir() searches both
# install/lib and install/lib64, so either layout is handled automatically.
install(
  TARGETS
    k4BenchTimingAction
    k4BenchRegionTimingAction
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
