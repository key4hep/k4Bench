cmake_minimum_required(VERSION 3.14)
project(DD4benchPlugin VERSION 0.1.0 LANGUAGES CXX)
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
  DD4benchTimingAction
  SHARED
  DD4benchTimingAction.cpp
)

target_link_libraries(
  DD4benchTimingAction
  PUBLIC
    DD4hep::DDCore
    DD4hep::DDG4
)

# -----------------------------------------------------------------------------
# Per-region timing plugin (stepping + tracking + event actions in one .so)
# -----------------------------------------------------------------------------

add_dd4hep_plugin(
  DD4benchRegionTimingAction
  SHARED
  DD4benchRegionTimingAction.cpp
)

target_link_libraries(
  DD4benchRegionTimingAction
  PUBLIC
    DD4hep::DDCore
    DD4hep::DDG4
)

# rdtscp lives in <x86intrin.h>; on x86_64 GCC/Clang it is available without
# extra flags, but be explicit so a non-x86 build cleanly falls back to the
# steady_clock path inside the source.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_definitions(DD4benchRegionTimingAction PRIVATE DD4BENCH_X86_64=1)
endif()

# -----------------------------------------------------------------------------
# Package + install
# -----------------------------------------------------------------------------

dd4hep_instantiate_package(DD4benchTimingAction)

# ${CMAKE_INSTALL_LIBDIR} is "lib" on Debian/Ubuntu and "lib64" on RHEL/Fedora
# (set by GNUInstallDirs based on the platform multiarch convention).
# dd4bench/plugin/runtime.py::find_plugin_lib_dir() searches both
# install/lib and install/lib64, so either layout is handled automatically.
install(
  TARGETS
    DD4benchTimingAction
    DD4benchRegionTimingAction
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
