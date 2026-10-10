# The k4BenchAuditor: a Gaudi auditor for k4run jobs.
#
# A project of its own, separate from the DDG4 plugins one directory up:
# configuring Gaudi and DD4hep in one project re-imports shared dependencies and
# fails when the build directory is reconfigured.

cmake_minimum_required(VERSION 3.19)
project(k4BenchGaudiPlugins VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Gaudi REQUIRED)
find_package(nlohmann_json REQUIRED)

gaudi_add_module(
  k4BenchAuditor
  SOURCES k4BenchAuditor.cpp
  LINK Gaudi::GaudiKernel nlohmann_json::nlohmann_json
)
# k4BenchProcStats.h is shared with the DDG4 plugins.
target_include_directories(k4BenchAuditor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Preloaded into an audited job so the auditor can count its heap allocations.
# Installed next to the auditor; no Gaudi, and the auditor does not link it.
# It calls glibc only, so --as-needed leaves libc its sole dependency and it
# preloads into any process of the job, whatever the C++ runtime on its path.
add_library(k4BenchAllocCounter SHARED k4BenchAllocCounter.cpp)
target_link_options(k4BenchAllocCounter PRIVATE LINKER:--as-needed)
install(TARGETS k4BenchAllocCounter LIBRARY DESTINATION "${GAUDI_INSTALL_PLUGINDIR}")
