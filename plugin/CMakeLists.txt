cmake_minimum_required(VERSION 3.14)
project(DD4benchPlugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DD4hep REQUIRED COMPONENTS DDG4 DDCore)

dd4hep_set_compiler_flags()

add_dd4hep_plugin(DD4benchTimingAction SHARED DD4benchTimingAction.cpp)

target_link_libraries(DD4benchTimingAction
  PUBLIC DD4hep::DDCore DD4hep::DDG4)

dd4hep_instantiate_package(DD4benchTimingAction)

install(TARGETS DD4benchTimingAction LIBRARY DESTINATION lib)
