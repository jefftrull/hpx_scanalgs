# CMake build control file for HPX exclusive/inclusive scan experiments

cmake_minimum_required( VERSION 3.12 )

project( hpx_scanalgs )

# create a compilation database for clangd etc.
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set( CMAKE_CXX_STANDARD 17 )

find_package( Boost 1.69 REQUIRED COMPONENTS system program_options filesystem )

# HPX provides an HPXConfig.cmake so as long as it's in HPX_DIR we can do this:

find_package( HPX REQUIRED NO_MODULE )

# Release build by default
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Default build type (Debug, Release, RelWithDebInfo)" FORCE )
endif()

# download and use Google Benchmark
include( ExternalProject )
ExternalProject_Add( GBENCH
  GIT_REPOSITORY https://github.com/google/benchmark
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gbench
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property( GBENCH SOURCE_DIR BINARY_DIR )
add_library( gbench STATIC IMPORTED )

# create include directory, which will be absent at initial run
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${SOURCE_DIR}/include )
set_target_properties( gbench PROPERTIES
  IMPORTED_LOCATION "${BINARY_DIR}/src/libbenchmark.a"
  INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR}/include"
  INTERFACE_LINK_LIBRARIES pthread
)
add_dependencies( gbench GBENCH )

add_hpx_executable( exsvp SOURCES exscan_seqvspar.cpp COMPONENT_DEPENDENCIES iostreams )
include(CheckIPOSupported)
check_ipo_supported( RESULT IPO_ENABLED )
if( IPO_ENABLED )
set_target_properties( exsvp PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION ON
)
endif()

add_executable( ba basic_algs.cpp )
target_link_libraries( ba gbench )

add_executable( mp manual_parallel.cpp )
find_package( Threads )
hpx_setup_target( mp )
target_link_libraries( mp PRIVATE Threads::Threads gbench )

hpx_setup_target( exsvp DEPENDENCIES gbench Boost::system Boost::program_options )

if( CMAKE_SYSTEM_NAME STREQUAL Linux )
  # tracepoint support
  add_library( tp tracepoints.cpp )
  target_include_directories( tp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )
  target_link_libraries( tp PUBLIC lttng-ust dl )

  target_link_libraries( exsvp PRIVATE dl tp )
  target_link_libraries( mp PRIVATE tp )

endif()

