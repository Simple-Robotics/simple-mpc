#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(example-robot-data 4.0.9 REQUIRED)

macro(ADD_TEST_CFLAGS test_name flag)
  set_property(
    TARGET ${test_name}
    APPEND_STRING
    PROPERTY COMPILE_FLAGS " ${flag}"
  )
endmacro()

file(GLOB TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
message(STATUS "Test headers: ${TEST_HEADERS}")

function(_add_test_prototype filename prefix dependencies)
  string(REGEX REPLACE "/" "_" name ${filename})
  set(test_name "test-cpp-${prefix}${name}")
  set(test_file ${filename}.cpp)
  message(STATUS "Adding C++ test: ${test_file} (${test_name})")

  ADD_UNIT_TEST(${test_name} ${test_file} ${TEST_HEADERS})
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  set(MODULE_NAME "${name}Test")
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})

  ADD_TEST_CFLAGS(${test_name} "-DBOOST_TEST_DYN_LINK")
  ADD_TEST_CFLAGS(${test_name} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  target_link_libraries(${test_name} PUBLIC ${dependencies})
  target_link_libraries(${test_name} PUBLIC example-robot-data::example-robot-data)
  target_link_libraries(${test_name} PRIVATE Boost::unit_test_framework)
endfunction()

function(add_aligator_test filename)
  _add_test_prototype(${filename} "" ${PROJECT_NAME})
endfunction()

set(
  TEST_NAMES
  robot_handler
  problem
  inverse-dynamics/kinodynamics-id
  inverse-dynamics/centroidal-id
  #mpc
  friction
  interpolator
)

foreach(test_name ${TEST_NAMES})
  add_aligator_test(${test_name})
endforeach()
