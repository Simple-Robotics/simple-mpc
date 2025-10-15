find_package(benchmark REQUIRED)
find_package(example-robot-data 4.0.9 REQUIRED)

# Add a benchmark.
function(create_bench benchfile)
  get_filename_component(benchname ${benchfile} NAME_WE)
  set(benchname "bench-${benchname}")
  add_executable(${benchname} ${benchfile})
  message(STATUS "Adding cpp example ${benchname}")
  set_target_properties(${benchname} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${benchname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${benchname} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${benchname} PUBLIC example-robot-data::example-robot-data)
  target_link_libraries(${benchname} PUBLIC benchmark::benchmark)

  # Add custom command to run this particular benchmark
  add_custom_target(run-${benchname}
      COMMAND ${benchname}
      DEPENDS ${benchname}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # Add this benchmark in the dependencies to run all benchmarks
  add_dependencies(run-benchmarks run-${benchname})
endfunction()

# Create a custom target to run all benchmarks
add_custom_target(run-benchmarks
    COMMAND ${CMAKE_COMMAND} -E echo "All benchmarks run."
)

# Add benchmarks
create_bench("talos.cpp")
create_bench("go2.cpp")
