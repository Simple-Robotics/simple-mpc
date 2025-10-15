find_package(benchmark REQUIRED)
find_package(example-robot-data 4.0.9 REQUIRED)

# Add a benchmark.
function(create_bench exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "bench-${exname}")
  add_executable(${exname} ${exfile})
  message(STATUS "Adding cpp example ${exname}")
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  set_standard_output_directory(${exname})
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${exname} PUBLIC example-robot-data::example-robot-data)
  target_link_libraries(${exname} PUBLIC benchmark::benchmark)
endfunction()

create_bench("talos.cpp")
create_bench("go2.cpp")
