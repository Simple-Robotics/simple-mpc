find_package(example-robot-data 4.0.9 REQUIRED)
ADD_PROJECT_DEPENDENCY(OpenMP REQUIRED)
ADD_PROJECT_DEPENDENCY(proxsuite REQUIRED)
ADD_PROJECT_DEPENDENCY(pinocchio REQUIRED)
ADD_PROJECT_DEPENDENCY(aligator 0.13.0 REQUIRED)

function(get_ndcurves)
  find_package(ndcurves QUIET)
  if(NOT ndcurves_FOUND)
    FetchContent_Declare(
      ndcurves
      GIT_REPOSITORY "https://github.com/loco-3d/ndcurves"
      GIT_PROGRESS True
      GIT_TAG devel
      SYSTEM
      EXCLUDE_FROM_ALL
    )
    set(PROJECT_CUSTOM_HEADER_DIR)
    set(PROJECT_CUSTOM_HEADER_EXTENSION)
    set(BUILD_PYTHON_INTERFACE OFF)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(ndcurves)
    add_library(ndcurves::ndcurves ALIAS ndcurves)
    install(TARGETS ndcurves EXPORT ${TARGETS_EXPORT_NAME})
  endif()
endfunction()

get_ndcurves()
