include("${JRL_CMAKE_MODULES}/python.cmake")
include("${JRL_CMAKE_MODULES}/stubs.cmake")

ADD_PROJECT_DEPENDENCY(eigenpy 3.9.1 REQUIRED)

set(
  ${PY_NAME}_SOURCES
  module.cpp
  expose-robot-handler.cpp
  expose-problem.cpp
  expose-mpc.cpp
  expose-lowlevel.cpp
  expose-centroidal.cpp
  expose-fulldynamics.cpp
  expose-kinodynamics.cpp
  expose-friction-compensation.cpp
)

Python3_add_library(${PY_NAME}_pywrap MODULE WITH_SOABI ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
set_target_properties(
  ${PY_NAME}_pywrap
  PROPERTIES
    INSTALL_RPATH "\$ORIGIN/../../.."
    LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
)
install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON __init__.py)

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()

if(GENERATE_PYTHON_STUBS)
  LOAD_STUBGEN()
  GENERATE_STUBS(
    ${CMAKE_CURRENT_BINARY_DIR}
    simple_mpc
    ${PYTHON_SITELIB}
    ${PY_NAME}_pywrap
  )
endif()
