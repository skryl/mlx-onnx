find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.10.2
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(
  _core
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_SOURCE_DIR}/module.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/onnx.cpp)

target_include_directories(
  _core
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(_core PRIVATE mlx_onnx)

install(TARGETS _core LIBRARY DESTINATION mlx_onnx COMPONENT python)
