cmake_minimum_required(VERSION 3.25)

project(mlx_onnx VERSION 0.30.7.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MLX_ONNX_USE_EXTERNAL_MLX "Build against an externally provided MLX install" OFF)
option(MLX_ONNX_BUILD_PYTHON_BINDINGS "Build Python IR bindings" OFF)
option(MLX_ONNX_INSTALL_CPP_ARTIFACTS "Install C++ library and headers" ON)

include(FetchContent)

if(MLX_ONNX_USE_EXTERNAL_MLX AND MLX_ONNX_BUILD_PYTHON_BINDINGS)
  message(
    FATAL_ERROR
      "MLX_ONNX_BUILD_PYTHON_BINDINGS requires bundled mlx sources; set MLX_ONNX_USE_EXTERNAL_MLX=OFF")
endif()

if(MLX_ONNX_USE_EXTERNAL_MLX)
  set(MLX_ONNX_EXTERNAL_MLX_INCLUDE_DIR "" CACHE PATH "Path to MLX include root")
  set(MLX_ONNX_EXTERNAL_MLX_LIB_DIR "" CACHE PATH "Path to MLX library directory")

  if(MLX_ONNX_EXTERNAL_MLX_INCLUDE_DIR STREQUAL "")
    message(FATAL_ERROR "MLX_ONNX_EXTERNAL_MLX_INCLUDE_DIR must be set when MLX_ONNX_USE_EXTERNAL_MLX=ON")
  endif()
  if(MLX_ONNX_EXTERNAL_MLX_LIB_DIR STREQUAL "")
    message(FATAL_ERROR "MLX_ONNX_EXTERNAL_MLX_LIB_DIR must be set when MLX_ONNX_USE_EXTERNAL_MLX=ON")
  endif()

  find_library(
    MLX_EXTERNAL_LIBRARY
    NAMES mlx
    PATHS ${MLX_ONNX_EXTERNAL_MLX_LIB_DIR}
    NO_DEFAULT_PATH)

  if(NOT MLX_EXTERNAL_LIBRARY)
    message(FATAL_ERROR "Could not find libmlx in ${MLX_ONNX_EXTERNAL_MLX_LIB_DIR}")
  endif()

  add_library(mlx SHARED IMPORTED GLOBAL)
  set_target_properties(
    mlx
    PROPERTIES
      IMPORTED_LOCATION ${MLX_EXTERNAL_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${MLX_ONNX_EXTERNAL_MLX_INCLUDE_DIR})
else()
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  if(MLX_ONNX_BUILD_PYTHON_BINDINGS)
    set(MLX_BUILD_PYTHON_BINDINGS ON CACHE BOOL "" FORCE)
  else()
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  endif()
  set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_SAFETENSORS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mlx)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(
  mlx_onnx
  src/export.cpp
  src/api.cpp
  src/compat.cpp
  src/io.cpp
  src/lowering.cpp
  src/mappings.cpp
  src/onnx.cpp
  src/shared.cpp)

set_target_properties(mlx_onnx PROPERTIES OUTPUT_NAME mlx_onnx)

target_include_directories(
  mlx_onnx
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MLX_ONNX_USE_EXTERNAL_MLX)
  target_include_directories(mlx_onnx PRIVATE ${MLX_ONNX_EXTERNAL_MLX_INCLUDE_DIR})
endif()

target_link_libraries(mlx_onnx PUBLIC mlx nlohmann_json::nlohmann_json)

if(MLX_ONNX_BUILD_PYTHON_BINDINGS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  set(MLX_ONNX_PY_INIT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/python/mlx_onnx/__init__.py)
  if(NOT EXISTS ${MLX_ONNX_PY_INIT_FILE})
    set(MLX_ONNX_PY_INIT_FILE ${CMAKE_CURRENT_BINARY_DIR}/mlx_onnx___init__.py)
    file(WRITE ${MLX_ONNX_PY_INIT_FILE} "from ._core import *  # noqa: F401,F403\n")
  endif()
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mlx/CMakeLists.txt)
    message(FATAL_ERROR "Bundled mlx sources are missing at ${CMAKE_CURRENT_SOURCE_DIR}/mlx")
  endif()
  if(NOT TARGET core)
    message(FATAL_ERROR "Bundled mlx Python extension target `core` was not built")
  endif()
  install(TARGETS core LIBRARY DESTINATION mlx COMPONENT python)
  if(APPLE AND MLX_BUILD_METAL)
    # MLX looks for mlx.metallib next to the extension module using MLX runtime.
    install(
      FILES ${CMAKE_CURRENT_BINARY_DIR}/mlx/mlx/backend/metal/kernels/mlx.metallib
      DESTINATION mlx
      COMPONENT python)
    # mlx_onnx._core also links MLX and resolves the same metallib at runtime.
    install(
      FILES ${CMAKE_CURRENT_BINARY_DIR}/mlx/mlx/backend/metal/kernels/mlx.metallib
      DESTINATION mlx_onnx
      COMPONENT python)
  endif()
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mlx/python/mlx/
    DESTINATION mlx
    COMPONENT python
    PATTERN "__pycache__" EXCLUDE)
  install(
    FILES ${MLX_ONNX_PY_INIT_FILE}
    DESTINATION mlx_onnx
    RENAME __init__.py
    COMPONENT python)
  set(MLX_ONNX_VENDOR_MLX_ROOT mlx_onnx/_vendor/mlx)
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/mlx/CMakeLists.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/mlx/mlx.pc.in
          ${CMAKE_CURRENT_SOURCE_DIR}/mlx/LICENSE
          ${CMAKE_CURRENT_SOURCE_DIR}/mlx/ACKNOWLEDGMENTS.md
    DESTINATION ${MLX_ONNX_VENDOR_MLX_ROOT}
    COMPONENT python)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mlx/cmake
              ${CMAKE_CURRENT_SOURCE_DIR}/mlx/mlx
    DESTINATION ${MLX_ONNX_VENDOR_MLX_ROOT}
    COMPONENT python)
endif()

if(MLX_ONNX_INSTALL_CPP_ARTIFACTS)
  include(GNUInstallDirs)
  install(
    TARGETS mlx_onnx
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT cpp)

  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT cpp)
endif()
