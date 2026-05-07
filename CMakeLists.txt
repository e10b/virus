cmake_minimum_required(VERSION 3.5...3.25)
project(atoms VERSION 0.1.0 LANGUAGES CXX C)

#add_subdirectory(deps/sdl)
#add_subdirectory(deps/webgpu)
#add_subdirectory(deps/sdl2webgpu)
#set(SHADERC_SKIP_TESTS ON CACHE INTERNAL "")
#add_subdirectory(deps/shaderc)
option(DEV_MODE "Set up development helper settings" ON)
option(ATOMS_ENABLE_ONNX "Enable ONNX Runtime integration for TDSE inference" OFF)
option(ATOMS_ENABLE_TORCHSCRIPT "Enable TorchScript/libtorch integration for TDSE inference" OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation root")
set(TORCH_ROOT "" CACHE PATH "Path to libtorch installation root")

# Disable LevelDB tests to avoid pulling in googletest when not required.
# You can also pass -DLEVELDB_BUILD_TESTS=OFF to cmake on the command line.
set(LEVELDB_BUILD_TESTS OFF CACHE BOOL "Disable LevelDB tests (do not pull googletest)" FORCE)
# Also disable LevelDB benchmarks to avoid building db_bench which depends on gmock.
set(LEVELDB_BUILD_BENCHMARKS OFF CACHE BOOL "Disable LevelDB benchmarks (do not build db_bench)" FORCE)

add_subdirectory(deps)


set(src
	example/context.cpp
	example/clock.cpp
)
file(GLOB_RECURSE srcc CONFIGURE_DEPENDS "deps/wgfx/src/*.cpp")
set(imgui_src
	deps/imgui/imgui.cpp
	deps/imgui/imgui_draw.cpp
	deps/imgui/imgui_tables.cpp
	deps/imgui/imgui_widgets.cpp
	deps/imgui/imgui_demo.cpp
	deps/imgui/backends/imgui_impl_sdl3.cpp
	deps/imgui/backends/imgui_impl_wgpu.cpp
)

add_executable( App ${src} ${srcc} ${imgui_src} example/main.cpp )
set_source_files_properties(deps/imgui/backends/imgui_impl_wgpu.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")
 #"sdl2webgpu/sdl2webgpu.c" "sdl2webgpu/sdl2webgpu.h")

 if(DEV_MODE)
 # In dev mode, we load resources from the source tree, so that when we
 # dynamically edit resources (like shaders), these are correctly
 # versionned.
 target_compile_definitions(App PRIVATE
	 RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res"
 )
else()
 # In release mode, we just load resources relatively to wherever the
 # executable is launched from, so that the binary is portable
 target_compile_definitions(App PRIVATE
	 RESOURCE_DIR="./res"
 )
endif()

target_link_libraries(App PRIVATE webgpu sdl2webgpu SDL3-static glm::glm)
target_include_directories(App PRIVATE "deps/wgfx/src" "deps/sdl3/include" "deps" "deps/imgui" "deps/imgui/backends")
target_compile_definitions(App PRIVATE IMGUI_IMPL_WEBGPU_BACKEND_WGPU)

if(ATOMS_ENABLE_ONNX)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session onnxruntime/include
  )

  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
  )

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ATOMS_ENABLE_ONNX=ON but ONNX Runtime not found. Set ONNXRUNTIME_ROOT to your ONNX Runtime install path.")
  endif()

  target_include_directories(App PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(App PRIVATE ${ONNXRUNTIME_LIBRARY})
  target_compile_definitions(App PRIVATE ATOMS_ENABLE_ONNX=1)
endif()

if(ATOMS_ENABLE_TORCHSCRIPT)
  if(TORCH_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_ROOT}")
  endif()

  find_package(Torch REQUIRED)
  target_link_libraries(App PRIVATE ${TORCH_LIBRARIES})
  target_compile_definitions(App PRIVATE ATOMS_ENABLE_TORCHSCRIPT=1)
endif()

target_copy_webgpu_binaries(App)


set_target_properties(App PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
#	COMPILE_WARNING_AS_ERROR ON
)