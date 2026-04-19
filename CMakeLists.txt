cmake_minimum_required(VERSION 3.5...3.25)
project(atoms VERSION 0.1.0 LANGUAGES CXX C)

#add_subdirectory(deps/sdl)
#add_subdirectory(deps/webgpu)
#add_subdirectory(deps/sdl2webgpu)
#set(SHADERC_SKIP_TESTS ON CACHE INTERNAL "")
#add_subdirectory(deps/shaderc)
option(DEV_MODE "Set up development helper settings" ON)

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

add_executable( App ${src} ${srcc} example/main.cpp )
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
target_include_directories(App PRIVATE "deps/wgfx/src" "deps/sdl3/include" "deps")

target_copy_webgpu_binaries(App)


set_target_properties(App PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
#	COMPILE_WARNING_AS_ERROR ON
)