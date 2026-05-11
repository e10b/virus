cmake_minimum_required(VERSION 3.5...3.25)
project(virus VERSION 0.1.0 LANGUAGES CXX C)

# Graphics stack: Vyscosity/wgfx (WebGPU + SDL3 + GLM helpers).
option(DEV_MODE "Load resources from the source tree" ON)

add_subdirectory(deps/wgfx)

set(APP_SRC
	example/context.cpp
	example/clock.cpp
)
file(GLOB_RECURSE WGFX_LIB_SRC CONFIGURE_DEPENDS "deps/wgfx/src/*.cpp")
set(IMGUI_SRC
	deps/imgui/imgui.cpp
	deps/imgui/imgui_draw.cpp
	deps/imgui/imgui_tables.cpp
	deps/imgui/imgui_widgets.cpp
	deps/imgui/imgui_demo.cpp
	deps/imgui/backends/imgui_impl_sdl3.cpp
	deps/imgui/backends/imgui_impl_wgpu.cpp
)

add_executable(App ${APP_SRC} ${WGFX_LIB_SRC} ${IMGUI_SRC} example/main.cpp)
set_source_files_properties(deps/imgui/backends/imgui_impl_wgpu.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")

if(DEV_MODE)
	target_compile_definitions(App PRIVATE
		RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res"
	)
else()
	target_compile_definitions(App PRIVATE
		RESOURCE_DIR="./res"
	)
endif()

target_link_libraries(App PRIVATE webgpu sdl2webgpu SDL3-static glm::glm)
target_include_directories(App PRIVATE
	"deps/wgfx/src"
	"deps/wgfx/deps"
	"deps/wgfx/deps/sdl3/include"
	"deps/imgui"
	"deps/imgui/backends"
	"example"
)
target_compile_definitions(App PRIVATE IMGUI_IMPL_WEBGPU_BACKEND_WGPU)

target_copy_webgpu_binaries(App)

set_target_properties(App PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
)
