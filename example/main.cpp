#define WGPU_IMPLEMENTATION
#include <wgfx.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_wgpu.h"
#include "context.h"
#include "clock.h"
#include "quad.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

int main()
{
	Context& context = Context::Instance();

	Quad& quad = Quad::Instance();

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplSDL3_InitForOther(context.window);
	ImGui_ImplWGPU_InitInfo init_info = {};
	init_info.Device = (WGPUDevice)wgfx::device;
	init_info.NumFramesInFlight = 2;
	init_info.RenderTargetFormat = (WGPUTextureFormat)wgfx::surfaceFormat;
	init_info.DepthStencilFormat = WGPUTextureFormat_Undefined;
	ImGui_ImplWGPU_Init(&init_info);

	wgfx::ColorTexture* color = new wgfx::ColorTexture();
	wgfx::RenderPass* pass = new wgfx::RenderPass();
	wgfx::RenderPass* uiPass = new wgfx::RenderPass();
	pass->addTarget(color);
	uiPass->addTarget(color);
	pass->setClear({ 0.0f, 0.0f, 0.0f, 1.0f });
	uiPass->shouldClear = false;

	struct AppState {
		Context* context;
		Quad* quad;
		wgfx::ColorTexture* color;
		wgfx::RenderPass* pass;
		wgfx::RenderPass* uiPass;
		float fpsTimer = 0.0f;
		float frameTimeAccumulator = 0.0f;
		int frameCount = 0;
	};

	AppState* appState = new AppState{ &context, &quad, color, pass, uiPass };

	auto loop = [](void* arg) {
		AppState* state = static_cast<AppState*>(arg);
		static Clock clock;
		float dt = clock.restart();
		
		// Accumulate frame times for averaging
		state->frameTimeAccumulator += dt;
		state->frameCount++;
		state->fpsTimer += dt;
		
		// Update FPS display every 0.5 seconds using average frame time
		if (state->fpsTimer > 0.5f)
		{
			float averageFrameTime = state->frameTimeAccumulator / state->frameCount;
			state->context->fps(averageFrameTime);
			state->fpsTimer = 0;
			state->frameTimeAccumulator = 0;
			state->frameCount = 0;
		}

		state->context->update();
		ImGui_ImplWGPU_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();
		state->quad->drawImGuiPanel();
		ImGui::Render();

		wgfx::touch(state->color);
		wgfx::start();

		state->quad->dispatchComputeSimulations(dt);

		state->pass->prepare();
			state->quad->render(dt);
		state->pass->draw(state->quad->pipeline);
		state->pass->end();

		state->uiPass->prepare();
		ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), (WGPURenderPassEncoder)state->uiPass->renderPass);
		state->uiPass->end();

		state->context->draw();

#ifdef __EMSCRIPTEN__
		if (state->context->close) {
			emscripten_cancel_main_loop();
		}
#endif
	};

#ifdef __EMSCRIPTEN__
	emscripten_set_main_loop_arg(loop, appState, 0, true);
#else
	while (!context.close)
	{
		loop(appState);
	}

	ImGui_ImplWGPU_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();
#endif
}
