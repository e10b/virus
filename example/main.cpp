#define WGPU_IMPLEMENTATION
#include <wgfx.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_wgpu.h"
#include "context.h"
#include "clock.h"
#include "quad.h"

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
	wgfx::RenderPass pass;
	wgfx::RenderPass uiPass;
	pass.addTarget(color);
	uiPass.addTarget(color);
	pass.setClear({ 0.0f, 0.0f, 0.0f, 1.0f });
	uiPass.shouldClear = false;

	float fpsTimer = 0;
	float frameTimeAccumulator = 0;
	int frameCount = 0;
	while (!context.close)
	{
		static Clock clock;
		float dt = clock.restart();
		
		// Accumulate frame times for averaging
		frameTimeAccumulator += dt;
		frameCount++;
		fpsTimer += dt;
		
		// Update FPS display every 0.5 seconds using average frame time
		if (fpsTimer > 0.5f)
		{
			float averageFrameTime = frameTimeAccumulator / frameCount;
			context.fps(averageFrameTime);
			fpsTimer = 0;
			frameTimeAccumulator = 0;
			frameCount = 0;
		}

		context.update();
		ImGui_ImplWGPU_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();
		quad.drawImGuiPanel();
		ImGui::Render();

		wgfx::touch(color);

		// Render the fullscreen quad with analytic sphere ray tracing in fragment WGSL.
		pass.prepare();
			quad.render(dt);
		pass.draw(quad.pipeline);
		pass.end();

		uiPass.prepare();
		ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), (WGPURenderPassEncoder)uiPass.renderPass);
		uiPass.end();

		context.draw();
	}

	ImGui_ImplWGPU_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();
}