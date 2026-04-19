#define WGPU_IMPLEMENTATION
#include <wgfx.h>
#include "context.h"
#include "clock.h"
#include "quad.h"

int main()
{
	Context& context = Context::Instance();

	Quad& quad = Quad::Instance();

	wgfx::ColorTexture* color = new wgfx::ColorTexture();
	wgfx::RenderPass pass;
	pass.addTarget(color);
	pass.setClear({ 0.0f, 0.0f, 0.0f, 1.0f });

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

		wgfx::touch(color);

		// Render the fullscreen quad with analytic sphere ray tracing in fragment WGSL.
		pass.prepare();
			quad.render();
		pass.draw(quad.pipeline);
		pass.end();

		context.draw();
	}
}