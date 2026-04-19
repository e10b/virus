#define WGPU_IMPLEMENTATION
#include <wgfx.h>
#include "context.h"
#include "clock.h"
#include "audio.h"

// Test without player first
 #include "player.h"
#include "quad.h"
#include "compute/voxel_volume.h"

#include "constants.h"

int main()
{
	Context& context = Context::Instance();
	AudioManager::Instance().initialize();

	Quad& quad = Quad::Instance();
	VoxelCompute& voxelCompute = VoxelCompute::Instance();
	Player player;

	wgfx::ColorTexture* color = new wgfx::ColorTexture();
	wgfx::RenderPass pass;
	pass.addTarget(color);
	pass.setClear({ 0.4, 0.7, 1, 1 });

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
		player.update(dt);

		const Camera& cam = player.getCamera();

		wgfx::touch(color);
		wgfx::start();
		voxelCompute.runAudioRaytraceGPU();

		// Run voxel raytracing compute shader
		voxelCompute.render(cam.getPosition(), glm::inverse(cam.getMatrix()), glm::vec2(raytraceWidth, raytraceHeight));

		// Copy compute output to quad texture
		wgfx::copyTextureToTexture(&voxelCompute.outputTexture, &quad.texture);
			
		// Render the quad with the computed texture
		pass.prepare();
			quad.render();
		pass.draw(quad.pipeline);
		pass.end();

		// Only submit once at the end of each frame
		context.draw();
		voxelCompute.processAudioRaytraceReadbacks();
	}
}