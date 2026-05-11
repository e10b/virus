#pragma once

#include <wgfx.h>
#include <string>
class Context
{
public:
	static Context& Instance()
	{
		static Context instance;
		return instance;
	}

	Context();
	void fps(float averageFrameTime)
	{
		float fps = 1.0f / averageFrameTime;

		std::string title = "Viral Dynamics - FPS: " + std::to_string(static_cast<int>(fps + 0.5f));
		SDL_SetWindowTitle(window, title.c_str());
	}

	void update();
	float consumeWheelDelta()
	{
		float d = wheelDeltaY;
		wheelDeltaY = 0.0f;
		return d;
	}

	void draw();

	bool close = false;
	SDL_Window* window;
	float wheelDeltaY = 0.0f;
};