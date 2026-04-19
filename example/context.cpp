#include "context.h"
#include "imgui_impl_sdl3.h"

Context::Context()
{
	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) { std::cout << "Couldn't init SDL!\n"; }
	window = SDL_CreateWindow("Learn WebGPU", 1280, 720, SDL_WINDOW_RESIZABLE);
	wgfx::init(wgfx::getSurface(window));

}

void Context::update()
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		ImGui_ImplSDL3_ProcessEvent(&event);
		switch (event.type)
		{
		case SDL_EVENT_WINDOW_RESIZED:
		{
			wgfx::initSurface();
			//wgfx::initDepth();
			//gonna need a different way to do this


			int width, height;
			SDL_GetWindowSize(window, &width, &height);

			//proj = glm::perspective(glm::radians(50.0f), float(width) / float(height), 0.1f, 100.0f);
			//pipeline.updateUniform(projUniform, glm::value_ptr(proj));
		}
		break;

		case SDL_EVENT_QUIT:
			close = true;
			break;

		case SDL_EVENT_KEY_DOWN:
			if (event.key.key == SDLK_ESCAPE) {
				close = true; // Close the application if Escape is pressed
			}
			break;

		case SDL_EVENT_WINDOW_EXPOSED:
			wgfx::initSurface();
			break;

		case SDL_EVENT_MOUSE_WHEEL:
			wheelDeltaY += event.wheel.y;
			break;
		}
	}
}

void Context::draw()
{
	wgfx::frame();
}

