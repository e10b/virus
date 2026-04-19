#pragma once

#include <glm/glm.hpp>
#include "model.h"
#include "wgfx.h"

#include "constants.h"

class Quad
{
public:
	static Quad& Instance()
	{
		static Quad instance;
		return instance;
	}

	wgfx::Pipeline* pipeline;

	Quad()
	{
		pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "atoms_sphere.wgsl").c_str()));

		float quadWidth = 1.0f;
		float quadHeight = 1.0f;

		// Offset from center
		float x = 0.0f;
		float y = 0.0f;

		quad_.addData({
			//   x,             y,            z
			-quadWidth + x, -quadHeight + y, 0.0f,  // bottom-left
			 quadWidth + x, -quadHeight + y, 0.0f,  // bottom-right
			 quadWidth + x,  quadHeight + y, 0.0f,  // top-right
			-quadWidth + x,  quadHeight + y, 0.0f   // top-left
			}, {
				0, 1, 2,
				0, 2, 3
			});

		wgfx::VertexBuffer* vbo = wgfx::createVertexBuffer();
		vbo->setAttribute(0, wgfx::vec3f, 0); // pos

		std::vector<float> verticies;
		verticies.push_back(0.0);

		std::vector<uint16_t> indicies;
		indicies.push_back(0);

		pipeline->setVertexBuffer(verticies);
		pipeline->setIndexBuffer(indicies);

		pipeline->targets = 1;
		pipeline->useDepth = false;

		pipeline->init(vbo);
	}

	void render()
	{
		pipeline->uniforms.clear(); // Reset uniform count
		quad_.bind(pipeline);
	}

private:
	Model quad_;

public:
	Quad(Quad const&) = delete;
	void operator=(Quad const&) = delete;
};