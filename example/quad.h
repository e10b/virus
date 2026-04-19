#pragma once

#include <memory>
#include <vector>
#include "wgfx.h"

class Quad
{
public:
	static Quad& Instance()
	{
		static Quad instance;
		return instance;
	}

	wgfx::Pipeline* pipeline;
	std::unique_ptr<wgfx::VertexBuffer> vbo_;
	std::unique_ptr<wgfx::IndexBuffer> ibo_;

	Quad()
	{
		pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "atoms_sphere.wgsl").c_str()));

		const std::vector<float> vertices = {
			-1.0f, -1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f
		};
		const std::vector<uint16_t> indices = {
			0, 1, 2,
			0, 2, 3
		};

		vbo_.reset(wgfx::createVertexBuffer(vertices));
		vbo_->setAttribute(0, wgfx::vec3f, 0);
		ibo_.reset(wgfx::createIndexBuffer(indices));

		pipeline->setVertexBuffer(vbo_.get());
		pipeline->setIndexBuffer(ibo_.get());

		pipeline->targets = 1;
		pipeline->useDepth = false;

		pipeline->init(vbo_.get());
	}

	void render()
	{
		pipeline->uniforms.clear(); // Reset uniform count
		pipeline->setVertexBuffer(vbo_.get());
		pipeline->setIndexBuffer(ibo_.get());
	}

public:
	Quad(Quad const&) = delete;
	void operator=(Quad const&) = delete;
};