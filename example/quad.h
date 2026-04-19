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
	wgfx::Texture texture;

	Quad()
	{
		pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "quad.wgsl").c_str()));

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

		// Simple quad pipeline setup - just texture and sampler
		texture = wgfx::loadTextureDst(raytraceWidth, raytraceHeight);
		pipeline->addTexture(0, texture);
		pipeline->addSampler(1, texture);

		pipeline->targets = 1;
		pipeline->useDepth = false;

		pipeline->init(vbo);
	}

	void render()
	{
		pipeline->uniforms.clear(); // Reset uniform count
		quad_.bind(pipeline);
	}

	// Brickmap configuration
	// Hierarchy: Sector → Brick → Voxel
	const int brickSize = 8; // 8x8x8 voxels per brick
	const int bricksPerSector = 2; // 2x2x2 bricks per sector (16x16x16 voxels per sector)
	
	const int size = 512;
	const int sizeX = size;
	const int sizeY = size;
	const int sizeZ = size;

	const int sectorSize = brickSize * bricksPerSector; // 16 voxels per sector side
	const int sectorsX = sizeX / sectorSize;
	const int sectorsY = sizeY / sectorSize;
	const int sectorsZ = sizeZ / sectorSize;
	
	const int bricksX = sizeX / brickSize; // 128 bricks
	const int bricksY = sizeY / brickSize; // 128 bricks
	const int bricksZ = sizeZ / brickSize; // 128 bricks

	size_t voxelCount = sizeX * sizeY * sizeZ;
	size_t brickCount = bricksX * bricksY * bricksZ;
	size_t sectorCount = sectorsX * sectorsY * sectorsZ;
	size_t quarterVoxelCount = voxelCount / 4;

	uint32_t* voxelData = new uint32_t[voxelCount]; // For voxelizer (full array)

private:
	Model quad_;

public:
	Quad(Quad const&) = delete;
	void operator=(Quad const&) = delete;
};