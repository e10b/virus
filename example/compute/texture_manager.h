#pragma once

#include <cstdint>
#include <string>

#include <glm/glm.hpp>

#include "voxel_volume.h"

namespace compute_layers {

using TextureData = VoxelCompute::TextureData;
using TextureManager = VoxelCompute::TextureManager;

uint32_t getTextureIndex(TextureManager& manager, const std::string& texturePath);
glm::vec3 sampleTexture(const TextureManager& manager, uint32_t textureIndex, const glm::vec2& uv);

} // namespace compute_layers
