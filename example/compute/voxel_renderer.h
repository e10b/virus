#pragma once

#include <glm/glm.hpp>

#include "voxel_volume.h"

namespace compute_layers {

void render(VoxelCompute& volume, const glm::vec3& cameraPos, const glm::mat4& invViewProj, const glm::vec2& resolution);
void flushGpuUpdates(VoxelCompute& volume);
void updateVoxelGpuCell(VoxelCompute& volume, int x, int y, int z);

} // namespace compute_layers
