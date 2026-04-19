#pragma once

#include <glm/glm.hpp>

#include "voxel_volume.h"

namespace compute_layers {

bool isVoxelOccupied(const VoxelCompute& volume, int x, int y, int z);
uint16_t getVoxel(const VoxelCompute& volume, int x, int y, int z);
bool raycast(
    VoxelCompute& volume,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float length,
    bool add,
    int toolMode = 1,
    bool preventPlayerIntersection = false,
    const glm::vec3& playerFeetPosition = glm::vec3(0.0f),
    const glm::vec3& playerSize = glm::vec3(0.0f));

} // namespace compute_layers
