#pragma once

#include <cstddef>
#include <cstdint>

#include "voxel_volume.h"

namespace compute_layers {

using BrickMeta = VoxelCompute::BrickMeta;

inline constexpr std::size_t kBrickVoxelCount = 512;

uint32_t allocateBrick(VoxelCompute& volume);
void freeBrick(VoxelCompute& volume, uint32_t brickIndex);

} // namespace compute_layers
