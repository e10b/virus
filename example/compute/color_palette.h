#pragma once

#include <cstdint>
#include <set>

#include "voxel_volume.h"

namespace compute_layers {

using ColorPalette = VoxelCompute::ColorPalette;

void initializeTerrainPalette(VoxelCompute& volume);
void buildColorPalette(VoxelCompute& volume, const std::set<uint32_t>& uniqueColors);
uint8_t findClosestPaletteIndex(const VoxelCompute& volume, uint32_t color);

} // namespace compute_layers
