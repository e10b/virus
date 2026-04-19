#pragma once

#include <vector>

#include "voxel_volume.h"

namespace compute_layers {

void voxelizeTriangles(VoxelCompute& volume, const std::vector<Triangle>& triangles);
void voxelizeTrianglesLegacy(VoxelCompute& volume, const std::vector<Triangle>& triangles);

} // namespace compute_layers
