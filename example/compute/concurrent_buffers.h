#pragma once

#include "voxel_volume.h"

namespace compute_layers {

using BatchedUpdate = VoxelCompute::BatchedUpdate;
using BrickUpdate = VoxelCompute::BrickUpdate;
using VoxelEditCommand = VoxelCompute::VoxelEditCommand;
using CircularBuffer = VoxelCompute::CircularBuffer;
using DoubleBuffer = VoxelCompute::DoubleBuffer;

} // namespace compute_layers
