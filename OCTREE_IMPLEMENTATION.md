# Octree Implementation for WGFX Raytracing

This document outlines the changes made to replace the brickmap hierarchy with an octree-based acceleration structure for voxel raytracing.

## Overview

The previous implementation used a regular grid-based "brickmap" system that divided the 512³ voxel space into 8³ voxel bricks (64 bricks per dimension). This has been replaced with a hierarchical octree that adapts to the actual voxel distribution, providing better performance for sparse voxel data.

## Key Changes

### 1. WGSL Shader Changes (`res/quad.wgsl`)

#### Storage Buffer Changes
- **Before**: `@group(0) @binding(4) var<storage, read> brickMap: array<u32>;`
- **After**: `@group(0) @binding(4) var<storage, read> octreeNodes: array<u32>;`

#### New Octree Constants and Functions
```wgsl
const OCTREE_MAX_DEPTH = 9; // For 512x512x512 grid
const OCTREE_ROOT_SIZE = 512;
```

**Key Functions Added:**
- `getChildMask(node: u32) -> u32`: Extract 8-bit child mask from node
- `getChildPointer(node: u32) -> u32`: Extract 24-bit child pointer from node
- `hasChild(node: u32, childIndex: u32) -> bool`: Check if node has specific child
- `getChildNodeIndex(node: u32, childIndex: u32) -> u32`: Get index of child in array
- `getOctant(pos: vec3<i32>, nodeMin: vec3<i32>, nodeSize: i32) -> u32`: Determine octant for position
- `getOctantMin(octant: u32, nodeMin: vec3<i32>, nodeSize: i32) -> vec3<i32>`: Get octant bounds
- `isRegionOccupied(regionMin: vec3<i32>, regionMax: vec3<i32>) -> bool`: Check region using octree traversal

#### Updated DDA Algorithm
The `hierarchicalVoxelDDA` function now uses:
- **Region-based empty space skipping** (16³ voxel regions instead of 8³ bricks)
- **Octree queries** to determine if regions contain solid voxels
- **Stack-based octree traversal** for efficient spatial queries

### 2. C++ Header Changes (`example/quad.h`)

#### Data Structure Changes
**Removed:**
```cpp
const int brickSize = 8;
const int bricksX = 16, bricksY = 16, bricksZ = 16;
size_t brickCount = bricksX * bricksY * bricksZ;
uint32_t* brickMap = new uint32_t[brickCount];
```

**Added:**
```cpp
const int octreeMaxDepth = 8;
const int octreeRootSize = 256;
std::vector<uint32_t> octreeNodes;
```

#### New Octree Functions
- `packOctreeNode(uint8_t childMask, uint32_t childPointer) -> uint32_t`: Pack node data
- `getChildMask(uint32_t node) -> uint8_t`: Extract child mask
- `getChildPointer(uint32_t node) -> uint32_t`: Extract child pointer
- `hasVoxelsInRegion(minX, minY, minZ, maxX, maxY, maxZ) -> bool`: Check region for voxels
- `buildOctreeNode(minX, minY, minZ, size, depth) -> uint32_t`: Recursively build octree
- `buildOctree()`: Build complete octree from voxel data

#### Node Format
Each octree node is packed into a single `uint32_t`:
- **Bits 0-7**: Child mask (8 bits for 8 children)
- **Bits 8-31**: Child pointer (24-bit index into octreeNodes array)
  - If 0: Leaf node
  - If non-zero: Points to first child in array

## Performance Benefits

### 1. Adaptive Structure
- **Brickmap**: Fixed 64³ = 262,144 bricks regardless of voxel distribution
- **Octree**: Adaptive structure that only creates nodes where voxels exist

### 2. Better Empty Space Skipping
- **Brickmap**: Could skip 8³ = 512 voxel regions
- **Octree**: Can skip much larger regions based on hierarchy level

### 3. Memory Efficiency
- **Brickmap**: Always uses 262,144 × 4 bytes = 1,048,576 bytes (1 MB)
- **Octree**: Variable size, typically much smaller for sparse data

### 4. Hierarchical Queries
- **Brickmap**: O(1) lookup, but no hierarchy
- **Octree**: O(log n) queries with early termination for large empty regions

## Usage Notes

### Building the Octree
The octree is automatically built when:
1. The scene is initialized (`buildOctree()` called in constructor)
2. Voxels are modified via raycast (`buildOctree()` called after changes)

### Shader Integration
The octree nodes array is passed to the shader as storage buffer binding 4, replacing the previous brickMap.

### Tuning Parameters
- `regionSize = 16` in the DDA function controls the granularity of empty space skipping
- `OCTREE_MAX_DEPTH = 9` matches the 512³ grid size (2^9 = 512)
- `OCTREE_MAX_DEPTH = 9` matches the 512³ grid size (2^9 = 512)
- Stack size of 16 in `isRegionOccupied` limits traversal depth for safety

## Future Optimizations

1. **Incremental Updates**: Instead of rebuilding the entire octree, implement localized updates
2. **GPU-Based Octree**: Move octree construction to GPU using compute shaders
3. **Compressed Nodes**: Use bit manipulation to pack more information per node
4. **Ray-Octree Intersection**: Implement direct ray-octree traversal instead of region-based skipping

## Testing

The implementation maintains the same external interface as the brickmap version, so existing raytracing and voxel modification code should work without changes. The octree provides the same spatial acceleration benefits while being more memory-efficient and adaptive to the actual voxel distribution.
