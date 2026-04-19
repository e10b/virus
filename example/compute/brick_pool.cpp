#include "brick_pool.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

uint32_t VoxelCompute::packColor(uint8_t r, uint8_t g, uint8_t b) {
    return (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
}

size_t VoxelCompute::getTopLevelIndex(int x, int y, int z) const {
    int topX = x / brickSize;
    int topY = y / brickSize;
    int topZ = z / brickSize;
    return topX + topY * topLevelSize + topZ * topLevelSize * topLevelSize;
}

size_t VoxelCompute::getSectorIndex(int x, int y, int z) const {
    int sectorX = x / sectorSize;
    int sectorY = y / sectorSize;
    int sectorZ = z / sectorSize;
    return sectorX + sectorY * sectorsPerDimension + sectorZ * sectorsPerDimension * sectorsPerDimension;
}

size_t VoxelCompute::getBrickVoxelOffset(int localX, int localY, int localZ) const {
    const uint32_t x = static_cast<uint32_t>(localX) & 0x7u;
    const uint32_t y = static_cast<uint32_t>(localY) & 0x7u;
    const uint32_t z = static_cast<uint32_t>(localZ) & 0x7u;

    // 3-bit Morton interleave: x0 y0 z0 x1 y1 z1 x2 y2 z2
    return ((x & 1u) << 0) | ((y & 1u) << 1) | ((z & 1u) << 2) |
           ((x & 2u) << 2) | ((y & 2u) << 3) | ((z & 2u) << 4) |
           ((x & 4u) << 4) | ((y & 4u) << 5) | ((z & 4u) << 6);
}

void VoxelCompute::updateSector(int sectorX, int sectorY, int sectorZ) {
    (void)sectorX;
    (void)sectorY;
    (void)sectorZ;
    sectorMapDirty = true;
}

void VoxelCompute::updateAllSectors() {
    sectorMapDirty = true;
}

void VoxelCompute::updateSectorForVoxel(int x, int y, int z) {
    int sectorX = x / sectorSize;
    int sectorY = y / sectorSize;
    int sectorZ = z / sectorSize;
    updateSector(sectorX, sectorY, sectorZ);
}

uint32_t VoxelCompute::allocateBrickInternal() {
    uint32_t brickIndex;
    if (freeList.pop(&brickIndex)) {
        if (brickIndex < maxBrickPoolSize) {
            brickMetadata[brickIndex].inUse = true;
            brickMetadata[brickIndex].nonEmptyCount = 0;
            memset(&brickPool[brickIndex * 512], 0, 512 * sizeof(uint16_t));
            return brickIndex;
        }
    }

    if (currentBrickPoolSize < maxBrickPoolSize) {
        brickIndex = currentBrickPoolSize.fetch_add(1);
        if (brickIndex < maxBrickPoolSize) {
            brickMetadata[brickIndex].inUse = true;
            brickMetadata[brickIndex].nonEmptyCount = 0;
            memset(&brickPool[brickIndex * 512], 0, 512 * sizeof(uint16_t));
            return brickIndex;
        }
    }

    return maxBrickPoolSize;
}

void VoxelCompute::freeBrickInternal(uint32_t brickIndex) {
    if (brickIndex < maxBrickPoolSize && brickMetadata[brickIndex].inUse) {
        brickMetadata[brickIndex].inUse = false;
        brickMetadata[brickIndex].nonEmptyCount = 0;
        freeList.push(brickIndex);
    }
}

uint32_t VoxelCompute::allocateBrick() {
    std::lock_guard<std::mutex> lock(brickMutex);
    return allocateBrickInternal();
}

void VoxelCompute::freeBrick(uint32_t brickIndex) {
    std::lock_guard<std::mutex> lock(brickMutex);
    freeBrickInternal(brickIndex);
}

uint32_t VoxelCompute::allocateBrickAtomic() {
    return allocateBrick();
}

void VoxelCompute::testVoxelPlacement() {
    printf("=== Testing Voxel Placement & Destruction ===\n");

    int testX = worldSize / 2;
    int testY = worldSize / 2;
    int testZ = worldSize / 2;

    printf("1. Placing test voxel at (%d, %d, %d)\n", testX, testY, testZ);
    setVoxel(testX, testY, testZ, 85);

    uint16_t voxelValue = getVoxel(testX, testY, testZ);
    printf("   Retrieved voxel value: %d (should be > 0)\n", voxelValue);

    bool placementWorked = (voxelValue > 0);
    if (placementWorked) {
        printf("   ✅ Voxel placement PASSED!\n");
    } else {
        printf("   ❌ Voxel placement FAILED!\n");
    }

    updateGPUBuffers(testX, testY, testZ);

    printf("2. Destroying test voxel at (%d, %d, %d)\n", testX, testY, testZ);
    clearVoxel(testX, testY, testZ);

    uint16_t voxelValueAfter = getVoxel(testX, testY, testZ);
    printf("   Retrieved voxel value after destruction: %d (should be 0)\n", voxelValueAfter);

    bool destructionWorked = (voxelValueAfter == 0);
    if (destructionWorked) {
        printf("   ✅ Voxel destruction PASSED!\n");
    } else {
        printf("   ❌ Voxel destruction FAILED!\n");
    }

    updateGPUBuffers(testX, testY, testZ);

    if (placementWorked && destructionWorked) {
        printf("🎉 ALL TESTS PASSED! Voxel placement and destruction working correctly.\n");
    } else {
        printf("💥 SOME TESTS FAILED! Check the voxel system.\n");
    }

    printf("=== End Voxel Test ===\n\n");
}

void VoxelCompute::testCircularBuffer() {
    printf("=== Testing Circular Buffer ===\n");
    printVRAMUsage("Before Freeing Bricks");

    std::vector<size_t> modifiedCells;

    uint32_t freedCount = 0;
    for (uint32_t i = 0; i < std::min(currentBrickPoolSize.load(), size_t(50)) && freedCount < 50; ++i) {
        if (brickMetadata[i].inUse) {
            for (size_t topIndex = 0; topIndex < topLevelCount; ++topIndex) {
                uint32_t value = topLevelGrid[topIndex];
                if ((value & IS_BRICK_FLAG) && ((value & BRICK_INDEX_MASK) == i)) {
                    topLevelGrid[topIndex] = 0;
                    modifiedCells.push_back(topIndex);
                    printf("  Cleared top-level cell %zu (was pointing to brick %u)\n", topIndex, i);
                    break;
                }
            }

            freeBrick(i);
            freedCount++;
        }
    }

    printVRAMUsage("After Freeing 50 Bricks");

    if (!modifiedCells.empty()) {
        copyTopLevelGridToTexture();
        printf("  Updated GPU texture with %zu cleared cells - BRICKS SHOULD DISAPPEAR NOW!\n", modifiedCells.size());
    }

    for (int i = 0; i < 25; ++i) {
        uint32_t newBrick = allocateBrick();
        printf("  Allocated brick %u (reused from circular buffer)\n", newBrick);

        for (int voxelIdx = 0; voxelIdx < 512; ++voxelIdx) {
            brickPool[newBrick * 512 + voxelIdx] = 0;
        }
        brickMetadata[newBrick].nonEmptyCount = 0;
    }

    printVRAMUsage("After Reallocating 25 Bricks");
    printf("=== End Circular Buffer Test - 50 BRICKS FREED, 25 REALLOCATED ===\n\n");
}

namespace compute_layers {

uint32_t allocateBrick(VoxelCompute& volume) {
    return volume.allocateBrick();
}

void freeBrick(VoxelCompute& volume, uint32_t brickIndex) {
    volume.freeBrick(brickIndex);
}

} // namespace compute_layers
