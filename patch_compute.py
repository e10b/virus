import sys
import re

with open('example/compute.h', 'r') as f:
    code = f.read()

# 1. Add headers
code = code.replace('#include <algorithm>\n', '#include <algorithm>\n#include <thread>\n#include <atomic>\n#include <mutex>\n')

# 2. Make currentBrickPoolSize atomic
code = code.replace('size_t currentBrickPoolSize = 0;', 'std::atomic<size_t> currentBrickPoolSize{0};')

# 3. Add an atomic allocateBrick method
code = code.replace('uint32_t allocateBrick() {', '''uint32_t allocateBrickAtomic() {
        uint32_t brickIndex = currentBrickPoolSize.fetch_add(1, std::memory_order_relaxed);
        if (brickIndex < maxBrickPoolSize) {
            brickMetadata[brickIndex].inUse = true;
            brickMetadata[brickIndex].nonEmptyCount = 0;
            memset(&brickPool[brickIndex * 512], 0, 512);
            return brickIndex;
        }
        return maxBrickPoolSize;
    }
    
    uint32_t allocateBrick() {''')

# 4. Modify voxelizeTriangles to be massively multithreaded with Atomic Allocator
old_voxelize = '''    void voxelizeTriangles(const std::vector<Triangle>& triangles) {'''

new_voxelize = '''    void voxelizeTriangles(const std::vector<Triangle>& triangles) {
        if (triangles.empty()) {
            printf("No triangles to voxelize!\\n");
            return;
        }

        printf("🚀 Voxelizing %zu triangles using HIGH-SPEED ATOMIC ALLOCATOR across all CPU cores...\\n", triangles.size());

        // Find bounding box of all triangles
        glm::vec3 minBounds(FLT_MAX);
        glm::vec3 maxBounds(-FLT_MAX);
        for (const auto& tri : triangles) {
            for (int i = 0; i < 3; ++i) {
                glm::vec3 vertex = (i == 0) ? tri.v0 : (i == 1) ? tri.v1 : tri.v2;
                minBounds = glm::min(minBounds, vertex);
                maxBounds = glm::max(maxBounds, vertex);
            }
        }

        glm::vec3 size = maxBounds - minBounds;
        float maxSize = glm::max(glm::max(size.x, size.y), size.z);
        float scale = worldSize * 0.8f / maxSize;
        glm::vec3 offset = glm::vec3(worldSize * 0.1f) - minBounds * scale;

        // Populate a fixed 256-color RGB332 palette instantly
        colorPalette.colorCount = 256;
        for (int i = 0; i < 256; ++i) {
            uint8_t r3 = (i >> 5) & 7;
            uint8_t g3 = (i >> 2) & 7;
            uint8_t b2 = i & 3;
            uint8_t r = (r3 * 255) / 7;
            uint8_t g = (g3 * 255) / 7;
            uint8_t b = (b2 * 255) / 3;
            colorPalette.colors[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        // Multithreaded Pass
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 8;
        std::vector<std::thread> threads;
        
        std::atomic<uint32_t>* topGridAtomic = reinterpret_cast<std::atomic<uint32_t>*>(topLevelGrid);
        std::atomic<uint8_t>* poolAtomic = reinterpret_cast<std::atomic<uint8_t>*>(brickPool);
        
        for (unsigned int t = 0; t < numThreads; ++t) {
            threads.push_back(std::thread([&, t]() {
                size_t startIdx = (triangles.size() * t) / numThreads;
                size_t endIdx = (triangles.size() * (t + 1)) / numThreads;

                for (size_t i = startIdx; i < endIdx; ++i) {
                    const auto& tri = triangles[i];
                    glm::vec3 v0 = tri.v0 * scale + offset;
                    glm::vec3 v1 = tri.v1 * scale + offset;
                    glm::vec3 v2 = tri.v2 * scale + offset;
                    
                    glm::vec3 triMin = glm::min(glm::min(v0, v1), v2);
                    glm::vec3 triMax = glm::max(glm::max(v0, v1), v2);
                    
                    int minX = glm::max(0, (int)floor(triMin.x));
                    int minY = glm::max(0, (int)floor(triMin.y));
                    int minZ = glm::max(0, (int)floor(triMin.z));
                    int maxX = glm::min(worldSize - 1, (int)ceil(triMax.x));
                    int maxY = glm::min(worldSize - 1, (int)ceil(triMax.y));
                    int maxZ = glm::min(worldSize - 1, (int)ceil(triMax.z));
                    
                    for (int z = minZ; z <= maxZ; ++z) {
                        for (int y = minY; y <= maxY; ++y) {
                            for (int x = minX; x <= maxX; ++x) {
                                glm::vec3 voxelCenter(x + 0.5f, y + 0.5f, z + 0.5f);
                                
                                if (isPointInTriangle(voxelCenter, v0, v1, v2)) {
                                    glm::vec3 baryCoords = calculateBarycentricCoords(voxelCenter, v0, v1, v2);
                                    glm::vec2 interpolatedUV = tri.uv0 * baryCoords.x + tri.uv1 * baryCoords.y + tri.uv2 * baryCoords.z;
                                    
                                    glm::vec3 sampledColor = textureManager.sampleTexture(tri.textureIndex, interpolatedUV);
                                    if (sampledColor.x < 0.0f) {
                                        sampledColor = tri.color;
                                    }
                                    
                                    uint8_t r3 = (uint8_t)(glm::clamp(sampledColor.r * 7.0f + 0.5f, 0.0f, 7.0f));
                                    uint8_t g3 = (uint8_t)(glm::clamp(sampledColor.g * 7.0f + 0.5f, 0.0f, 7.0f));
                                    uint8_t b2 = (uint8_t)(glm::clamp(sampledColor.b * 3.0f + 0.5f, 0.0f, 3.0f));
                                    uint8_t paletteIndex = (r3 << 5) | (g3 << 2) | b2;
                                    if (paletteIndex == 0) paletteIndex = 1; // Avoid transparent
                                    
                                    // ATOMIC ALLOCATOR LOGIC
                                    size_t topIndex = getTopLevelIndex(x, y, z);
                                    uint32_t topEntry = topGridAtomic[topIndex].load(std::memory_order_relaxed);
                                    
                                    while ((topEntry & IS_BRICK_FLAG) == 0) {
                                        // Need to allocate a sub-voxel brick
                                        uint32_t newBrick = allocateBrickAtomic();
                                        if (newBrick >= maxBrickPoolSize) break; // Out of memory
                                        uint32_t newVal = IS_BRICK_FLAG | newBrick;
                                        if (topGridAtomic[topIndex].compare_exchange_strong(topEntry, newVal, std::memory_order_relaxed)) {
                                            topEntry = newVal;
                                            break;
                                        }
                                        // If failed, topEntry is automatically updated to the new value by compare_exchange_strong! Loop again!
                                    }
                                    
                                    if ((topEntry & IS_BRICK_FLAG) != 0) {
                                        uint32_t brickIndex = topEntry & BRICK_INDEX_MASK;
                                        if (brickIndex < maxBrickPoolSize) {
                                            int localX = x % brickSize;
                                            int localY = y % brickSize;
                                            int localZ = z % brickSize;
                                            size_t voxelOffset = localX + localY * brickSize + localZ * brickSize * brickSize;
                                            poolAtomic[brickIndex * 512 + voxelOffset].store(paletteIndex, std::memory_order_relaxed);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }));
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        updateAllSectors();
        printf("✅ Multithreaded voxelization complete! Allocated %zu bricks.\\n", currentBrickPoolSize.load());
        
        // Skip the legacy function immediately to prevent double processing
        return;
    }

    void voxelizeTrianglesLegacy(const std::vector<Triangle>& triangles) {'''

code = code.replace(old_voxelize, new_voxelize)


with open('example/compute.h', 'w') as f:
    f.write(code)

print("Injected Atomic Allocator and Multithreading!")
