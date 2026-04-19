#include "mesh_voxelizer.h"

#include <cmath>
#include <thread>

bool VoxelCompute::isPointInTriangle(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 v0p = p - v0;

    float dot00 = glm::dot(v0v2, v0v2);
    float dot01 = glm::dot(v0v2, v0v1);
    float dot02 = glm::dot(v0v2, v0p);
    float dot11 = glm::dot(v0v1, v0v1);
    float dot12 = glm::dot(v0v1, v0p);

    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

glm::vec3 VoxelCompute::calculateBarycentricCoords(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 v0p = p - v0;

    float dot00 = glm::dot(v0v2, v0v2);
    float dot01 = glm::dot(v0v2, v0v1);
    float dot02 = glm::dot(v0v2, v0p);
    float dot11 = glm::dot(v0v1, v0v1);
    float dot12 = glm::dot(v0v1, v0p);

    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    float w = 1.0f - u - v;

    return glm::vec3(w, v, u);
}

void VoxelCompute::voxelizeTriangles(const std::vector<Triangle>& triangles) {
    if (triangles.empty()) {
        printf("No triangles to voxelize!\n");
        return;
    }

    printf("🚀 Voxelizing %zu triangles using HIGH-SPEED ATOMIC ALLOCATOR across all CPU cores...\n", triangles.size());

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

    std::set<uint32_t> uniqueColors;
    for (const auto& tri : triangles) {
        uint8_t mr = (uint8_t)(glm::clamp(tri.color.r * 255.0f, 0.0f, 255.0f));
        uint8_t mg = (uint8_t)(glm::clamp(tri.color.g * 255.0f, 0.0f, 255.0f));
        uint8_t mb = (uint8_t)(glm::clamp(tri.color.b * 255.0f, 0.0f, 255.0f));
        uniqueColors.insert(((uint32_t)mr << 16) | ((uint32_t)mg << 8) | mb);

        if (tri.textureIndex > 0) {
            for (float u = 0.1f; u < 1.0f; u += 0.2f) {
                for (float v = 0.1f; v < 1.0f - u; v += 0.2f) {
                    float w = 1.0f - u - v;
                    glm::vec2 interpolatedUV = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
                    glm::vec3 sampledColor = textureManager.sampleTexture(tri.textureIndex, interpolatedUV);
                    if (sampledColor.x >= 0.0f) {
                        uint8_t r = (uint8_t)(glm::clamp(sampledColor.r * 255.0f, 0.0f, 255.0f));
                        uint8_t g = (uint8_t)(glm::clamp(sampledColor.g * 255.0f, 0.0f, 255.0f));
                        uint8_t b = (uint8_t)(glm::clamp(sampledColor.b * 255.0f, 0.0f, 255.0f));
                        uniqueColors.insert(((uint32_t)r << 16) | ((uint32_t)g << 8) | b);
                    }
                }
            }
        }
    }
    buildColorPalette(uniqueColors);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 8;
    std::vector<std::thread> threads;

    std::atomic<uint32_t>* topGridAtomic = reinterpret_cast<std::atomic<uint32_t>*>(topLevelGrid);
    std::atomic<uint16_t>* poolAtomic = reinterpret_cast<std::atomic<uint16_t>*>(brickPool);

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

                                uint8_t r = (uint8_t)(glm::clamp(sampledColor.r * 255.0f, 0.0f, 255.0f));
                                uint8_t g = (uint8_t)(glm::clamp(sampledColor.g * 255.0f, 0.0f, 255.0f));
                                uint8_t b = (uint8_t)(glm::clamp(sampledColor.b * 255.0f, 0.0f, 255.0f));
                                uint16_t paletteIndex = findClosestPaletteIndex(((uint32_t)r << 16) | ((uint32_t)g << 8) | b);
                                if (paletteIndex == 0) paletteIndex = 1;

                                size_t topIndex = getTopLevelIndex(x, y, z);
                                uint32_t topEntry = topGridAtomic[topIndex].load(std::memory_order_relaxed);

                                while ((topEntry & IS_BRICK_FLAG) == 0) {
                                    uint32_t newBrick = allocateBrickAtomic();
                                    if (newBrick >= maxBrickPoolSize) break;
                                    uint32_t newVal = IS_BRICK_FLAG | newBrick;
                                    if (topGridAtomic[topIndex].compare_exchange_strong(topEntry, newVal, std::memory_order_relaxed)) {
                                        topEntry = newVal;
                                        break;
                                    }
                                }

                                if ((topEntry & IS_BRICK_FLAG) != 0) {
                                    uint32_t brickIndex = topEntry & BRICK_INDEX_MASK;
                                    if (brickIndex < maxBrickPoolSize) {
                                        int localX = x % brickSize;
                                        int localY = y % brickSize;
                                        int localZ = z % brickSize;
                                        size_t voxelOffset = getBrickVoxelOffset(localX, localY, localZ);
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

    size_t allocatedCount = currentBrickPoolSize.load();
    std::vector<std::thread> recountThreads;
    for (unsigned int t = 0; t < numThreads; ++t) {
        recountThreads.push_back(std::thread([&, t]() {
            size_t startIdx = (allocatedCount * t) / numThreads;
            size_t endIdx = (allocatedCount * (t + 1)) / numThreads;
            for (size_t i = startIdx; i < endIdx; ++i) {
                if (brickMetadata[i].inUse) {
                    uint16_t count = 0;
                    for (int j = 0; j < 512; ++j) {
                        if (brickPool[i * 512 + j] != 0) count++;
                    }
                    brickMetadata[i].nonEmptyCount = count;
                }
            }
        }));
    }
    for (auto& rt : recountThreads) {
        rt.join();
    }

    updateAllSectors();
    printf("✅ Multithreaded voxelization complete! Allocated %zu bricks.\n", currentBrickPoolSize.load());
}

void VoxelCompute::voxelizeTrianglesLegacy(const std::vector<Triangle>& triangles) {
    if (triangles.empty()) {
        printf("No triangles to voxelize!\n");
        return;
    }

    printf("Voxelizing %zu triangles into hierarchical structure...\n", triangles.size());

    for (size_t i = 0; i < std::min(size_t(5), triangles.size()); ++i) {
        printf("🔍 VOXEL TRIANGLE %zu: textureIndex=%u, color(%.3f,%.3f,%.3f)\n",
               i, triangles[i].textureIndex, triangles[i].color.r, triangles[i].color.g, triangles[i].color.b);
    }

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

    printf("Scaling by %f, offset by (%f, %f, %f)\n", scale, offset.x, offset.y, offset.z);

    std::set<uint32_t> uniqueColors;
    for (const auto& tri : triangles) {
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

                        uint8_t r = (uint8_t)(glm::clamp(sampledColor.r * 255.0f, 0.0f, 255.0f));
                        uint8_t g = (uint8_t)(glm::clamp(sampledColor.g * 255.0f, 0.0f, 255.0f));
                        uint8_t b = (uint8_t)(glm::clamp(sampledColor.b * 255.0f, 0.0f, 255.0f));

                        uint32_t rgb = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
                        uniqueColors.insert(rgb);
                    }
                }
            }
        }
    }

    buildColorPalette(uniqueColors);

    printf("🎨 Second pass: voxelizing with palette indices...\n");
    for (const auto& tri : triangles) {
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

                        uint8_t r = (uint8_t)(glm::clamp(sampledColor.r * 255.0f, 0.0f, 255.0f));
                        uint8_t g = (uint8_t)(glm::clamp(sampledColor.g * 255.0f, 0.0f, 255.0f));
                        uint8_t b = (uint8_t)(glm::clamp(sampledColor.b * 255.0f, 0.0f, 255.0f));

                        uint32_t rgb = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
                        uint16_t paletteIdx = findClosestPaletteIndex(rgb);

                        setVoxel(x, y, z, paletteIdx);
                    }
                }
            }
        }
    }

    printf("✅ Voxelization complete with %d-color palette!\n", colorPalette.colorCount);
}

namespace compute_layers {

void voxelizeTriangles(VoxelCompute& volume, const std::vector<Triangle>& triangles) {
    volume.voxelizeTriangles(triangles);
}

void voxelizeTrianglesLegacy(VoxelCompute& volume, const std::vector<Triangle>& triangles) {
    volume.voxelizeTrianglesLegacy(triangles);
}

} // namespace compute_layers
