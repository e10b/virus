#include "terrain_generator.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <glm/gtc/noise.hpp>

uint32_t VoxelCompute::packRGB(uint8_t r, uint8_t g, uint8_t b) {
    return (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
}

float VoxelCompute::fbm2D(glm::vec2 pos, int octaves, float lacunarity, float gain) const {
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float sum = 0.0f;
    float normalizer = 0.0f;

    for (int octave = 0; octave < octaves; ++octave) {
        sum += glm::simplex(pos * frequency) * amplitude;
        normalizer += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return normalizer > 0.0f ? sum / normalizer : 0.0f;
}

float VoxelCompute::ridgeFbm2D(glm::vec2 pos, int octaves, float lacunarity, float gain) const {
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float sum = 0.0f;
    float normalizer = 0.0f;

    for (int octave = 0; octave < octaves; ++octave) {
        float n = 1.0f - glm::abs(glm::simplex(pos * frequency));
        n *= n;
        sum += n * amplitude;
        normalizer += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return normalizer > 0.0f ? sum / normalizer : 0.0f;
}

float VoxelCompute::fbm3D(glm::vec3 pos, int octaves, float lacunarity, float gain) const {
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float sum = 0.0f;
    float normalizer = 0.0f;

    for (int octave = 0; octave < octaves; ++octave) {
        sum += glm::simplex(pos * frequency) * amplitude;
        normalizer += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return normalizer > 0.0f ? sum / normalizer : 0.0f;
}

int VoxelCompute::sampleTerrainHeight(int x, int z) const {
    glm::vec2 pos{float(x), float(z)};
    
    // Multi-octave Perlin noise (Minecraft-style) for realistic terrain
    // Octave 1: Large-scale terrain features (1/512 frequency)
    float octave1 = fbm2D(pos * 0.00195f, 2, 2.0f, 0.5f);
    
    // Octave 2: Medium-scale variation (1/256 frequency)
    float octave2 = fbm2D(pos * 0.00391f, 3, 2.0f, 0.5f);
    
    // Octave 3: Small-scale detail (1/128 frequency)
    float octave3 = fbm2D(pos * 0.00781f, 3, 2.0f, 0.5f);
    
    // Combine with proper amplitude scaling - increased variation
    float terrainNoise = (octave1 * 50.0f + octave2 * 40.0f + octave3 * 20.0f) / 110.0f;
    
    // Biome selector: low-frequency noise determines region type
    float biomeNoise = fbm2D(pos * 0.0002f, 3, 2.0f, 0.5f);
    
    // Biome masks - adjust ranges for better distribution
    float plainsMask = glm::smoothstep(-0.7f, -0.1f, biomeNoise);        // More plains
    float hillMask = glm::smoothstep(-0.4f, 0.3f, biomeNoise);            // Rolling hills
    float mountainMask = glm::smoothstep(0.25f, 0.9f, biomeNoise);       // Mountains
    
    // Normalize masks
    float totalMask = plainsMask + hillMask + mountainMask;
    if (totalMask > 0.01f) {
        plainsMask /= totalMask;
        hillMask /= totalMask;
        mountainMask /= totalMask;
    }
    
    // Base elevations and terrain scales per biome - more variation
    float plainsHeight = 80.0f + terrainNoise * 60.0f;                    // 20-140 voxels
    float hillHeight = 150.0f + terrainNoise * 200.0f;                    // -50-350 voxels (clamped)
    float mountainHeight = 280.0f + terrainNoise * 600.0f;                // -320-880 voxels (clamped)
    
    // Blend biome heights
    float height = plainsMask * plainsHeight + 
                   hillMask * hillHeight + 
                   mountainMask * mountainHeight;

    return std::max(6, static_cast<int>(height));
}

uint16_t VoxelCompute::chooseTerrainMaterial(
    int x,
    int y,
    int z,
    int surfaceY,
    int dx,
    int dz,
    int minTerrainHeight,
    int maxTerrainHeight) const {
    
    int slope = glm::clamp(glm::abs(dx) + glm::abs(dz), 0, 64);
    float slopeNorm = float(slope) / 64.0f;
    float elevation = float(surfaceY);
    
    // Lower frequency = larger, more cohesive patches
    float grassVar = 0.5f + 0.5f * glm::simplex(glm::vec2(float(x), float(z)) * 0.01f + glm::vec2(17.0f, 93.0f));
    float stoneVar = 0.5f + 0.5f * glm::simplex(glm::vec2(float(x), float(z)) * 0.02f + glm::vec2(43.0f, 71.0f));
    
    int depthFromSurface = surfaceY - y;
    
    // === SURFACE TOP LAYER (depthFromSurface == 0, the actual surface) ===
    if (depthFromSurface == 0) {
        // Snow peaks (elevation > 400)
        if (elevation > 400.0f) {
            return 5;  // Snow
        }
        // High mountains (250-400): mostly stone/rock
        else if (elevation > 250.0f) {
            if (slopeNorm > 0.6f) {
                return 6;  // Dark rock on steep faces
            } else {
                // Mountain slopes mostly grass with occasional moss
                return grassVar > 0.4f ? 2 : 7;
            }
        }
        // Mid-elevation foothills (120-250): GREEN
        else if (elevation > 120.0f) {
            if (slopeNorm > 0.7f) {
                return 4;  // Stone only on very steep cliffs
            }
            // Heavily favor grass - occasional moss for texture
            return grassVar > 0.3f ? 2 : 7;
        }
        // Lowlands (< 120): LUSH GREEN
        else {
            if (slopeNorm > 0.75f) {
                return 3;  // Dirt only on extreme slopes
            }
            // Almost all grass with rare moss patches
            return grassVar > 0.15f ? 2 : 7;
        }
    }
    
    // === LAYER BELOW SURFACE (1-3 blocks down) ===
    // Put dirt below grass, or stone in high rocky areas
    else if (depthFromSurface >= 1 && depthFromSurface <= 3) {
        if (elevation > 250.0f) {
            // Rocky high mountains: stone
            return stoneVar > 0.5f ? 4 : 6;
        } else {
            // Dirt below grassy terrain
            return 3;
        }
    }
    
    // === DEEP LAYER (4+ blocks down) ===
    // Stone all the way down
    else {
        return stoneVar > 0.5f ? 4 : 6;
    }
}

void VoxelCompute::initializeTerrainStreaming() {
    residentGlobalToLocal.clear();
    residentLocalToGlobal.clear();
    generatedSectorVoxelData.clear();
    generatedSectorNonAirCounts.clear();
    pendingSectorBrickFlush.clear();
    sectorCandidates.clear();
    sphereMask.clear();
    lastCachedCenterSectorX = INT32_MIN;
    lastCachedCenterSectorY = INT32_MIN;
    lastCachedCenterSectorZ = INT32_MIN;
    streamAnchorSectorX = INT32_MIN;
    streamAnchorSectorY = INT32_MIN;
    streamAnchorSectorZ = INT32_MIN;
    cachedSphereMaskRadius = -1;
    sectorCandidateCursor = 0;
    terrainStreamedSectorCount = 0;

    sectorBoundsData[0] = 0;
    sectorBoundsData[1] = 0;
    sectorBoundsData[2] = 0;
    sectorBoundsData[3] = 0;
    sectorBoundsData[4] = 0;
    sectorBoundsData[5] = 0;
    sectorBoundsData[6] = 0;
    sectorBoundsData[7] = 0;
}

void VoxelCompute::generateSphereMask() {
    const int maxNoAliasRadius = std::max(1, (sectorsPerDimension - 1) / 2);
    const int radius = std::min(std::max(1, terrainStreamRadiusSectors), maxNoAliasRadius);
    sphereMask.clear();
    sphereMask.reserve(static_cast<size_t>((radius * 2 + 1) * (radius * 2 + 1) * (radius * 2 + 1)));

    const int radiusSq = radius * radius;
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int distSq = dx * dx + dy * dy + dz * dz;
                if (distSq > radiusSq) {
                    continue;
                }

                SectorCandidate candidate;
                candidate.x = dx;
                candidate.y = dy;
                candidate.z = dz;
                candidate.distSq = distSq;
                sphereMask.push_back(candidate);
            }
        }
    }

    std::sort(
        sphereMask.begin(),
        sphereMask.end(),
        [](const SectorCandidate& a, const SectorCandidate& b) {
            return a.distSq < b.distSq;
        });
    cachedSphereMaskRadius = radius;
}

void VoxelCompute::rebuildSectorCandidateCache(int centerSectorX, int centerSectorY, int centerSectorZ) {
    const int maxNoAliasRadius = std::max(1, (sectorsPerDimension - 1) / 2);
    const int effectiveRadius = std::min(std::max(1, terrainStreamRadiusSectors), maxNoAliasRadius);
    if (cachedSphereMaskRadius != effectiveRadius || sphereMask.empty()) {
        generateSphereMask();
    }

    sectorCandidates.clear();
    std::unordered_map<uint64_t, SectorCandidate> bestPerLocalSlot;
    bestPerLocalSlot.reserve(static_cast<size_t>(sectorCount));

    for (const SectorCandidate& offset : sphereMask) {
        SectorCandidate candidate;
        candidate.x = centerSectorX + offset.x;
        candidate.y = centerSectorY + offset.y;
        candidate.z = centerSectorZ + offset.z;
        candidate.distSq = offset.distSq;

        const int localSectorX = wrapSectorCoord(candidate.x);
        const int localSectorY = wrapSectorCoord(candidate.y);
        const int localSectorZ = wrapSectorCoord(candidate.z);
        const uint64_t localKey = packLocalSectorKey(localSectorX, localSectorY, localSectorZ);

        // sphereMask is already sorted nearest-first, so first writer wins per local slot.
        if (bestPerLocalSlot.find(localKey) == bestPerLocalSlot.end()) {
            bestPerLocalSlot.emplace(localKey, candidate);
            if (bestPerLocalSlot.size() >= static_cast<size_t>(sectorCount)) {
                break;
            }
        }
    }

    sectorCandidates.reserve(bestPerLocalSlot.size());
    for (const auto& entry : bestPerLocalSlot) {
        sectorCandidates.push_back(entry.second);
    }
    std::sort(
        sectorCandidates.begin(),
        sectorCandidates.end(),
        [](const SectorCandidate& a, const SectorCandidate& b) {
            return a.distSq < b.distSq;
        });

    lastCachedCenterSectorX = centerSectorX;
    lastCachedCenterSectorY = centerSectorY;
    lastCachedCenterSectorZ = centerSectorZ;
    sectorCandidateCursor = 0;
}

void VoxelCompute::updateResidentSectorBounds() {
    int minX = INT_MAX;
    int minY = INT_MAX;
    int minZ = INT_MAX;
    int maxX = INT_MIN;
    int maxY = INT_MIN;
    int maxZ = INT_MIN;
    bool hasResident = false;

    for (int i = 0; i < sectorCount; ++i) {
        if (sectorCoordMap[i * 4 + 3] == 0) {
            continue;
        }

        const int sx = sectorCoordMap[i * 4 + 0];
        const int sy = sectorCoordMap[i * 4 + 1];
        const int sz = sectorCoordMap[i * 4 + 2];

        minX = std::min(minX, sx);
        minY = std::min(minY, sy);
        minZ = std::min(minZ, sz);
        maxX = std::max(maxX, sx);
        maxY = std::max(maxY, sy);
        maxZ = std::max(maxZ, sz);
        hasResident = true;
    }

    if (!hasResident) {
        sectorBoundsData[0] = 0;
        sectorBoundsData[1] = 0;
        sectorBoundsData[2] = 0;
        sectorBoundsData[3] = 0;
        sectorBoundsData[4] = 0;
        sectorBoundsData[5] = 0;
        sectorBoundsData[6] = 0;
        sectorBoundsData[7] = 0;
    } else {
        sectorBoundsData[0] = minX;
        sectorBoundsData[1] = minY;
        sectorBoundsData[2] = minZ;
        sectorBoundsData[3] = 1;
        sectorBoundsData[4] = maxX;
        sectorBoundsData[5] = maxY;
        sectorBoundsData[6] = maxZ;
        sectorBoundsData[7] = 0;
    }

    if (compute && sectorBoundsUniform) {
        compute->updateStorageBuffer(sectorBoundsUniform, sectorBoundsData, 8 * sizeof(int32_t), 0);
    }
}

std::string VoxelCompute::makeSectorKey(int sectorX, int sectorY, int sectorZ) const {
    return std::to_string(sectorX) + ":" + std::to_string(sectorY) + ":" + std::to_string(sectorZ);
}

uint32_t VoxelCompute::getOrComputeSectorNonAirCount(const std::string& sectorKey) const {
    const auto countIt = generatedSectorNonAirCounts.find(sectorKey);
    if (countIt != generatedSectorNonAirCounts.end()) {
        return countIt->second;
    }

    const auto voxelsIt = generatedSectorVoxelData.find(sectorKey);
    if (voxelsIt == generatedSectorVoxelData.end()) {
        return 0u;
    }

    uint32_t nonAirCount = 0u;
    for (const uint16_t voxel : voxelsIt->second) {
        if (voxel != 0u) {
            ++nonAirCount;
        }
    }

    generatedSectorNonAirCounts[sectorKey] = nonAirCount;
    return nonAirCount;
}

bool VoxelCompute::shouldDiscardSectorGpuUpload(int globalSectorX, int globalSectorY, int globalSectorZ) const {
    const std::string sectorKey = makeSectorKey(globalSectorX, globalSectorY, globalSectorZ);
    if (getOrComputeSectorNonAirCount(sectorKey) != 0u) {
        return false;
    }

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }

                const std::string neighborKey = makeSectorKey(
                    globalSectorX + dx,
                    globalSectorY + dy,
                    globalSectorZ + dz);

                const auto neighborIt = generatedSectorNonAirCounts.find(neighborKey);
                if (neighborIt == generatedSectorNonAirCounts.end()) {
                    return false;
                }
                if (neighborIt->second != 0u) {
                    return false;
                }
            }
        }
    }

    return true;
}

uint64_t VoxelCompute::packLocalSectorKey(int localSectorX, int localSectorY, int localSectorZ) const {
    return static_cast<uint64_t>(localSectorX) |
           (static_cast<uint64_t>(localSectorY) << 16u) |
           (static_cast<uint64_t>(localSectorZ) << 32u);
}

int VoxelCompute::wrapSectorCoord(int sectorCoord) const {
    int wrapped = sectorCoord % sectorsPerDimension;
    if (wrapped < 0) {
        wrapped += sectorsPerDimension;
    }
    return wrapped;
}

bool VoxelCompute::isSectorInCameraFrustum(
    int globalSectorX,
    int globalSectorY,
    int globalSectorZ,
    const glm::mat4& viewProj,
    const glm::vec3& cameraPos) const {
    const float minX = static_cast<float>(globalSectorX * sectorSize);
    const float minY = static_cast<float>(globalSectorY * sectorSize);
    const float minZ = static_cast<float>(globalSectorZ * sectorSize);
    const float maxX = minX + static_cast<float>(sectorSize);
    const float maxY = minY + static_cast<float>(sectorSize);
    const float maxZ = minZ + static_cast<float>(sectorSize);

    glm::vec3 corners[8] = {
        {minX, minY, minZ}, {maxX, minY, minZ}, {minX, maxY, minZ}, {maxX, maxY, minZ},
        {minX, minY, maxZ}, {maxX, minY, maxZ}, {minX, maxY, maxZ}, {maxX, maxY, maxZ}
    };

    uint8_t outsideLeft = 0;
    uint8_t outsideRight = 0;
    uint8_t outsideBottom = 0;
    uint8_t outsideTop = 0;
    uint8_t outsideNear = 0;
    uint8_t outsideFar = 0;

    for (int i = 0; i < 8; ++i) {
        const glm::vec3 relativeCorner = corners[i] - cameraPos;
        const glm::vec4 clip = viewProj * glm::vec4(relativeCorner, 1.0f);

        if (clip.x < -clip.w) {
            ++outsideLeft;
        }
        if (clip.x > clip.w) {
            ++outsideRight;
        }
        if (clip.y < -clip.w) {
            ++outsideBottom;
        }
        if (clip.y > clip.w) {
            ++outsideTop;
        }
        if (clip.z < 0.0f) {
            ++outsideNear;
        }
        if (clip.z > clip.w) {
            ++outsideFar;
        }
    }

    return outsideLeft < 8 &&
           outsideRight < 8 &&
           outsideBottom < 8 &&
           outsideTop < 8 &&
           outsideNear < 8 &&
           outsideFar < 8;
}

bool VoxelCompute::getSectorFrustumVisibility(
    int globalSectorX,
    int globalSectorY,
    int globalSectorZ,
    const glm::mat4& viewProj,
    const glm::vec3& cameraPos,
    bool refreshCache) {
    const std::string sectorKey = makeSectorKey(globalSectorX, globalSectorY, globalSectorZ);

    if (!refreshCache) {
        const auto cached = sectorFrustumVisibilityCache.find(sectorKey);
        if (cached != sectorFrustumVisibilityCache.end()) {
            return cached->second;
        }
    }

    const bool isVisible = isSectorInCameraFrustum(
        globalSectorX,
        globalSectorY,
        globalSectorZ,
        viewProj,
        cameraPos);
    sectorFrustumVisibilityCache[sectorKey] = isVisible;
    return isVisible;
}

void VoxelCompute::rebuildOcclusionHiZ(const glm::mat4& viewProj, const glm::vec3& cameraPos) {
    const int width = std::max(1, hiZBaseWidth);
    const int height = std::max(1, hiZBaseHeight);
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    hiZBaseDepth.assign(pixelCount, 1.0f);

    auto projectSector = [&](int sx, int sy, int sz, int& pxMin, int& pyMin, int& pxMax, int& pyMax, float& depthMin, float& depthMax) {
        const float minX = static_cast<float>(sx * sectorSize);
        const float minY = static_cast<float>(sy * sectorSize);
        const float minZ = static_cast<float>(sz * sectorSize);
        const float maxX = minX + static_cast<float>(sectorSize);
        const float maxY = minY + static_cast<float>(sectorSize);
        const float maxZ = minZ + static_cast<float>(sectorSize);

        const glm::vec3 corners[8] = {
            {minX, minY, minZ}, {maxX, minY, minZ}, {minX, maxY, minZ}, {maxX, maxY, minZ},
            {minX, minY, maxZ}, {maxX, minY, maxZ}, {minX, maxY, maxZ}, {maxX, maxY, maxZ}
        };

        float ndcXMin = 1.0f;
        float ndcYMin = 1.0f;
        float ndcXMax = -1.0f;
        float ndcYMax = -1.0f;
        depthMin = 1.0f;
        depthMax = 0.0f;
        bool anyProjected = false;

        for (const glm::vec3& corner : corners) {
            const glm::vec4 clip = viewProj * glm::vec4(corner - cameraPos, 1.0f);
            if (clip.w <= 0.0f) {
                continue;
            }

            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            ndcXMin = std::min(ndcXMin, ndc.x);
            ndcYMin = std::min(ndcYMin, ndc.y);
            ndcXMax = std::max(ndcXMax, ndc.x);
            ndcYMax = std::max(ndcYMax, ndc.y);
            depthMin = std::min(depthMin, ndc.z);
            depthMax = std::max(depthMax, ndc.z);
            anyProjected = true;
        }

        if (!anyProjected) {
            return false;
        }

        ndcXMin = glm::clamp(ndcXMin, -1.0f, 1.0f);
        ndcYMin = glm::clamp(ndcYMin, -1.0f, 1.0f);
        ndcXMax = glm::clamp(ndcXMax, -1.0f, 1.0f);
        ndcYMax = glm::clamp(ndcYMax, -1.0f, 1.0f);

        if (ndcXMin >= ndcXMax || ndcYMin >= ndcYMax) {
            return false;
        }

        pxMin = glm::clamp(static_cast<int>((ndcXMin * 0.5f + 0.5f) * static_cast<float>(width)), 0, width - 1);
        pxMax = glm::clamp(static_cast<int>((ndcXMax * 0.5f + 0.5f) * static_cast<float>(width)), 0, width - 1);
        pyMin = glm::clamp(static_cast<int>((1.0f - (ndcYMax * 0.5f + 0.5f)) * static_cast<float>(height)), 0, height - 1);
        pyMax = glm::clamp(static_cast<int>((1.0f - (ndcYMin * 0.5f + 0.5f)) * static_cast<float>(height)), 0, height - 1);

        if (pxMin > pxMax || pyMin > pyMax) {
            return false;
        }

        depthMin = glm::clamp(depthMin, 0.0f, 1.0f);
        depthMax = glm::clamp(depthMax, 0.0f, 1.0f);
        return true;
    };

    const float sectorVoxelCount = static_cast<float>(sectorSize * sectorSize * sectorSize);
    for (const auto& residentEntry : residentLocalToGlobal) {
        const uint64_t localKey = residentEntry.first;
        const int localSectorX = static_cast<int>(localKey & 0xFFFFu);
        const int localSectorY = static_cast<int>((localKey >> 16u) & 0xFFFFu);
        const int localSectorZ = static_cast<int>((localKey >> 32u) & 0xFFFFu);
        const int localSectorIndex =
            localSectorX +
            localSectorY * sectorsPerDimension +
            localSectorZ * sectorsPerDimension * sectorsPerDimension;

        if (sectorCoordMap[localSectorIndex * 4 + 3] == 0) {
            continue;
        }

        const int globalSectorX = sectorCoordMap[localSectorIndex * 4 + 0];
        const int globalSectorY = sectorCoordMap[localSectorIndex * 4 + 1];
        const int globalSectorZ = sectorCoordMap[localSectorIndex * 4 + 2];

        const std::string sectorKey = residentEntry.second;
        const uint32_t nonAirCount = getOrComputeSectorNonAirCount(sectorKey);
        if (nonAirCount == 0u) {
            continue;
        }

        const float fillRatio = static_cast<float>(nonAirCount) / sectorVoxelCount;
        if (fillRatio < hiZOccluderFillRatioThreshold) {
            continue;
        }

        if (!isSectorInCameraFrustum(globalSectorX, globalSectorY, globalSectorZ, viewProj, cameraPos)) {
            continue;
        }

        int pxMin = 0;
        int pyMin = 0;
        int pxMax = 0;
        int pyMax = 0;
        float depthMin = 1.0f;
        float depthMax = 0.0f;
        if (!projectSector(globalSectorX, globalSectorY, globalSectorZ, pxMin, pyMin, pxMax, pyMax, depthMin, depthMax)) {
            continue;
        }

        for (int py = pyMin; py <= pyMax; ++py) {
            const size_t row = static_cast<size_t>(py) * static_cast<size_t>(width);
            for (int px = pxMin; px <= pxMax; ++px) {
                const size_t index = row + static_cast<size_t>(px);
                hiZBaseDepth[index] = std::min(hiZBaseDepth[index], depthMax);
            }
        }
    }

    hiZMips.clear();
    hiZMips.push_back(hiZBaseDepth);

    int currentWidth = width;
    int currentHeight = height;
    while (currentWidth > 1 || currentHeight > 1) {
        const std::vector<float>& prev = hiZMips.back();
        const int nextWidth = std::max(1, (currentWidth + 1) / 2);
        const int nextHeight = std::max(1, (currentHeight + 1) / 2);
        std::vector<float> next(static_cast<size_t>(nextWidth) * static_cast<size_t>(nextHeight), 1.0f);

        for (int y = 0; y < nextHeight; ++y) {
            for (int x = 0; x < nextWidth; ++x) {
                float farthestDepth = 0.0f;
                for (int oy = 0; oy < 2; ++oy) {
                    for (int ox = 0; ox < 2; ++ox) {
                        const int sx = x * 2 + ox;
                        const int sy = y * 2 + oy;
                        if (sx >= currentWidth || sy >= currentHeight) {
                            continue;
                        }
                        const size_t sampleIndex =
                            static_cast<size_t>(sy) * static_cast<size_t>(currentWidth) +
                            static_cast<size_t>(sx);
                        farthestDepth = std::max(farthestDepth, prev[sampleIndex]);
                    }
                }

                const size_t outIndex =
                    static_cast<size_t>(y) * static_cast<size_t>(nextWidth) +
                    static_cast<size_t>(x);
                next[outIndex] = farthestDepth;
            }
        }

        hiZMips.push_back(std::move(next));
        currentWidth = nextWidth;
        currentHeight = nextHeight;
    }
}

bool VoxelCompute::isSectorOccludedByHiZ(
    int globalSectorX,
    int globalSectorY,
    int globalSectorZ,
    const glm::mat4& viewProj,
    const glm::vec3& cameraPos) const {
    if (hiZMips.empty()) {
        return false;
    }

    const int width = std::max(1, hiZBaseWidth);
    const int height = std::max(1, hiZBaseHeight);

    const float minX = static_cast<float>(globalSectorX * sectorSize);
    const float minY = static_cast<float>(globalSectorY * sectorSize);
    const float minZ = static_cast<float>(globalSectorZ * sectorSize);
    const float maxX = minX + static_cast<float>(sectorSize);
    const float maxY = minY + static_cast<float>(sectorSize);
    const float maxZ = minZ + static_cast<float>(sectorSize);

    const glm::vec3 corners[8] = {
        {minX, minY, minZ}, {maxX, minY, minZ}, {minX, maxY, minZ}, {maxX, maxY, minZ},
        {minX, minY, maxZ}, {maxX, minY, maxZ}, {minX, maxY, maxZ}, {maxX, maxY, maxZ}
    };

    float ndcXMin = 1.0f;
    float ndcYMin = 1.0f;
    float ndcXMax = -1.0f;
    float ndcYMax = -1.0f;
    float sectorClosestDepth = 1.0f;
    bool anyProjected = false;

    for (const glm::vec3& corner : corners) {
        const glm::vec4 clip = viewProj * glm::vec4(corner - cameraPos, 1.0f);
        if (clip.w <= 0.0f) {
            continue;
        }

        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        ndcXMin = std::min(ndcXMin, ndc.x);
        ndcYMin = std::min(ndcYMin, ndc.y);
        ndcXMax = std::max(ndcXMax, ndc.x);
        ndcYMax = std::max(ndcYMax, ndc.y);
        sectorClosestDepth = std::min(sectorClosestDepth, ndc.z);
        anyProjected = true;
    }

    if (!anyProjected) {
        return false;
    }

    ndcXMin = glm::clamp(ndcXMin, -1.0f, 1.0f);
    ndcYMin = glm::clamp(ndcYMin, -1.0f, 1.0f);
    ndcXMax = glm::clamp(ndcXMax, -1.0f, 1.0f);
    ndcYMax = glm::clamp(ndcYMax, -1.0f, 1.0f);
    sectorClosestDepth = glm::clamp(sectorClosestDepth, 0.0f, 1.0f);

    if (ndcXMin >= ndcXMax || ndcYMin >= ndcYMax) {
        return false;
    }

    int pxMin = glm::clamp(static_cast<int>((ndcXMin * 0.5f + 0.5f) * static_cast<float>(width)), 0, width - 1);
    int pxMax = glm::clamp(static_cast<int>((ndcXMax * 0.5f + 0.5f) * static_cast<float>(width)), 0, width - 1);
    int pyMin = glm::clamp(static_cast<int>((1.0f - (ndcYMax * 0.5f + 0.5f)) * static_cast<float>(height)), 0, height - 1);
    int pyMax = glm::clamp(static_cast<int>((1.0f - (ndcYMin * 0.5f + 0.5f)) * static_cast<float>(height)), 0, height - 1);

    if (pxMin > pxMax || pyMin > pyMax) {
        return false;
    }

    int regionW = pxMax - pxMin + 1;
    int regionH = pyMax - pyMin + 1;
    int mipLevel = 0;
    int mipWidth = width;
    int mipHeight = height;
    while (mipLevel + 1 < static_cast<int>(hiZMips.size()) && regionW > 2 && regionH > 2) {
        ++mipLevel;
        pxMin /= 2;
        pxMax = std::max(pxMin, pxMax / 2);
        pyMin /= 2;
        pyMax = std::max(pyMin, pyMax / 2);
        mipWidth = std::max(1, (mipWidth + 1) / 2);
        mipHeight = std::max(1, (mipHeight + 1) / 2);
        regionW = pxMax - pxMin + 1;
        regionH = pyMax - pyMin + 1;
    }

    const std::vector<float>& mip = hiZMips[static_cast<size_t>(mipLevel)];
    pxMin = glm::clamp(pxMin, 0, mipWidth - 1);
    pxMax = glm::clamp(pxMax, 0, mipWidth - 1);
    pyMin = glm::clamp(pyMin, 0, mipHeight - 1);
    pyMax = glm::clamp(pyMax, 0, mipHeight - 1);

    float farthestOccluderDepth = 0.0f;
    for (int py = pyMin; py <= pyMax; ++py) {
        const size_t row = static_cast<size_t>(py) * static_cast<size_t>(mipWidth);
        for (int px = pxMin; px <= pxMax; ++px) {
            const size_t index = row + static_cast<size_t>(px);
            farthestOccluderDepth = std::max(farthestOccluderDepth, mip[index]);
        }
    }

    if (farthestOccluderDepth >= 1.0f) {
        return false;
    }

    return sectorClosestDepth > (farthestOccluderDepth + hiZOcclusionDepthBias);
}

void VoxelCompute::clearLocalSectorSlot(int localSectorX, int localSectorY, int localSectorZ) {
    const uint64_t localKey = packLocalSectorKey(localSectorX, localSectorY, localSectorZ);
    pendingSectorBrickFlush.erase(localKey);

    const int localSectorIndex =
        localSectorX +
        localSectorY * sectorsPerDimension +
        localSectorZ * sectorsPerDimension * sectorsPerDimension;

    const int topStartX = localSectorX * bricksPerSector;
    const int topStartY = localSectorY * bricksPerSector;
    const int topStartZ = localSectorZ * bricksPerSector;

    for (int localTopZ = 0; localTopZ < bricksPerSector; ++localTopZ) {
        for (int localTopY = 0; localTopY < bricksPerSector; ++localTopY) {
            for (int localTopX = 0; localTopX < bricksPerSector; ++localTopX) {
                const int topX = topStartX + localTopX;
                const int topY = topStartY + localTopY;
                const int topZ = topStartZ + localTopZ;
                if (topX < 0 || topX >= topLevelSize ||
                    topY < 0 || topY >= topLevelSize ||
                    topZ < 0 || topZ >= topLevelSize) {
                    continue;
                }

                const size_t topIndex = static_cast<size_t>(topX) +
                                        static_cast<size_t>(topY) * static_cast<size_t>(topLevelSize) +
                                        static_cast<size_t>(topZ) * static_cast<size_t>(topLevelSize) *
                                            static_cast<size_t>(topLevelSize);
                const uint32_t topEntry = topLevelGrid[topIndex];
                if ((topEntry & IS_BRICK_FLAG) != 0u) {
                    freeBrickInternal(topEntry & BRICK_INDEX_MASK);
                }
                topLevelGrid[topIndex] = 0u;
                queueTopLevelUpdate(topX * brickSize, topY * brickSize, topZ * brickSize);
            }
        }
    }

    sectorCoordMap[localSectorIndex * 4 + 3] = 0;
    if (compute && sectorCoordUniform) {
        compute->updateStorageBuffer(
            sectorCoordUniform,
            &sectorCoordMap[localSectorIndex * 4],
            4 * sizeof(int32_t),
            static_cast<size_t>(localSectorIndex) * 4 * sizeof(int32_t));
    }

    sectorMapDirty = true;
}

void VoxelCompute::ensureSectorResident(int globalSectorX, int globalSectorY, int globalSectorZ) {
    const std::string globalKey = makeSectorKey(globalSectorX, globalSectorY, globalSectorZ);
    if (residentGlobalToLocal.find(globalKey) != residentGlobalToLocal.end()) {
        return;
    }

    const int localSectorX = wrapSectorCoord(globalSectorX);
    const int localSectorY = wrapSectorCoord(globalSectorY);
    const int localSectorZ = wrapSectorCoord(globalSectorZ);
    const uint64_t localKey = packLocalSectorKey(localSectorX, localSectorY, localSectorZ);

    const auto occupied = residentLocalToGlobal.find(localKey);
    if (occupied != residentLocalToGlobal.end() && occupied->second != globalKey) {
        residentGlobalToLocal.erase(occupied->second);
        clearLocalSectorSlot(localSectorX, localSectorY, localSectorZ);
    }

    const int localSectorIndex =
        localSectorX +
        localSectorY * sectorsPerDimension +
        localSectorZ * sectorsPerDimension * sectorsPerDimension;
    if (localSectorX < 0 || localSectorX >= sectorsPerDimension ||
        localSectorY < 0 || localSectorY >= sectorsPerDimension ||
        localSectorZ < 0 || localSectorZ >= sectorsPerDimension ||
        localSectorIndex < 0 || localSectorIndex >= sectorCount) {
        return;
    }

    residentGlobalToLocal[globalKey] = localKey;
    residentLocalToGlobal[localKey] = globalKey;
    sectorCoordMap[localSectorIndex * 4 + 0] = globalSectorX;
    sectorCoordMap[localSectorIndex * 4 + 1] = globalSectorY;
    sectorCoordMap[localSectorIndex * 4 + 2] = globalSectorZ;
    sectorCoordMap[localSectorIndex * 4 + 3] = 0;
    pendingSectorBrickFlush.insert(localKey);
    if (compute && sectorCoordUniform) {
        compute->updateStorageBuffer(
            sectorCoordUniform,
            &sectorCoordMap[localSectorIndex * 4],
            4 * sizeof(int32_t),
            static_cast<size_t>(localSectorIndex) * 4 * sizeof(int32_t));
    }

    const auto cachedSector = generatedSectorVoxelData.find(globalKey);
    if (cachedSector != generatedSectorVoxelData.end()) {
        const int localStartX = localSectorX * sectorSize;
        const int localStartY = localSectorY * sectorSize;
        const int localStartZ = localSectorZ * sectorSize;

        auto voxelIndex = [this](int lx, int ly, int lz) {
            return static_cast<size_t>(lx) +
                   static_cast<size_t>(ly) * static_cast<size_t>(sectorSize) +
                   static_cast<size_t>(lz) * static_cast<size_t>(sectorSize) * static_cast<size_t>(sectorSize);
        };

        std::lock_guard<std::mutex> lock(brickMutex);
        for (int lz = 0; lz < sectorSize; ++lz) {
            for (int ly = 0; ly < sectorSize; ++ly) {
                for (int lx = 0; lx < sectorSize; ++lx) {
                    const uint16_t material = cachedSector->second[voxelIndex(lx, ly, lz)];
                    if (material == 0u) {
                        continue;
                    }

                    const int worldX = localStartX + lx;
                    const int worldY = localStartY + ly;
                    const int worldZ = localStartZ + lz;
                    setVoxelInternal(worldX, worldY, worldZ, material, false);
                }
            }
        }

        const bool discardUpload = shouldDiscardSectorGpuUpload(globalSectorX, globalSectorY, globalSectorZ);
        if (!discardUpload) {
            const int topStartX = localSectorX * bricksPerSector;
            const int topStartY = localSectorY * bricksPerSector;
            const int topStartZ = localSectorZ * bricksPerSector;
            for (int localTopZ = 0; localTopZ < bricksPerSector; ++localTopZ) {
                for (int localTopY = 0; localTopY < bricksPerSector; ++localTopY) {
                    for (int localTopX = 0; localTopX < bricksPerSector; ++localTopX) {
                        const int topX = topStartX + localTopX;
                        const int topY = topStartY + localTopY;
                        const int topZ = topStartZ + localTopZ;

                        queueTopLevelUpdate(topX * brickSize, topY * brickSize, topZ * brickSize);

                        const size_t topIndex = static_cast<size_t>(topX) +
                                                static_cast<size_t>(topY) * static_cast<size_t>(topLevelSize) +
                                                static_cast<size_t>(topZ) * static_cast<size_t>(topLevelSize) *
                                                    static_cast<size_t>(topLevelSize);
                        const uint32_t topEntry = topLevelGrid[topIndex];
                        if ((topEntry & IS_BRICK_FLAG) != 0u) {
                            BrickUpdate brickUpdate;
                            brickUpdate.brickIndex = topEntry & BRICK_INDEX_MASK;
                            memcpy(
                                brickUpdate.data,
                                &brickPool[static_cast<size_t>(brickUpdate.brickIndex) * 512],
                                512 * sizeof(uint16_t));
                            pendingBrickUpdates.push_back(brickUpdate);
                        }
                    }
                }
            }
        }
        sectorMapDirty = true;
    } else {
        generateProceduralTerrainSector(
            globalSectorX,
            globalSectorY,
            globalSectorZ,
            localSectorX,
            localSectorY,
            localSectorZ);
    }

    ++terrainStreamedSectorCount;
}

void VoxelCompute::generateProceduralTerrainSector(
    int globalSectorX,
    int globalSectorY,
    int globalSectorZ,
    int localSectorX,
    int localSectorY,
    int localSectorZ) {
    const int localStartX = localSectorX * sectorSize;
    const int localStartY = localSectorY * sectorSize;
    const int localStartZ = localSectorZ * sectorSize;
    const int localEndX = std::min(worldSize, localStartX + sectorSize);
    const int localEndY = std::min(worldSize, localStartY + sectorSize);
    const int localEndZ = std::min(worldSize, localStartZ + sectorSize);

    const int globalStartY = globalSectorY * sectorSize;

    const int globalStartX = globalSectorX * sectorSize;
    const int globalStartZ = globalSectorZ * sectorSize;

    std::vector<uint16_t> sectorVoxels(static_cast<size_t>(sectorSize) * sectorSize * sectorSize, 0u);
    auto voxelIndex = [this](int lx, int ly, int lz) {
        return static_cast<size_t>(lx) +
               static_cast<size_t>(ly) * static_cast<size_t>(sectorSize) +
               static_cast<size_t>(lz) * static_cast<size_t>(sectorSize) * static_cast<size_t>(sectorSize);
    };

    auto sampleHeightInfinite = [this](int x, int z) {
        return sampleTerrainHeight(x, z);
    };

    constexpr int minTerrainHeight = 6;
    constexpr int maxTerrainHeight = 1024;
    uint32_t nonAirCount = 0u;

    for (int localZ = localStartZ; localZ < localEndZ; ++localZ) {
        const int globalZ = globalStartZ + (localZ - localStartZ);
        for (int localX = localStartX; localX < localEndX; ++localX) {
            const int globalX = globalStartX + (localX - localStartX);
            const int surfaceY = sampleHeightInfinite(globalX, globalZ);
            const int globalSectorBottomY = globalStartY;
            if (surfaceY < globalSectorBottomY) {
                continue;
            }

            const int slopeX = std::abs(sampleHeightInfinite(globalX + 1, globalZ) - sampleHeightInfinite(globalX - 1, globalZ));
            const int slopeZ = std::abs(sampleHeightInfinite(globalX, globalZ + 1) - sampleHeightInfinite(globalX, globalZ - 1));
            const int maxGlobalFillY = std::min(globalStartY + sectorSize - 1, surfaceY);

            for (int globalY = globalStartY; globalY <= maxGlobalFillY; ++globalY) {
                const bool nearSurface = (surfaceY - globalY) < 9;
                if (!nearSurface && globalY > 24) {
                    const float caveNoise = fbm3D(glm::vec3(float(globalX), float(globalY), float(globalZ)) * 0.035f, 3, 2.0f, 0.52f);
                    if (caveNoise > 0.36f) {
                        continue;
                    }
                }

                const uint16_t material = chooseTerrainMaterial(
                    globalX,
                    globalY,
                    globalZ,
                    surfaceY,
                    slopeX,
                    slopeZ,
                    minTerrainHeight,
                    maxTerrainHeight);
                const int localY = localStartY + (globalY - globalStartY);
                if (localY < localStartY || localY >= localEndY) {
                    continue;
                }

                setVoxelInternal(localX, localY, localZ, material, false);
                const int lx = localX - localStartX;
                const int ly = localY - localStartY;
                const int lz = localZ - localStartZ;
                if (lx >= 0 && lx < sectorSize && ly >= 0 && ly < sectorSize && lz >= 0 && lz < sectorSize) {
                    sectorVoxels[voxelIndex(lx, ly, lz)] = material;
                    ++nonAirCount;
                }
            }
        }
    }

    const std::string globalKey = makeSectorKey(globalSectorX, globalSectorY, globalSectorZ);
    generatedSectorNonAirCounts[globalKey] = nonAirCount;
    const bool discardUpload = shouldDiscardSectorGpuUpload(globalSectorX, globalSectorY, globalSectorZ);

    if (!discardUpload) {
        const int topStartX = localSectorX * bricksPerSector;
        const int topStartY = localSectorY * bricksPerSector;
        const int topStartZ = localSectorZ * bricksPerSector;
        for (int localTopZ = 0; localTopZ < bricksPerSector; ++localTopZ) {
            for (int localTopY = 0; localTopY < bricksPerSector; ++localTopY) {
                for (int localTopX = 0; localTopX < bricksPerSector; ++localTopX) {
                    const int topX = topStartX + localTopX;
                    const int topY = topStartY + localTopY;
                    const int topZ = topStartZ + localTopZ;
                    if (topX < 0 || topX >= topLevelSize ||
                        topY < 0 || topY >= topLevelSize ||
                        topZ < 0 || topZ >= topLevelSize) {
                        continue;
                    }

                    queueTopLevelUpdate(topX * brickSize, topY * brickSize, topZ * brickSize);

                    const size_t topIndex = static_cast<size_t>(topX) +
                                            static_cast<size_t>(topY) * static_cast<size_t>(topLevelSize) +
                                            static_cast<size_t>(topZ) * static_cast<size_t>(topLevelSize) *
                                                static_cast<size_t>(topLevelSize);
                    const uint32_t topEntry = topLevelGrid[topIndex];
                    if ((topEntry & IS_BRICK_FLAG) != 0u) {
                        BrickUpdate brickUpdate;
                        brickUpdate.brickIndex = topEntry & BRICK_INDEX_MASK;
                        memcpy(
                            brickUpdate.data,
                            &brickPool[static_cast<size_t>(brickUpdate.brickIndex) * 512],
                            512 * sizeof(uint16_t));
                        pendingBrickUpdates.push_back(brickUpdate);
                    }
                }
            }
        }
    }

    generatedSectorVoxelData[globalKey] = std::move(sectorVoxels);
    sectorMapDirty = true;
}

void VoxelCompute::streamTerrainAroundPlayer(const glm::vec3& cameraPos, const glm::mat4& invViewProj) {
    if (!terrainStreamingEnabled) {
        return;
    }
    (void)invViewProj;

    auto updateAnchorSector = [&](float position, int& anchor) {
        if (anchor == INT32_MIN) {
            anchor = static_cast<int>(std::floor(position / static_cast<float>(sectorSize)));
            return;
        }

        const float hysteresis = std::max(0.0f, streamCenterHysteresisBlocks);
        for (;;) {
            const float lowerBound = static_cast<float>(anchor * sectorSize) - hysteresis;
            const float upperBound = static_cast<float>((anchor + 1) * sectorSize) + hysteresis;
            if (position < lowerBound) {
                --anchor;
                continue;
            }
            if (position >= upperBound) {
                ++anchor;
                continue;
            }
            break;
        }
    };

    updateAnchorSector(cameraPos.x, streamAnchorSectorX);
    updateAnchorSector(cameraPos.y, streamAnchorSectorY);
    updateAnchorSector(cameraPos.z, streamAnchorSectorZ);

    const int centerSectorX = streamAnchorSectorX;
    const int centerSectorY = streamAnchorSectorY;
    const int centerSectorZ = streamAnchorSectorZ;
    const int maxNoAliasRadius = std::max(1, (sectorsPerDimension - 1) / 2);
    const int effectiveRadius = std::min(std::max(1, terrainStreamRadiusSectors), maxNoAliasRadius);

    static bool warnedRadiusClamp = false;
    if (!warnedRadiusClamp && terrainStreamRadiusSectors > maxNoAliasRadius) {
        printf(
            "Terrain stream radius %d exceeds stable slot limit for %d sectors/dimension; clamping streaming radius to %d to prevent slot-alias flicker.\n",
            terrainStreamRadiusSectors,
            sectorsPerDimension,
            effectiveRadius);
        warnedRadiusClamp = true;
    }

    const int sectorsPerFrame = std::max(1, static_cast<int>(World::renderSpeed));

    if (centerSectorX != lastCachedCenterSectorX ||
        centerSectorY != lastCachedCenterSectorY ||
        centerSectorZ != lastCachedCenterSectorZ ||
        cachedSphereMaskRadius != effectiveRadius ||
        sectorCandidates.empty()) {
        rebuildSectorCandidateCache(centerSectorX, centerSectorY, centerSectorZ);
    }
    int evictedThisFrame = 0;

    auto evictOneResidentIfNeeded = [&]() -> bool {
        if (residentLocalToGlobal.size() < static_cast<size_t>(sectorCount)) {
            return true;
        }

        std::vector<std::pair<uint64_t, std::string>> invalidEntries;
        invalidEntries.reserve(4);

        uint64_t farthestLocalKey = 0u;
        std::string farthestGlobalKey;
        int farthestDistSq = -1;
        bool foundFarthest = false;

        for (const auto& residentEntry : residentLocalToGlobal) {
            const uint64_t localKey = residentEntry.first;
            const int localSectorX = static_cast<int>(localKey & 0xFFFFu);
            const int localSectorY = static_cast<int>((localKey >> 16u) & 0xFFFFu);
            const int localSectorZ = static_cast<int>((localKey >> 32u) & 0xFFFFu);

            if (localSectorX < 0 || localSectorX >= sectorsPerDimension ||
                localSectorY < 0 || localSectorY >= sectorsPerDimension ||
                localSectorZ < 0 || localSectorZ >= sectorsPerDimension) {
                invalidEntries.emplace_back(localKey, residentEntry.second);
                continue;
            }

            const int localSectorIndex =
                localSectorX +
                localSectorY * sectorsPerDimension +
                localSectorZ * sectorsPerDimension * sectorsPerDimension;
            if (localSectorIndex < 0 || localSectorIndex >= sectorCount) {
                invalidEntries.emplace_back(localKey, residentEntry.second);
                continue;
            }

            const int globalSectorX = sectorCoordMap[localSectorIndex * 4 + 0];
            const int globalSectorY = sectorCoordMap[localSectorIndex * 4 + 1];
            const int globalSectorZ = sectorCoordMap[localSectorIndex * 4 + 2];

            const int dx = globalSectorX - centerSectorX;
            const int dy = globalSectorY - centerSectorY;
            const int dz = globalSectorZ - centerSectorZ;
            const int distSq = dx * dx + dy * dy + dz * dz;
            if (!foundFarthest || distSq > farthestDistSq) {
                farthestDistSq = distSq;
                farthestLocalKey = localKey;
                farthestGlobalKey = residentEntry.second;
                foundFarthest = true;
            }
        }

        for (const auto& invalid : invalidEntries) {
            const uint64_t localKey = invalid.first;
            const int localSectorX = static_cast<int>(localKey & 0xFFFFu);
            const int localSectorY = static_cast<int>((localKey >> 16u) & 0xFFFFu);
            const int localSectorZ = static_cast<int>((localKey >> 32u) & 0xFFFFu);
            residentGlobalToLocal.erase(invalid.second);
            residentLocalToGlobal.erase(localKey);
            if (localSectorX >= 0 && localSectorX < sectorsPerDimension &&
                localSectorY >= 0 && localSectorY < sectorsPerDimension &&
                localSectorZ >= 0 && localSectorZ < sectorsPerDimension) {
                clearLocalSectorSlot(localSectorX, localSectorY, localSectorZ);
            }
            ++evictedThisFrame;
        }

        if (!foundFarthest) {
            return residentLocalToGlobal.size() < static_cast<size_t>(sectorCount);
        }

        const int localSectorX = static_cast<int>(farthestLocalKey & 0xFFFFu);
        const int localSectorY = static_cast<int>((farthestLocalKey >> 16u) & 0xFFFFu);
        const int localSectorZ = static_cast<int>((farthestLocalKey >> 32u) & 0xFFFFu);
        residentGlobalToLocal.erase(farthestGlobalKey);
        residentLocalToGlobal.erase(farthestLocalKey);
        clearLocalSectorSlot(localSectorX, localSectorY, localSectorZ);
        ++evictedThisFrame;
        return true;
    };

    int generatedThisFrame = 0;
    while (generatedThisFrame < sectorsPerFrame && sectorCandidateCursor < sectorCandidates.size()) {
        const SectorCandidate& candidate = sectorCandidates[sectorCandidateCursor++];
        const std::string globalKey = makeSectorKey(candidate.x, candidate.y, candidate.z);
        if (residentGlobalToLocal.find(globalKey) != residentGlobalToLocal.end()) {
            continue;
        }

        const int candidateLocalSectorX = wrapSectorCoord(candidate.x);
        const int candidateLocalSectorY = wrapSectorCoord(candidate.y);
        const int candidateLocalSectorZ = wrapSectorCoord(candidate.z);
        const uint64_t candidateLocalKey =
            packLocalSectorKey(candidateLocalSectorX, candidateLocalSectorY, candidateLocalSectorZ);
        const auto occupiedLocal = residentLocalToGlobal.find(candidateLocalKey);
        const bool willReplaceExistingSlot =
            occupiedLocal != residentLocalToGlobal.end() && occupiedLocal->second != globalKey;

        // If this candidate maps to an occupied local slot, ensureSectorResident will evict that slot directly.
        if (!willReplaceExistingSlot && !evictOneResidentIfNeeded()) {
            break;
        }

        ensureSectorResident(candidate.x, candidate.y, candidate.z);
        ++generatedThisFrame;
    }

    updateResidentSectorBounds();

    if ((generatedThisFrame > 0 || evictedThisFrame > 0) && (terrainStreamedSectorCount % 64u) == 0u) {
        printf("Terrain streaming loads: %zu sectors (resident=%zu, generated=%zu, evicted=%d, cacheSlots=%d)\n",
               terrainStreamedSectorCount,
               residentGlobalToLocal.size(),
               generatedSectorVoxelData.size(),
               evictedThisFrame,
               sectorCount);
    }
}

void VoxelCompute::generateProceduralTerrain() {
    initializeTerrainPalette();

    constexpr int surfaceBandThickness = 14;
    const uint32_t deepRockColor = packRGB(88, 92, 98);

    std::vector<uint16_t> heightMap(worldSize * worldSize);
    int minTerrainHeight = worldSize;
    int maxTerrainHeight = 0;
    std::vector<int> solidBaseHeights(topLevelSize * topLevelSize, 0);

    auto heightIndex = [this](int x, int z) {
        return x + z * worldSize;
    };

    for (int z = 0; z < worldSize; ++z) {
        for (int x = 0; x < worldSize; ++x) {
            const int h = sampleTerrainHeight(x, z);
            heightMap[heightIndex(x, z)] = static_cast<uint16_t>(h);
            minTerrainHeight = std::min(minTerrainHeight, h);
            maxTerrainHeight = std::max(maxTerrainHeight, h);
        }
    }

    auto getHeight = [&](int x, int z) -> int {
        x = glm::clamp(x, 0, worldSize - 1);
        z = glm::clamp(z, 0, worldSize - 1);
        return heightMap[heightIndex(x, z)];
    };

    for (int topZ = 0; topZ < topLevelSize; ++topZ) {
        for (int topX = 0; topX < topLevelSize; ++topX) {
            int minHeight = worldSize;
            for (int localZ = 0; localZ < brickSize; ++localZ) {
                for (int localX = 0; localX < brickSize; ++localX) {
                    int worldX = topX * brickSize + localX;
                    int worldZ = topZ * brickSize + localZ;
                    minHeight = std::min(minHeight, getHeight(worldX, worldZ));
                }
            }

            int detailStart = std::max(0, minHeight - surfaceBandThickness);
            int fullySolidCells = std::max(0, (detailStart + 1) / brickSize);
            solidBaseHeights[topX + topZ * topLevelSize] = fullySolidCells * brickSize;

            for (int topY = 0; topY < fullySolidCells; ++topY) {
                size_t topIndex = topX + topY * topLevelSize + topZ * topLevelSize * topLevelSize;
                topLevelGrid[topIndex] = deepRockColor;
            }
        }
    }

    size_t detailedVoxels = 0;
    for (int z = 0; z < worldSize; ++z) {
        for (int x = 0; x < worldSize; ++x) {
            int surfaceY = getHeight(x, z);
            int slopeX = std::abs(getHeight(x + 1, z) - getHeight(x - 1, z));
            int slopeZ = std::abs(getHeight(x, z + 1) - getHeight(x, z - 1));
            int baseY = solidBaseHeights[(x / brickSize) + (z / brickSize) * topLevelSize];

            for (int y = baseY; y <= surfaceY; ++y) {
                bool nearSurface = (surfaceY - y) < 9;
                if (!nearSurface && y > 24) {
                    float caveNoise = fbm3D(glm::vec3(float(x), float(y), float(z)) * 0.035f, 3, 2.0f, 0.52f);
                    if (caveNoise > 0.36f) {
                        continue;
                    }
                }

                uint16_t material = chooseTerrainMaterial(
                    x,
                    y,
                    z,
                    surfaceY,
                    slopeX,
                    slopeZ,
                    minTerrainHeight,
                    maxTerrainHeight);
                setVoxelInternal(x, y, z, material, false);
                detailedVoxels++;
            }
        }
    }

    printf("Generated procedural terrain with %zu detailed voxels\n", detailedVoxels);
}

namespace compute_layers {

void generateProceduralTerrain(VoxelCompute& volume) {
    volume.generateProceduralTerrain();
}

} // namespace compute_layers
