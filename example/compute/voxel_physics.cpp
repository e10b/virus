#include "voxel_physics.h"

#include <cmath>
#include <algorithm>

uint16_t VoxelCompute::getMaterialForToolMode(int toolMode) const {
    if (toolMode == 9) {
        return SOUND_VOXEL_MATERIAL;
    }
    return PLAYER_PLACED_MATERIAL;
}

namespace {
int floorDiv(int value, int divisor) {
    if (value >= 0) {
        return value / divisor;
    }
    return -(((-value) + divisor - 1) / divisor);
}

int floorMod(int value, int divisor) {
    return value - floorDiv(value, divisor) * divisor;
}

bool mapGlobalVoxelToLocal(const VoxelCompute& volume, int gx, int gy, int gz, int& lx, int& ly, int& lz) {
    if (gy < 0) {
        return false;
    }

    const int sectorX = floorDiv(gx, volume.sectorSize);
    const int sectorY = floorDiv(gy, volume.sectorSize);
    const int sectorZ = floorDiv(gz, volume.sectorSize);

    const std::string globalKey = volume.makeSectorKey(sectorX, sectorY, sectorZ);
    const auto it = volume.residentGlobalToLocal.find(globalKey);
    if (it == volume.residentGlobalToLocal.end()) {
        return false;
    }

    const uint64_t localKey = it->second;
    const int localSectorX = static_cast<int>(localKey & 0xFFFFu);
    const int localSectorY = static_cast<int>((localKey >> 16u) & 0xFFFFu);
    const int localSectorZ = static_cast<int>((localKey >> 32u) & 0xFFFFu);

    const int ox = floorMod(gx, volume.sectorSize);
    const int oy = floorMod(gy, volume.sectorSize);
    const int oz = floorMod(gz, volume.sectorSize);

    lx = localSectorX * volume.sectorSize + ox;
    ly = localSectorY * volume.sectorSize + oy;
    lz = localSectorZ * volume.sectorSize + oz;

    return lx >= 0 && lx < volume.worldSize &&
           ly >= 0 && ly < volume.worldSize &&
           lz >= 0 && lz < volume.worldSize;
}
}

void VoxelCompute::setVoxelInternal(int x, int y, int z, uint16_t voxelData, bool updateSector) {
    if (x < 0 || x >= worldSize || y < 0 || y >= worldSize || z < 0 || z >= worldSize) {
        return;
    }

    size_t topIndex = getTopLevelIndex(x, y, z);
    uint32_t& topEntry = topLevelGrid[topIndex];
    bool oldPopulated = (topEntry != 0u);

    int localX = x % brickSize;
    int localY = y % brickSize;
    int localZ = z % brickSize;
    size_t voxelOffset = getBrickVoxelOffset(localX, localY, localZ);

    bool isEmpty = (voxelData == 0);

    if (topEntry & IS_BRICK_FLAG) {
        uint32_t brickIndex = topEntry & BRICK_INDEX_MASK;
        size_t dataOffset = brickIndex * 512 + voxelOffset;

        bool wasEmpty = (brickPool[dataOffset] == 0);

        if (wasEmpty && !isEmpty) {
            brickMetadata[brickIndex].nonEmptyCount++;
        } else if (!wasEmpty && isEmpty) {
            brickMetadata[brickIndex].nonEmptyCount--;
            if (brickMetadata[brickIndex].nonEmptyCount == 0) {
                freeBrickInternal(brickIndex);
                topEntry = 0;
                if (updateSector) {
                    queueTopLevelUpdate(x, y, z);
                    updateSectorForVoxel(x, y, z);
                }
                return;
            }
        }

        brickPool[dataOffset] = voxelData;
        if (updateSector) {
            pendingVoxelWordIndices.insert(static_cast<uint32_t>(dataOffset / 2));
        }
    } else {
        if (topEntry == 0u && isEmpty) {
            return;
        }

        uint16_t oldMaterial = 0;
        if (topEntry != 0u) {
            oldMaterial = static_cast<uint16_t>(findClosestPaletteIndex(0xFF000000u | (topEntry & BRICK_INDEX_MASK)));
            if (oldMaterial == 0) {
                oldMaterial = 4;
            }
        }

        uint32_t brickIndex = allocateBrickInternal();
        if (brickIndex >= maxBrickPoolSize) {
            return;
        }

        if (oldMaterial != 0) {
            std::fill_n(&brickPool[brickIndex * 512], 512, oldMaterial);
            brickMetadata[brickIndex].nonEmptyCount = 512;
        } else {
            memset(&brickPool[brickIndex * 512], 0, 512 * sizeof(uint16_t));
            brickMetadata[brickIndex].nonEmptyCount = 0;
        }

        size_t dataOffset = brickIndex * 512 + voxelOffset;
        bool wasEmpty = (brickPool[dataOffset] == 0);

        if (wasEmpty && !isEmpty) {
            brickMetadata[brickIndex].nonEmptyCount++;
        } else if (!wasEmpty && isEmpty) {
            brickMetadata[brickIndex].nonEmptyCount--;
        }

        brickPool[dataOffset] = voxelData;
        topEntry = IS_BRICK_FLAG | brickIndex;
        if (updateSector) {
            queueTopLevelUpdate(x, y, z);
            BrickUpdate brickUpdate;
            brickUpdate.brickIndex = brickIndex;
            memcpy(brickUpdate.data, &brickPool[brickIndex * 512], 512 * sizeof(uint16_t));
            pendingBrickUpdates.push_back(brickUpdate);
        }
    }

    bool newPopulated = (topEntry != 0u);
    if (updateSector && oldPopulated != newPopulated) {
        updateSectorForVoxel(x, y, z);
    }
}

void VoxelCompute::setVoxel(int x, int y, int z, uint16_t voxelData) {
    if (y < 0) {
        return;
    }

    const int globalSectorX = floorDiv(x, sectorSize);
    const int globalSectorY = floorDiv(y, sectorSize);
    const int globalSectorZ = floorDiv(z, sectorSize);
    const int sectorLocalX = floorMod(x, sectorSize);
    const int sectorLocalY = floorMod(y, sectorSize);
    const int sectorLocalZ = floorMod(z, sectorSize);
    ensureSectorResident(globalSectorX, globalSectorY, globalSectorZ);

    int localX = 0;
    int localY = 0;
    int localZ = 0;
    if (!mapGlobalVoxelToLocal(*this, x, y, z, localX, localY, localZ)) {
        return;
    }

    std::lock_guard<std::mutex> lock(brickMutex);
    setVoxelInternal(localX, localY, localZ, voxelData, true);

    const std::string globalKey = makeSectorKey(globalSectorX, globalSectorY, globalSectorZ);
    auto& sectorData = generatedSectorVoxelData[globalKey];
    const size_t sectorVoxelCount = static_cast<size_t>(sectorSize) * sectorSize * sectorSize;
    if (sectorData.size() != sectorVoxelCount) {
        sectorData.assign(sectorVoxelCount, 0u);
    }

    const size_t sectorIndex = static_cast<size_t>(sectorLocalX) +
                               static_cast<size_t>(sectorLocalY) * static_cast<size_t>(sectorSize) +
                               static_cast<size_t>(sectorLocalZ) * static_cast<size_t>(sectorSize) *
                                   static_cast<size_t>(sectorSize);
    const uint16_t previousVoxel = sectorData[sectorIndex];
    sectorData[sectorIndex] = voxelData;

    uint32_t& nonAirCount = generatedSectorNonAirCounts[globalKey];
    if (previousVoxel == 0u && voxelData != 0u) {
        ++nonAirCount;
    } else if (previousVoxel != 0u && voxelData == 0u && nonAirCount > 0u) {
        --nonAirCount;
    }
}

void VoxelCompute::clearVoxel(int x, int y, int z) {
    setVoxel(x, y, z, 0);
}

bool VoxelCompute::isVoxelOccupied(int x, int y, int z) const {
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    if (!mapGlobalVoxelToLocal(*this, x, y, z, localX, localY, localZ)) {
        return false;
    }

    size_t topIndex = getTopLevelIndex(localX, localY, localZ);
    uint32_t topEntry = topLevelGrid[topIndex];

    if (topEntry & IS_BRICK_FLAG) {
        uint32_t brickIndex = topEntry & BRICK_INDEX_MASK;
        int brickLocalX = localX % brickSize;
        int brickLocalY = localY % brickSize;
        int brickLocalZ = localZ % brickSize;
        size_t voxelOffset = getBrickVoxelOffset(brickLocalX, brickLocalY, brickLocalZ);
        size_t dataOffset = brickIndex * 512 + voxelOffset;

        return brickPool[dataOffset] != 0;
    }

    return topEntry != 0u;
}

uint16_t VoxelCompute::getVoxel(int x, int y, int z) const {
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    if (!mapGlobalVoxelToLocal(*this, x, y, z, localX, localY, localZ)) {
        return 0;
    }

    size_t topIndex = getTopLevelIndex(localX, localY, localZ);
    uint32_t topEntry = topLevelGrid[topIndex];

    if (topEntry & IS_BRICK_FLAG) {
        uint32_t brickIndex = topEntry & BRICK_INDEX_MASK;
        int brickLocalX = localX % brickSize;
        int brickLocalY = localY % brickSize;
        int brickLocalZ = localZ % brickSize;
        size_t voxelOffset = getBrickVoxelOffset(brickLocalX, brickLocalY, brickLocalZ);
        size_t dataOffset = brickIndex * 512 + voxelOffset;

        return brickPool[dataOffset];
    }

    if (topEntry == 0u) {
        return 0;
    }

    uint16_t material = static_cast<uint16_t>(findClosestPaletteIndex(0xFF000000u | (topEntry & BRICK_INDEX_MASK)));
    return material == 0 ? 4 : material;
}

bool VoxelCompute::voxelIntersectsCapsule(
    int voxelX,
    int voxelY,
    int voxelZ,
    const glm::vec3& capsuleFeetPosition,
    const glm::vec3& capsuleSize) const {
    const float radius = capsuleSize.x * 0.5f;
    const float cylinderHeight = glm::max(0.0f, capsuleSize.y - radius * 2.0f);
    const int sampleCount = glm::max(
        2,
        static_cast<int>(std::ceil(cylinderHeight / glm::max(radius, 1.0f))) + 1);
    const float radiusSq = radius * radius;

    for (int sample = 0; sample < sampleCount; ++sample) {
        const float t = sampleCount == 1 ? 0.0f : static_cast<float>(sample) / static_cast<float>(sampleCount - 1);
        const glm::vec3 sphereCenter = capsuleFeetPosition + glm::vec3(0.0f, radius + cylinderHeight * t, 0.0f);
        const float closestX = glm::clamp(sphereCenter.x, static_cast<float>(voxelX), static_cast<float>(voxelX + 1));
        const float closestY = glm::clamp(sphereCenter.y, static_cast<float>(voxelY), static_cast<float>(voxelY + 1));
        const float closestZ = glm::clamp(sphereCenter.z, static_cast<float>(voxelZ), static_cast<float>(voxelZ + 1));
        const glm::vec3 delta = sphereCenter - glm::vec3(closestX, closestY, closestZ);
        if (glm::dot(delta, delta) < radiusSq) {
            return true;
        }
    }

    return false;
}

bool VoxelCompute::placementWouldIntersectPlayer(
    const glm::ivec3& center,
    int toolMode,
    const glm::vec3& playerFeetPosition,
    const glm::vec3& playerSize) const {
    int rCube = 0;
    int rSphere = 0;

    if (toolMode == 2) { rCube = 4; }
    else if (toolMode == 3) { rSphere = 8; }
    else if (toolMode == 5) { rCube = 16; }
    else if (toolMode == 6) { rSphere = 16; }
    else if (toolMode == 7) { rCube = 64; }
    else if (toolMode == 8) { rSphere = 64; }

    if (rCube > 0) {
        for (int dx = -rCube; dx <= rCube; ++dx) {
            for (int dy = -rCube; dy <= rCube; ++dy) {
                for (int dz = -rCube; dz <= rCube; ++dz) {
                    if (voxelIntersectsCapsule(center.x + dx, center.y + dy, center.z + dz, playerFeetPosition, playerSize)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    if (rSphere > 0) {
        const int rSq = rSphere * rSphere;
        for (int dx = -rSphere; dx <= rSphere; ++dx) {
            for (int dy = -rSphere; dy <= rSphere; ++dy) {
                for (int dz = -rSphere; dz <= rSphere; ++dz) {
                    if (dx * dx + dy * dy + dz * dz > rSq) {
                        continue;
                    }
                    if (voxelIntersectsCapsule(center.x + dx, center.y + dy, center.z + dz, playerFeetPosition, playerSize)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    return voxelIntersectsCapsule(center.x, center.y, center.z, playerFeetPosition, playerSize);
}

bool VoxelCompute::raycast(
    const glm::vec3& origin,
    const glm::vec3& direction,
    float length,
    bool add,
    int toolMode,
    bool preventPlayerIntersection,
    const glm::vec3& playerFeetPosition,
    const glm::vec3& playerSize) {
    glm::vec3 pos = origin;
    glm::vec3 dir = glm::normalize(direction);

    const float epsilon = 1e-6f;

    glm::ivec3 voxel = glm::ivec3(glm::floor(pos + glm::vec3(
        0.5f * glm::sign(dir.x) * epsilon,
        0.5f * glm::sign(dir.y) * epsilon,
        0.5f * glm::sign(dir.z) * epsilon
    )));

    glm::ivec3 step = glm::ivec3(
        dir.x >= 0 ? 1 : -1,
        dir.y >= 0 ? 1 : -1,
        dir.z >= 0 ? 1 : -1
    );

    glm::vec3 safeDir = glm::vec3(
        glm::abs(dir.x) < epsilon ? (dir.x >= 0 ? epsilon : -epsilon) : dir.x,
        glm::abs(dir.y) < epsilon ? (dir.y >= 0 ? epsilon : -epsilon) : dir.y,
        glm::abs(dir.z) < epsilon ? (dir.z >= 0 ? epsilon : -epsilon) : dir.z
    );

    glm::vec3 deltaDist = glm::abs(1.0f / safeDir);

    glm::vec3 nextBoundary = glm::vec3(voxel) + glm::vec3(
        step.x > 0 ? 1.0f : 0.0f,
        step.y > 0 ? 1.0f : 0.0f,
        step.z > 0 ? 1.0f : 0.0f
    );

    glm::vec3 tMax = (nextBoundary - pos) / safeDir;

    float traveled = 0.0f;
    glm::ivec3 lastEmpty = voxel;

    static int debugCounter = 0;
    if (add && debugCounter++ % 60 == 0) {
        printf("🔍 RAYCAST DEBUG: add=%s, origin=(%.2f,%.2f,%.2f), dir=(%.2f,%.2f,%.2f)\n",
               add ? "true" : "false", origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);
    }

    while (traveled < length) {
        if (voxel.y < 0) {
            break;
        }

        uint16_t voxelValue = getVoxel(voxel.x, voxel.y, voxel.z);
        bool isSolid = isVoxelOccupied(voxel.x, voxel.y, voxel.z);

        if (add && debugCounter % 60 == 0 && traveled < 10.0f) {
            printf("  Voxel(%d,%d,%d): value=%d, solid=%s\n",
                   voxel.x, voxel.y, voxel.z, voxelValue, isSolid ? "true" : "false");
        }

        if ((add && isSolid && lastEmpty.y >= 0) ||
            (!add && isSolid)) {

            glm::ivec3 center = add ? lastEmpty : voxel;
            int rCube = 0;
            int rSphere = 0;

            if (toolMode == 2) { rCube = 4; }
            else if (toolMode == 3) { rSphere = 8; }
            else if (toolMode == 5) { rCube = 16; }
            else if (toolMode == 6) { rSphere = 16; }
            else if (toolMode == 7) { rCube = 64; }
            else if (toolMode == 8) { rSphere = 64; }

            const int editRadius = rCube > 0 ? rCube : rSphere;
            const int minEditX = center.x - editRadius;
            const int maxEditX = center.x + editRadius;
            const int minEditY = std::max(0, center.y - editRadius);
            const int maxEditY = center.y + editRadius;
            const int minEditZ = center.z - editRadius;
            const int maxEditZ = center.z + editRadius;

            const int minSectorX = floorDiv(minEditX, sectorSize);
            const int maxSectorX = floorDiv(maxEditX, sectorSize);
            const int minSectorY = floorDiv(minEditY, sectorSize);
            const int maxSectorY = floorDiv(maxEditY, sectorSize);
            const int minSectorZ = floorDiv(minEditZ, sectorSize);
            const int maxSectorZ = floorDiv(maxEditZ, sectorSize);

            for (int sz = minSectorZ; sz <= maxSectorZ; ++sz) {
                for (int sy = minSectorY; sy <= maxSectorY; ++sy) {
                    for (int sx = minSectorX; sx <= maxSectorX; ++sx) {
                        ensureSectorResident(sx, sy, sz);
                    }
                }
            }

            if (add && preventPlayerIntersection &&
                placementWouldIntersectPlayer(center, toolMode, playerFeetPosition, playerSize)) {
                return false;
            }

            uint16_t material = add ? getMaterialForToolMode(toolMode) : 0;
            bool isSound = (material == SOUND_VOXEL_MATERIAL);

            if (rCube > 0) {
                for (int dx = -rCube; dx <= rCube; dx++) {
                    for (int dy = -rCube; dy <= rCube; dy++) {
                        for (int dz = -rCube; dz <= rCube; dz++) {
                            glm::ivec3 voxelPos(center.x + dx, center.y + dy, center.z + dz);
                            if (add) {
                                setVoxel(voxelPos.x, voxelPos.y, voxelPos.z, material);
                                if (isSound) {
                                    auto it = std::find(soundVoxelPositions.begin(), soundVoxelPositions.end(), voxelPos);
                                    if (it == soundVoxelPositions.end()) {
                                        soundVoxelPositions.push_back(voxelPos);
                                    }
                                }
                            } else {
                                clearVoxel(voxelPos.x, voxelPos.y, voxelPos.z);
                                auto it = std::find(soundVoxelPositions.begin(), soundVoxelPositions.end(), voxelPos);
                                if (it != soundVoxelPositions.end()) {
                                    soundVoxelPositions.erase(it);
                                }
                            }
                        }
                    }
                }
            } else if (rSphere > 0) {
                int rSq = rSphere * rSphere;
                for (int dx = -rSphere; dx <= rSphere; dx++) {
                    for (int dy = -rSphere; dy <= rSphere; dy++) {
                        for (int dz = -rSphere; dz <= rSphere; dz++) {
                            if (dx * dx + dy * dy + dz * dz <= rSq) {
                                glm::ivec3 voxelPos(center.x + dx, center.y + dy, center.z + dz);
                                if (add) {
                                    setVoxel(voxelPos.x, voxelPos.y, voxelPos.z, material);
                                    if (isSound) {
                                        auto it = std::find(soundVoxelPositions.begin(), soundVoxelPositions.end(), voxelPos);
                                        if (it == soundVoxelPositions.end()) {
                                            soundVoxelPositions.push_back(voxelPos);
                                        }
                                    }
                                } else {
                                    clearVoxel(voxelPos.x, voxelPos.y, voxelPos.z);
                                    auto it = std::find(soundVoxelPositions.begin(), soundVoxelPositions.end(), voxelPos);
                                    if (it != soundVoxelPositions.end()) {
                                        soundVoxelPositions.erase(it);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                glm::ivec3 voxelPos(center);
                if (add) {
                    setVoxel(voxelPos.x, voxelPos.y, voxelPos.z, material);
                    if (isSound) {
                        auto it = std::find(soundVoxelPositions.begin(), soundVoxelPositions.end(), voxelPos);
                        if (it == soundVoxelPositions.end()) {
                            soundVoxelPositions.push_back(voxelPos);
                        }
                    }
                } else {
                    clearVoxel(voxelPos.x, voxelPos.y, voxelPos.z);
                    auto it = std::find(soundVoxelPositions.begin(), soundVoxelPositions.end(), voxelPos);
                    if (it != soundVoxelPositions.end()) {
                        soundVoxelPositions.erase(it);
                    }
                }
            }

            return true;
        }

        lastEmpty = voxel;

        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            voxel.x += step.x;
            traveled = tMax.x;
            tMax.x += deltaDist.x;
        } else if (tMax.y < tMax.z) {
            voxel.y += step.y;
            traveled = tMax.y;
            tMax.y += deltaDist.y;
        } else {
            voxel.z += step.z;
            traveled = tMax.z;
            tMax.z += deltaDist.z;
        }
    }

    return false;
}

namespace compute_layers {

bool isVoxelOccupied(const VoxelCompute& volume, int x, int y, int z) {
    return volume.isVoxelOccupied(x, y, z);
}

uint16_t getVoxel(const VoxelCompute& volume, int x, int y, int z) {
    return volume.getVoxel(x, y, z);
}

bool raycast(
    VoxelCompute& volume,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float length,
    bool add,
    int toolMode,
    bool preventPlayerIntersection,
    const glm::vec3& playerFeetPosition,
    const glm::vec3& playerSize) {
    return volume.raycast(
        origin,
        direction,
        length,
        add,
        toolMode,
        preventPlayerIntersection,
        playerFeetPosition,
        playerSize);
}

} // namespace compute_layers
