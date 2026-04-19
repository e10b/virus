#include "voxel_renderer.h"

#include <algorithm>
#include <glm/gtc/type_ptr.hpp>

void VoxelCompute::copyTopLevelGridToTexture() {
    wgpu::Queue queue = wgfx::device.getQueue();

    wgpu::ImageCopyTexture destination = {};
    destination.texture = topLevelTexture3D.texture;
    destination.mipLevel = 0;
    destination.origin = {0, 0, 0};
    destination.aspect = wgpu::TextureAspect::All;

    wgpu::TextureDataLayout source = {};
    source.offset = 0;
    source.bytesPerRow = topLevelSize * sizeof(uint32_t);
    source.rowsPerImage = topLevelSize;

    wgpu::Extent3D copySize = {
        static_cast<uint32_t>(topLevelSize),
        static_cast<uint32_t>(topLevelSize),
        static_cast<uint32_t>(topLevelSize)
    };

    queue.writeTexture(destination, topLevelGrid, topLevelCount * sizeof(uint32_t), source, copySize);
    queue.release();

    sectorMapDirty = true;
}

bool VoxelCompute::rebuildSectorMapGPU() {
    if (!sectorCompute || !sectorUniform || !wgfx::encoder) {
        return false;
    }

    wgfx::ComputePass sectorPass;
    sectorPass.prepare();
    sectorPass.drawXYZ(
        sectorCompute,
        static_cast<uint32_t>(sectorsPerDimension),
        static_cast<uint32_t>(sectorsPerDimension),
        static_cast<uint32_t>(sectorsPerDimension));
    sectorPass.end();
    return true;
}

uint32_t VoxelCompute::getPackedBrickWord(uint32_t wordIndex) const {
    const size_t voxelBase = static_cast<size_t>(wordIndex) * 2;
    if (voxelBase + 1 >= maxBrickPoolSize * 512) {
        return 0u;
    }

    const uint32_t lo = static_cast<uint32_t>(brickPool[voxelBase]);
    const uint32_t hi = static_cast<uint32_t>(brickPool[voxelBase + 1]);
    return lo | (hi << 16);
}

void VoxelCompute::queueTopLevelUpdate(int x, int y, int z) {
    int topX = x / brickSize;
    int topY = y / brickSize;
    int topZ = z / brickSize;

    size_t topIndex = getTopLevelIndex(x, y, z);
    uint32_t value = topLevelGrid[topIndex];

    pendingTopLevelUpdates.push_back({
        static_cast<uint32_t>(topX),
        static_cast<uint32_t>(topY),
        static_cast<uint32_t>(topZ),
        value
    });
    sectorMapDirty = true;

    if (pendingTopLevelUpdates.size() >= MAX_BATCH_SIZE) {
        flushBatchedUpdates();
    }
}

void VoxelCompute::flushBatchedUpdates() {
    if (pendingTopLevelUpdates.empty() && pendingBrickUpdates.empty() &&
        pendingVoxelEdits.empty() && !sectorMapDirty) {
        return;
    }

    wgpu::Queue queue = wgfx::device.getQueue();

    if (!pendingTopLevelUpdates.empty()) {
        for (const auto& update : pendingTopLevelUpdates) {
            wgpu::ImageCopyTexture destination = {};
            destination.texture = topLevelTexture3D.texture;
            destination.mipLevel = 0;
            destination.origin = {update.topX, update.topY, update.topZ};
            destination.aspect = wgpu::TextureAspect::All;

            wgpu::TextureDataLayout source = {};
            source.offset = 0;
            source.bytesPerRow = sizeof(uint32_t);
            source.rowsPerImage = 1;

            wgpu::Extent3D copySize = {1, 1, 1};
            queue.writeTexture(destination, &update.value, sizeof(uint32_t), source, copySize);
        }
        pendingTopLevelUpdates.clear();
    }

    const bool hadPendingBrickUpdates = !pendingBrickUpdates.empty();
    if (hadPendingBrickUpdates) {
        for (const auto& brickUpdate : pendingBrickUpdates) {
            size_t brickOffset = brickUpdate.brickIndex * 512 * sizeof(uint16_t);
            compute->updateStorageBuffer(brickPoolUniform,
                                         brickUpdate.data, 512 * sizeof(uint16_t), brickOffset);
        }
        pendingBrickUpdates.clear();
    }

    if (!pendingSectorBrickFlush.empty() && compute && sectorCoordUniform) {
        for (const uint64_t localKey : pendingSectorBrickFlush) {
            const int localSectorX = static_cast<int>(localKey & 0xFFFFu);
            const int localSectorY = static_cast<int>((localKey >> 16u) & 0xFFFFu);
            const int localSectorZ = static_cast<int>((localKey >> 32u) & 0xFFFFu);
            if (localSectorX < 0 || localSectorX >= sectorsPerDimension ||
                localSectorY < 0 || localSectorY >= sectorsPerDimension ||
                localSectorZ < 0 || localSectorZ >= sectorsPerDimension) {
                continue;
            }

            const int localSectorIndex =
                localSectorX +
                localSectorY * sectorsPerDimension +
                localSectorZ * sectorsPerDimension * sectorsPerDimension;
            if (localSectorIndex < 0 || localSectorIndex >= sectorCount) {
                continue;
            }

            if (sectorCoordMap[localSectorIndex * 4 + 3] != 0) {
                continue;
            }

            sectorCoordMap[localSectorIndex * 4 + 3] = 1;
            compute->updateStorageBuffer(
                sectorCoordUniform,
                &sectorCoordMap[localSectorIndex * 4],
                4 * sizeof(int32_t),
                static_cast<size_t>(localSectorIndex) * 4 * sizeof(int32_t));
        }
        pendingSectorBrickFlush.clear();
        sectorMapDirty = true;
    }

    if (!pendingVoxelWordIndices.empty()) {
        pendingVoxelEdits.clear();
        pendingVoxelEdits.reserve(pendingVoxelWordIndices.size());
        for (const uint32_t wordIndex : pendingVoxelWordIndices) {
            pendingVoxelEdits.push_back({
                wordIndex,
                getPackedBrickWord(wordIndex),
                0u,
                0u
            });
        }
        pendingVoxelWordIndices.clear();
    }

    if (!pendingVoxelEdits.empty() && voxelEditCompute && voxelEditCountUniform &&
        voxelEditBufferUniform && wgfx::encoder) {
        size_t remaining = pendingVoxelEdits.size();
        size_t offset = 0;
        while (remaining > 0) {
            const size_t chunkSize = std::min(remaining, MAX_VOXEL_EDIT_BATCH);
            voxelEditCountData[0] = static_cast<uint32_t>(chunkSize);
            voxelEditCompute->updateStorageBuffer(voxelEditCountUniform, voxelEditCountData);
            voxelEditCompute->updateStorageBuffer(
                voxelEditBufferUniform,
                pendingVoxelEdits.data() + offset,
                chunkSize * sizeof(VoxelEditCommand),
                0);

            wgfx::ComputePass editPass;
            editPass.prepare();
            const uint32_t groups = static_cast<uint32_t>((chunkSize + 63) / 64);
            editPass.draw(voxelEditCompute, groups);
            editPass.end();

            offset += chunkSize;
            remaining -= chunkSize;
        }
        pendingVoxelEdits.clear();
    }

    if (sectorMapDirty && rebuildSectorMapGPU()) {
        sectorMapDirty = false;
    }

    queue.release();
}

void VoxelCompute::printVRAMUsage(const char* context) {
    size_t topLevelVRAM = topLevelCount * sizeof(uint32_t);
    size_t brickPoolVRAM = maxBrickPoolSize * 512 * sizeof(uint16_t);
    size_t outputTextureVRAM = outputTexture.width * outputTexture.height * 4;

    size_t usedBricks = currentBrickPoolSize;
    size_t usedBrickVRAM = usedBricks * 512 * sizeof(uint16_t);

    size_t totalAllocatedVRAM = topLevelVRAM + brickPoolVRAM + outputTextureVRAM;
    size_t totalUsedVRAM = topLevelVRAM + usedBrickVRAM + outputTextureVRAM;

    printf("[VRAM Usage%s%s] - OPTIMIZED SYSTEM\n", context[0] ? " - " : "", context);
    printf("  Brick pool allocated: %.2f MB (%zu max bricks × 1024 bytes)\n",
           brickPoolVRAM / (1024.0f * 1024.0f), maxBrickPoolSize);
    printf("  Brick pool used:      %.2f MB (%zu used bricks × 1024 bytes)\n",
           usedBrickVRAM / (1024.0f * 1024.0f), usedBricks);
    printf("  Output texture:       %.2f MB (%dx%d RGBA8)\n",
           outputTextureVRAM / (1024.0f * 1024.0f), outputTexture.width, outputTexture.height);
    printf("  Total allocated:      %.2f MB\n", totalAllocatedVRAM / (1024.0f * 1024.0f));
    printf("  Total actually used:  %.2f MB (%.1f%% efficiency)\n",
           totalUsedVRAM / (1024.0f * 1024.0f),
           (totalUsedVRAM * 100.0f) / totalAllocatedVRAM);
    printf("  Brick pool efficiency: %.1f%% (%zu/%zu bricks used)\n",
           (usedBricks * 100.0f) / maxBrickPoolSize, usedBricks, maxBrickPoolSize);
    printf("  Circular buffer usage: %zu/%zu entries\n", freeList.size(), freeList.BUFFER_SIZE);
    printf("  Pending updates: %zu top-level, %zu brick, %zu voxel edits\n\n",
           pendingTopLevelUpdates.size(), pendingBrickUpdates.size(), pendingVoxelEdits.size());
}

void VoxelCompute::updateGPUBuffers(int x, int y, int z) {
    queueTopLevelUpdate(x, y, z);
}

void VoxelCompute::updateGPUBuffersForCells(const std::set<size_t>& affectedCells) {
    for (size_t topIndex : affectedCells) {
        int topZ = topIndex / (topLevelSize * topLevelSize);
        int topY = (topIndex % (topLevelSize * topLevelSize)) / topLevelSize;
        int topX = topIndex % topLevelSize;

        int worldX = topX * brickSize;
        int worldY = topY * brickSize;
        int worldZ = topZ * brickSize;

        queueTopLevelUpdate(worldX, worldY, worldZ);
    }

    flushBatchedUpdates();
}

void VoxelCompute::render(const glm::vec3& cameraPos, const glm::mat4& invViewProj, const glm::vec2& resolution) {
    static int frameCount = 0;
    static size_t lastBrickCount = 0;

    frameCount++;
    bool shouldPrintVRAM = (frameCount == 1) || (frameCount % 60 == 0) || (currentBrickPoolSize != lastBrickCount);

    if (shouldPrintVRAM) {
        lastBrickCount = currentBrickPoolSize;
    }

    streamTerrainAroundPlayer(cameraPos, invViewProj);
    flushBatchedUpdates();

    if (frameCount > 1) {
        topLevelDoubleBuffer.swap();
        memcpy(topLevelDoubleBuffer.currentWrite, topLevelGrid, topLevelCount * sizeof(uint32_t));
    }

    wgfx::ComputePass computePass;
    computePass.prepare();

    compute->updateUniform(0, glm::value_ptr(cameraPos));
    compute->updateUniform(2, glm::value_ptr(invViewProj));

    const int maxNoAliasRadius = std::max(1, (sectorsPerDimension - 1) / 2);
    const int effectiveStreamRadius = std::min(std::max(1, terrainStreamRadiusSectors), maxNoAliasRadius);
    const float fogDistance = static_cast<float>(effectiveStreamRadius * sectorSize) * 0.95f;
    const float fogControl = enableSkyFog ? fogDistance : -fogDistance;
    glm::vec4 resData(resolution.x, resolution.y, showHierarchy ? 1.0f : 0.0f, fogControl);
    compute->updateUniform(3, glm::value_ptr(resData));

    compute->uniforms.clear();

    uint32_t workgroupsX = (uint32_t(resolution.x) + 7) / 8;
    uint32_t workgroupsY = (uint32_t(resolution.y) + 7) / 8;
    computePass.drawXY(compute, workgroupsX, workgroupsY);
    computePass.end();
}

namespace compute_layers {

void render(VoxelCompute& volume, const glm::vec3& cameraPos, const glm::mat4& invViewProj, const glm::vec2& resolution) {
    volume.render(cameraPos, invViewProj, resolution);
}

void flushGpuUpdates(VoxelCompute& volume) {
    volume.flushBatchedUpdates();
}

void updateVoxelGpuCell(VoxelCompute& volume, int x, int y, int z) {
    volume.updateGPUBuffers(x, y, z);
}

} // namespace compute_layers
