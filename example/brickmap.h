#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include "wgfx.h"

class BrickmapState {
public:
    // Two-level hierarchical brickmap configuration.
    static constexpr int topLevelSize = 64;
    static constexpr int brickSize = 8;
    static constexpr int bricksPerSector = 64;
    static constexpr int sectorSize = brickSize * bricksPerSector;
    static constexpr int sectorsPerDimension = topLevelSize / bricksPerSector;
    static constexpr int worldSize = topLevelSize * brickSize;
    static constexpr int sectorCount = sectorsPerDimension * sectorsPerDimension * sectorsPerDimension;

    static constexpr size_t topLevelCount =
        static_cast<size_t>(topLevelSize) * topLevelSize * topLevelSize;

    struct ColorPalette {
        static constexpr int MAX_COLORS = 256;
        uint32_t colors[MAX_COLORS] = {0};
        int colorCount = 0;
    };

    struct alignas(64) BrickMeta {
        uint16_t nonEmptyCount = 0;
        bool inUse = false;
        char padding[61] = {0};
    };

    struct CircularBuffer {
        static constexpr size_t BUFFER_SIZE = 524288;
        uint32_t buffer[BUFFER_SIZE] = {0};
        size_t head = 0;
        size_t tail = 0;
        size_t count = 0;

        bool push(uint32_t value) {
            if (count >= BUFFER_SIZE) return false;
            buffer[tail] = value;
            tail = (tail + 1) & (BUFFER_SIZE - 1);
            count++;
            return true;
        }

        bool pop(uint32_t* value) {
            if (count == 0) return false;
            *value = buffer[head];
            head = (head + 1) & (BUFFER_SIZE - 1);
            count--;
            return true;
        }

        bool empty() const { return count == 0; }
        size_t size() const { return count; }
    };

    struct BatchedUpdate {
        uint32_t topX = 0;
        uint32_t topY = 0;
        uint32_t topZ = 0;
        uint32_t value = 0;
    };

    struct BrickUpdate {
        uint32_t brickIndex = 0;
        uint16_t data[512] = {0};
    };

    struct VoxelEditCommand {
        uint32_t wordIndex = 0;
        uint32_t wordValue = 0;
        uint32_t pad = 0;
        uint32_t pad2 = 0;
    };

    struct DoubleBuffer {
        uint32_t* buffer1 = nullptr;
        uint32_t* buffer2 = nullptr;
        uint32_t* currentWrite = nullptr;
        uint32_t* currentRead = nullptr;
        bool useBuffer1 = true;

        void init(size_t size) {
            buffer1 = new uint32_t[size];
            buffer2 = new uint32_t[size];
            currentWrite = buffer1;
            currentRead = buffer2;
        }

        void swap() {
            if (useBuffer1) {
                currentWrite = buffer2;
                currentRead = buffer1;
            } else {
                currentWrite = buffer1;
                currentRead = buffer2;
            }
            useBuffer1 = !useBuffer1;
        }

        ~DoubleBuffer() {
            delete[] buffer1;
            delete[] buffer2;
        }
    };

    static constexpr size_t maxBrickPoolSize = 524288;
    static constexpr size_t MAX_BATCH_SIZE = 1024;
    static constexpr size_t MAX_VOXEL_EDIT_BATCH = 1 << 20;

    static constexpr uint32_t BRICK_INDEX_MASK = 0x7FFFFFFFu;
    static constexpr uint32_t IS_BRICK_FLAG = 0x80000000u;

    std::mutex brickMutex;
    bool showHierarchy = false;

    ColorPalette colorPalette;

    uint32_t* topLevelGrid = nullptr;
    uint32_t* sectorMap = nullptr;

    wgfx::Uniform* sectorUniform = nullptr;
    bool sectorMapDirty = true;

    std::atomic<size_t> currentBrickPoolSize{0};
    uint16_t* brickPool = nullptr;
    BrickMeta* brickMetadata = nullptr;

    CircularBuffer freeList;

    std::vector<BatchedUpdate> pendingTopLevelUpdates;
    std::vector<BrickUpdate> pendingBrickUpdates;
    std::vector<VoxelEditCommand> pendingVoxelEdits;
    std::set<uint32_t> pendingVoxelWordIndices;
    std::set<uint32_t> dirtySectorIndices;
    std::set<uint32_t> queuedBrickIndicesForVoxelUpload;
    std::set<uint32_t> sectorsPendingVoxelUpload;

    uint32_t voxelEditCountData[1] = {0};

    DoubleBuffer topLevelDoubleBuffer;

    BrickmapState();
    ~BrickmapState();

    uint32_t packColor(uint8_t r, uint8_t g, uint8_t b) const {
        return (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
    }

    size_t getTopLevelIndex(int x, int y, int z) const {
        int topX = x / brickSize;
        int topY = y / brickSize;
        int topZ = z / brickSize;
        return static_cast<size_t>(topX + topY * topLevelSize + topZ * topLevelSize * topLevelSize);
    }

    size_t getSectorIndex(int x, int y, int z) const {
        int sectorX = x / sectorSize;
        int sectorY = y / sectorSize;
        int sectorZ = z / sectorSize;
        return static_cast<size_t>(sectorX + sectorY * sectorsPerDimension +
                                   sectorZ * sectorsPerDimension * sectorsPerDimension);
    }

    void updateSector(int sectorX, int sectorY, int sectorZ);
    void updateAllSectors();
    void queueDirtySectorUploads();
    void uploadDirtySectorVoxels();

    // Internal allocation (no mutex - caller must hold lock)
    uint32_t allocateBrickInternal();

    // Internal free (no mutex - caller must hold lock)
    void freeBrickInternal(uint32_t brickIndex);

    // Public thread-safe brick allocator
    uint32_t allocateBrick();

    // Public thread-safe free
    void freeBrick(uint32_t brickIndex);

    // Legacy mapping
    uint32_t allocateBrickAtomic();

    void updateSectorForVoxel(int x, int y, int z) {
        int sectorX = x / sectorSize;
        int sectorY = y / sectorSize;
        int sectorZ = z / sectorSize;
        updateSector(sectorX, sectorY, sectorZ);
    }
};
