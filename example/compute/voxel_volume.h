#pragma once

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <chrono>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../constants.h"
#include "../itexturemanager.h"
#include "../model.h"
#include "../voxelizer.h"
#include "wgfx.h"

extern "C" {
    unsigned char* stbi_load(char const* filename, int* x, int* y, int* channels_in_file, int desired_channels);
    unsigned char* stbi_load_from_memory(const unsigned char* buffer, int len, int* x, int* y, int* channels_in_file, int desired_channels);
    void stbi_image_free(void* retval_from_stbi_load);
    const char* stbi_failure_reason(void);
}

class VoxelCompute {
public:
    static constexpr uint16_t PLAYER_PLACED_MATERIAL = 9;
    static constexpr uint16_t SOUND_VOXEL_MATERIAL = 10;

    static VoxelCompute& Instance();

    struct ColorPalette {
        static constexpr int MAX_COLORS = 256;
        uint32_t colors[MAX_COLORS];
        int colorCount = 0;
    };

    struct alignas(64) BrickMeta {
        uint16_t nonEmptyCount = 0;
        bool inUse = false;
        char padding[61];
    };

    struct CircularBuffer {
        static constexpr size_t BUFFER_SIZE = 524288;
        uint32_t buffer[BUFFER_SIZE];
        size_t head = 0;
        size_t tail = 0;
        size_t count = 0;

        bool push(uint32_t value);
        bool pop(uint32_t* value);
        bool empty() const;
        size_t size() const;
    };

    struct BatchedUpdate {
        uint32_t topX, topY, topZ;
        uint32_t value;
    };

    struct BrickUpdate {
        uint32_t brickIndex;
        uint16_t data[512];
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

        void init(size_t size);
        void swap();
        ~DoubleBuffer();
    };

    struct TextureData {
        unsigned char* data;
        int width;
        int height;
        int channels;

        TextureData();
        ~TextureData();
    };

    struct SectorCandidate {
        int x = 0;
        int y = 0;
        int z = 0;
        int distSq = 0;
    };

    struct AudioRaytraceOutput {
        float directGain = 0.0f;
        float directDistance = 0.0f;
        bool occluded = false;
        std::array<glm::vec2, 4> reflectionTaps{}; // x = delaySeconds, y = gain
        int reflectionCount = 0;
    };

    struct AudioReadbackSlot {
        wgpu::Buffer buffer;
        std::array<float, 24> cpuData{};
        std::unique_ptr<wgpu::BufferMapCallback> callback;
        enum class State {
            Idle,
            AwaitingMap,
            Mapping,
        };
        State state = State::Idle;
    };

    class TextureManager : public ITextureManager {
    public:
        uint32_t getTextureFromMemory(const unsigned char* data, int length, const std::string& name) override;
        uint32_t getTextureIndex(const std::string& texturePath) override;
        glm::vec3 sampleTexture(uint32_t textureIndex, const glm::vec2& uv) const override;
        size_t getTextureCount() const;
        const TextureData* getTextureData(uint32_t textureIndex) const;
        void printMemoryUsage() const;

    private:
        std::vector<std::unique_ptr<TextureData>> textureArray;
        std::map<std::string, uint32_t> pathToIndex;
    };

    uint32_t packColor(uint8_t r, uint8_t g, uint8_t b);
    static uint32_t packRGB(uint8_t r, uint8_t g, uint8_t b);

    size_t getTopLevelIndex(int x, int y, int z) const;
    size_t getSectorIndex(int x, int y, int z) const;
    size_t getBrickVoxelOffset(int localX, int localY, int localZ) const;

    void updateSector(int sectorX, int sectorY, int sectorZ);
    void updateAllSectors();
    void updateSectorForVoxel(int x, int y, int z);

    uint32_t allocateBrickInternal();
    void freeBrickInternal(uint32_t brickIndex);
    uint32_t allocateBrick();
    void freeBrick(uint32_t brickIndex);
    uint32_t allocateBrickAtomic();

    void testVoxelPlacement();
    void testCircularBuffer();

    void copyTopLevelGridToTexture();
    bool rebuildSectorMapGPU();
    uint32_t getPackedBrickWord(uint32_t wordIndex) const;
    void queueTopLevelUpdate(int x, int y, int z);
    void flushBatchedUpdates();
    void printVRAMUsage(const char* context = "");

    void initializeTerrainPalette();
    float fbm2D(glm::vec2 pos, int octaves, float lacunarity, float gain) const;
    float ridgeFbm2D(glm::vec2 pos, int octaves, float lacunarity, float gain) const;
    float fbm3D(glm::vec3 pos, int octaves, float lacunarity, float gain) const;
    int sampleTerrainHeight(int x, int z) const;
    uint16_t chooseTerrainMaterial(
        int x,
        int y,
        int z,
        int surfaceY,
        int dx,
        int dz,
        int minTerrainHeight,
        int maxTerrainHeight) const;
    void generateProceduralTerrain();
    void initializeTerrainStreaming();
    void streamTerrainAroundPlayer(const glm::vec3& cameraPos, const glm::mat4& invViewProj);
    void updateResidentSectorBounds();
    std::string makeSectorKey(int sectorX, int sectorY, int sectorZ) const;
    uint64_t packLocalSectorKey(int localSectorX, int localSectorY, int localSectorZ) const;
    int wrapSectorCoord(int sectorCoord) const;
    bool isSectorInCameraFrustum(
        int globalSectorX,
        int globalSectorY,
        int globalSectorZ,
        const glm::mat4& viewProj,
        const glm::vec3& cameraPos) const;
    bool getSectorFrustumVisibility(
        int globalSectorX,
        int globalSectorY,
        int globalSectorZ,
        const glm::mat4& viewProj,
        const glm::vec3& cameraPos,
        bool refreshCache);
    void rebuildOcclusionHiZ(const glm::mat4& viewProj, const glm::vec3& cameraPos);
    bool isSectorOccludedByHiZ(
        int globalSectorX,
        int globalSectorY,
        int globalSectorZ,
        const glm::mat4& viewProj,
        const glm::vec3& cameraPos) const;
    uint32_t getOrComputeSectorNonAirCount(const std::string& sectorKey) const;
    bool shouldDiscardSectorGpuUpload(int globalSectorX, int globalSectorY, int globalSectorZ) const;
    void clearLocalSectorSlot(int localSectorX, int localSectorY, int localSectorZ);
    void generateSphereMask();
    void rebuildSectorCandidateCache(int centerSectorX, int centerSectorY, int centerSectorZ);
    void ensureSectorResident(int globalSectorX, int globalSectorY, int globalSectorZ);
    void generateProceduralTerrainSector(
        int globalSectorX,
        int globalSectorY,
        int globalSectorZ,
        int localSectorX,
        int localSectorY,
        int localSectorZ);

    void setVoxelInternal(int x, int y, int z, uint16_t voxelData, bool updateSector);
    void setVoxel(int x, int y, int z, uint16_t voxelData);
    void clearVoxel(int x, int y, int z);

    void buildColorPalette(const std::set<uint32_t>& uniqueColors);
    uint8_t findClosestPaletteIndex(uint32_t color) const;
    void voxelizeTriangles(const std::vector<Triangle>& triangles);
    void voxelizeTrianglesLegacy(const std::vector<Triangle>& triangles);
    bool isPointInTriangle(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2);
    glm::vec3 calculateBarycentricCoords(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2);

    wgfx::Texture createOwlGPUTexture();
    glm::vec3 sampleTexture(const std::string& texturePath, const glm::vec2& uv);

    bool isVoxelOccupied(int x, int y, int z) const;
    uint16_t getVoxel(int x, int y, int z) const;
    bool voxelIntersectsCapsule(
        int voxelX,
        int voxelY,
        int voxelZ,
        const glm::vec3& capsuleFeetPosition,
        const glm::vec3& capsuleSize) const;
    bool placementWouldIntersectPlayer(
        const glm::ivec3& center,
        int toolMode,
        const glm::vec3& playerFeetPosition,
        const glm::vec3& playerSize) const;
    bool raycast(
        const glm::vec3& origin,
        const glm::vec3& direction,
        float length,
        bool add,
        int toolMode = 1,
        bool preventPlayerIntersection = false,
        const glm::vec3& playerFeetPosition = glm::vec3(0.0f),
        const glm::vec3& playerSize = glm::vec3(0.0f));

    uint16_t getMaterialForToolMode(int toolMode) const;

    void updateGPUBuffers(int x, int y, int z);
    void updateGPUBuffersForCells(const std::set<size_t>& affectedCells);
    void render(const glm::vec3& cameraPos, const glm::mat4& invViewProj, const glm::vec2& resolution);
    void setAudioRaytraceRequest(const glm::vec3& sourcePos, const glm::vec3& listenerPos, float maxDistance);
    void runAudioRaytraceGPU();
    void processAudioRaytraceReadbacks();
    AudioRaytraceOutput getLatestAudioRaytrace() const;

    VoxelCompute();
    ~VoxelCompute();

    VoxelCompute(VoxelCompute const&) = delete;
    void operator=(VoxelCompute const&) = delete;

public:
    std::mutex brickMutex;
    bool showHierarchy = false;
    bool enableSkyFog = true;

    const int topLevelSize = 64;
    const int brickSize = 8;
    const int bricksPerSector = 4;
    const int sectorSize = brickSize * bricksPerSector;
    const int sectorsPerDimension = topLevelSize / bricksPerSector;
    const int worldSize = topLevelSize * brickSize;
    const int sectorCount = sectorsPerDimension * sectorsPerDimension * sectorsPerDimension;

    const size_t topLevelCount = topLevelSize * topLevelSize * topLevelSize;
    uint32_t* topLevelGrid = new uint32_t[topLevelCount];

    uint32_t* sectorMap = new uint32_t[sectorCount];
    int32_t* sectorCoordMap = new int32_t[sectorCount * 4];
    int32_t* sectorBoundsData = new int32_t[8];
    wgfx::Uniform* sectorUniform = nullptr;
    wgfx::Uniform* sectorCoordUniform = nullptr;
    wgfx::Uniform* sectorBoundsUniform = nullptr;
    bool sectorMapDirty = true;

    const size_t maxBrickPoolSize = 524288;
    std::atomic<size_t> currentBrickPoolSize{0};

    uint16_t* brickPool = nullptr;
    BrickMeta* brickMetadata = nullptr;
    CircularBuffer freeList;

    std::vector<BatchedUpdate> pendingTopLevelUpdates;
    std::vector<BrickUpdate> pendingBrickUpdates;
    std::vector<VoxelEditCommand> pendingVoxelEdits;
    std::set<uint32_t> pendingVoxelWordIndices;
    static constexpr size_t MAX_BATCH_SIZE = 1024;
    static constexpr size_t MAX_VOXEL_EDIT_BATCH = 1 << 20;
    uint32_t voxelEditCountData[1] = { 0 };

    DoubleBuffer topLevelDoubleBuffer;

    static constexpr uint32_t BRICK_INDEX_MASK = 0x7FFFFFFF;
    static constexpr uint32_t IS_BRICK_FLAG = 0x80000000;

    wgfx::Compute* compute = nullptr;
    wgfx::Compute* sectorCompute = nullptr;
    wgfx::Compute* voxelEditCompute = nullptr;
    wgfx::Compute* audioCompute = nullptr;
    wgfx::Texture outputTexture;
    wgfx::Texture topLevelTexture3D;
    wgfx::Uniform* topLevelTextureUniform = nullptr;
    wgfx::Uniform* brickPoolUniform = nullptr;
    wgfx::Uniform* voxelEditCountUniform = nullptr;
    wgfx::Uniform* voxelEditBufferUniform = nullptr;
    wgfx::Uniform* audioParamsUniform = nullptr;
    wgfx::Uniform* audioOutputUniform = nullptr;

    std::array<float, 24> audioOutputCPU{};
    mutable AudioRaytraceOutput latestAudioRaytrace;
    mutable std::mutex audioRaytraceMutex;
    bool audioRaytracePending = false;
    static constexpr uint32_t AUDIO_READBACK_SLOT_COUNT = 3;
    std::array<AudioReadbackSlot, AUDIO_READBACK_SLOT_COUNT> audioReadbackSlots;
    uint32_t audioReadbackWriteIndex = 0;
    std::array<float, 16> audioParamsData{};

    ColorPalette colorPalette;
    std::vector<glm::ivec3> soundVoxelPositions;
    mutable TextureManager textureManager;

    bool terrainStreamingEnabled = true;
    int terrainStreamRadiusSectors = 36;
    size_t terrainStreamedSectorCount = 0;
    std::unordered_map<std::string, uint64_t> residentGlobalToLocal;
    std::unordered_map<uint64_t, std::string> residentLocalToGlobal;
    std::unordered_map<std::string, std::vector<uint16_t>> generatedSectorVoxelData;
    mutable std::unordered_map<std::string, uint32_t> generatedSectorNonAirCounts;
    std::unordered_set<uint64_t> pendingSectorBrickFlush;
    std::vector<SectorCandidate> sectorCandidates;
    std::vector<SectorCandidate> sphereMask;
    int lastCachedCenterSectorX = INT32_MIN;
    int lastCachedCenterSectorY = INT32_MIN;
    int lastCachedCenterSectorZ = INT32_MIN;
    int streamAnchorSectorX = INT32_MIN;
    int streamAnchorSectorY = INT32_MIN;
    int streamAnchorSectorZ = INT32_MIN;
    float streamCenterHysteresisBlocks = 1.0f;
    int cachedSphereMaskRadius = -1;
    size_t sectorCandidateCursor = 0;
    std::unordered_map<std::string, bool> sectorFrustumVisibilityCache;
    std::chrono::steady_clock::time_point lastFrustumRefreshTime = std::chrono::steady_clock::time_point::min();
    float frustumRefreshIntervalSeconds = 0.1f;

    int hiZBaseWidth = 128;
    int hiZBaseHeight = 72;
    float hiZOccluderFillRatioThreshold = 0.30f;
    float hiZOcclusionDepthBias = 0.01f;
    std::vector<float> hiZBaseDepth;
    std::vector<std::vector<float>> hiZMips;
    std::chrono::steady_clock::time_point lastOcclusionRefreshTime = std::chrono::steady_clock::time_point::min();
    float occlusionRefreshIntervalSeconds = 0.1f;
};
