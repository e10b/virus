#include "voxel_volume.h"

#include <algorithm>
#include <cstring>
#include <string>

VoxelCompute& VoxelCompute::Instance() {
	static VoxelCompute instance;
	return instance;
}

VoxelCompute::VoxelCompute() {
	brickPool = new uint16_t[maxBrickPoolSize * 512];
	brickMetadata = new BrickMeta[maxBrickPoolSize];

	topLevelDoubleBuffer.init(topLevelCount);

	memset(topLevelGrid, 0, topLevelCount * sizeof(uint32_t));
	memcpy(topLevelDoubleBuffer.currentWrite, topLevelGrid, topLevelCount * sizeof(uint32_t));
	memset(sectorMap, 0, sectorCount * sizeof(uint32_t));
	for (int i = 0; i < sectorCount; ++i) {
		sectorCoordMap[i * 4 + 0] = 0;
		sectorCoordMap[i * 4 + 1] = 0;
		sectorCoordMap[i * 4 + 2] = 0;
		sectorCoordMap[i * 4 + 3] = 0; // validity flag
	}
	// Sector bounds are packed as two vec4<i32>: [minX,minY,minZ,valid], [maxX,maxY,maxZ,pad]
	sectorBoundsData[0] = 0;
	sectorBoundsData[1] = 0;
	sectorBoundsData[2] = 0;
	sectorBoundsData[3] = 0;
	sectorBoundsData[4] = 0;
	sectorBoundsData[5] = 0;
	sectorBoundsData[6] = 0;
	sectorBoundsData[7] = 0;

	initializeTerrainPalette();

	pendingTopLevelUpdates.reserve(MAX_BATCH_SIZE);
	pendingBrickUpdates.reserve(MAX_BATCH_SIZE);
	pendingVoxelEdits.reserve(MAX_BATCH_SIZE);

	compute = wgfx::loadCompute(
		wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "compute.wgsl").c_str()));

	compute->addUniform(0);
	outputTexture = wgfx::loadTextureSrc(raytraceWidth, raytraceHeight);
	compute->addStorageTexture(1, outputTexture);
	compute->addUniform(2);
	compute->addUniform(3);

	initializeTerrainStreaming();
	printf("Terrain streaming enabled: loading sectors around player each frame (radius=%d sectors).\n",
	       terrainStreamRadiusSectors);

	size_t topLevelSizeMB = (topLevelCount * sizeof(uint32_t)) / (1024 * 1024);
	size_t brickPoolSizeMB = (maxBrickPoolSize * 512) / (1024 * 1024);
	printf("Top-level grid size: %zu entries (%zu MB)\n", topLevelCount, topLevelSizeMB);
	printf("Brick pool size: %zu bricks (%zu MB)\n", maxBrickPoolSize, brickPoolSizeMB);
	printf("Total GPU memory usage: %zu MB\n", topLevelSizeMB + brickPoolSizeMB);

	topLevelTexture3D = wgfx::loadTexture3D(topLevelSize, topLevelSize, topLevelSize, wgpu::TextureFormat::R32Uint);
	sectorMapDirty = true;
	copyTopLevelGridToTexture();

	topLevelTextureUniform = compute->addTexture3D_Uint(4, topLevelTexture3D);
	compute->addSampler(5, topLevelTexture3D);
	brickPoolUniform = compute->addStorage(6, maxBrickPoolSize * 512 * sizeof(uint16_t), brickPool);
	compute->addStorage(7, sizeof(colorPalette.colors), colorPalette.colors);

	sectorUniform = compute->addStorage(10, sectorCount * sizeof(uint32_t), sectorMap, false);
	sectorCoordUniform = compute->addStorage(11, sectorCount * 4 * sizeof(int32_t), sectorCoordMap, false);
	sectorBoundsUniform = compute->addStorage(12, 8 * sizeof(int32_t), sectorBoundsData, false);

	wgfx::Texture owlTexture = createOwlGPUTexture();
	compute->addTexture(8, owlTexture);
	compute->addSampler(9, owlTexture);

	compute->init();

	sectorCompute = wgfx::loadCompute(
		wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "sector_occupancy.wgsl").c_str()));
	sectorCompute->uniforms.visibility = wgpu::ShaderStage::Compute;
	sectorCompute->uniforms.setTexture(topLevelTextureUniform);
	sectorCompute->uniforms.setStorage(sectorUniform);
	sectorCompute->init();

	voxelEditCompute = wgfx::loadCompute(
		wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "voxel_edits.wgsl").c_str()));
	voxelEditCountUniform = voxelEditCompute->addStorage(11, sizeof(voxelEditCountData), voxelEditCountData, true);
	voxelEditBufferUniform = voxelEditCompute->addStorage(
		12,
		MAX_VOXEL_EDIT_BATCH * sizeof(VoxelEditCommand),
		nullptr,
		true);
	voxelEditCompute->uniforms.visibility = wgpu::ShaderStage::Compute;
	voxelEditCompute->uniforms.setStorage(brickPoolUniform);
	voxelEditCompute->init();

	audioCompute = wgfx::loadCompute(
		wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "audio_trace.wgsl").c_str()));
	audioParamsUniform = audioCompute->addUniform(0);
	audioOutputUniform = audioCompute->addStorage(1, sizeof(audioOutputCPU), audioOutputCPU.data(), false);
	audioCompute->uniforms.visibility = wgpu::ShaderStage::Compute;
	audioCompute->uniforms.setTexture(topLevelTextureUniform);
	audioCompute->uniforms.setStorage(brickPoolUniform);
	audioCompute->uniforms.setStorage(sectorUniform);
	audioCompute->uniforms.setStorage(sectorCoordUniform);
	audioCompute->init();

	wgpu::BufferDescriptor readbackDesc = {};
	readbackDesc.size = sizeof(audioOutputCPU);
	readbackDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
	readbackDesc.mappedAtCreation = false;
	for (auto& slot : audioReadbackSlots) {
		slot.buffer = wgfx::device.createBuffer(readbackDesc);
		slot.cpuData.fill(0.0f);
		slot.state = AudioReadbackSlot::State::Idle;
	}

	audioParamsData.fill(0.0f);
}

VoxelCompute::~VoxelCompute() {
	for (auto& slot : audioReadbackSlots) {
		slot.callback.reset();
		slot.buffer.release();
	}

	delete[] topLevelGrid;
	delete[] brickPool;
	delete[] brickMetadata;
	delete[] sectorCoordMap;
	delete[] sectorBoundsData;
}

void VoxelCompute::setAudioRaytraceRequest(const glm::vec3& sourcePos, const glm::vec3& listenerPos, float maxDistance)
{
	audioParamsData.fill(0.0f);
	audioParamsData[0] = sourcePos.x;
	audioParamsData[1] = sourcePos.y;
	audioParamsData[2] = sourcePos.z;
	audioParamsData[4] = listenerPos.x;
	audioParamsData[5] = listenerPos.y;
	audioParamsData[6] = listenerPos.z;
	audioParamsData[7] = maxDistance;
	audioParamsData[8] = 343.0f * World::voxelsPerMeter;
	audioRaytracePending = true;
}

void VoxelCompute::runAudioRaytraceGPU()
{
	if (!audioCompute || !audioRaytracePending || !wgfx::encoder) {
		return;
	}

	audioCompute->updateUniform(0, audioParamsData.data());

	wgfx::ComputePass pass;
	pass.prepare();
	pass.drawXYZ(audioCompute, 1, 1, 1);
	pass.end();
	audioRaytracePending = false;

	AudioReadbackSlot& slot = audioReadbackSlots[audioReadbackWriteIndex];
	if (slot.state != AudioReadbackSlot::State::Idle) {
		return;
	}

	wgfx::encoder.copyBufferToBuffer(audioOutputUniform->buffer, 0, slot.buffer, 0, sizeof(audioOutputCPU));
	slot.state = AudioReadbackSlot::State::AwaitingMap;

	audioReadbackWriteIndex = (audioReadbackWriteIndex + 1) % AUDIO_READBACK_SLOT_COUNT;
}

void VoxelCompute::processAudioRaytraceReadbacks()
{
	for (uint32_t slotIndex = 0; slotIndex < AUDIO_READBACK_SLOT_COUNT; ++slotIndex) {
		AudioReadbackSlot& slot = audioReadbackSlots[slotIndex];
		if (slot.state != AudioReadbackSlot::State::AwaitingMap) {
			continue;
		}

		slot.state = AudioReadbackSlot::State::Mapping;
		slot.callback = slot.buffer.mapAsync(
			wgpu::MapMode::Read,
			0,
			sizeof(audioOutputCPU),
			[this, slotIndex](wgpu::BufferMapAsyncStatus status) {
				AudioReadbackSlot& cbSlot = audioReadbackSlots[slotIndex];
				if (status == wgpu::BufferMapAsyncStatus::Success) {
					const void* mapped = cbSlot.buffer.getMappedRange(0, sizeof(audioOutputCPU));
					if (mapped) {
						std::memcpy(cbSlot.cpuData.data(), mapped, sizeof(audioOutputCPU));
					}
					cbSlot.buffer.unmap();

					AudioRaytraceOutput parsed;
					parsed.directGain = std::max(0.0f, cbSlot.cpuData[0]);
					parsed.directDistance = std::max(0.0f, cbSlot.cpuData[1]);
					parsed.occluded = cbSlot.cpuData[2] > 0.5f;
					parsed.reflectionCount = std::clamp(static_cast<int>(cbSlot.cpuData[3]), 0, 4);
					for (int i = 0; i < 4; ++i) {
						parsed.reflectionTaps[i] = glm::vec2(
							std::max(0.0f, cbSlot.cpuData[4 + i * 4 + 0]),
							std::max(0.0f, cbSlot.cpuData[4 + i * 4 + 1]));
					}

					{
						std::lock_guard<std::mutex> lock(audioRaytraceMutex);
						latestAudioRaytrace = parsed;
					}
				}

				cbSlot.state = AudioReadbackSlot::State::Idle;
				cbSlot.callback.reset();
			});
	}
}

VoxelCompute::AudioRaytraceOutput VoxelCompute::getLatestAudioRaytrace() const
{
	std::lock_guard<std::mutex> lock(audioRaytraceMutex);
	return latestAudioRaytrace;
}
