#pragma once

const int raytraceWidth = 1280;
const int raytraceHeight = 720;
namespace World
{


	const float renderDistance = 200.0f; // block render radius
	const unsigned renderSpeed = 1; // chunks generated per frame

	// Player scale: 24 voxels = 2 meters, so 1 meter = 12 voxels.
	const float voxelsPerMeter = 12.0f;

	// Earth gravity converted to voxel units: 9.8 m/s^2 * 12 voxels/m = 117.6 voxels/s^2.
	const float gravity = 9.8f * voxelsPerMeter;

	// Chunk dimensions
	const unsigned chunkHeight = 256;
	const unsigned chunkSize = 16;
	const unsigned chunkArea = chunkSize * chunkSize;
	const unsigned chunkVolume = chunkArea * chunkHeight;

	// Configurable world gen variables
	namespace Generation
	{
		// Lowest block elevation possible
		const unsigned minHeight = 1;

		// Sea level for water generation
		const unsigned seaLevel = 30;

		// Sand generation range around sea level
		const unsigned sandDepth = 5; // How deep sand goes below sea level
		const unsigned sandHeight = 3; // How high sand goes above sea level

		// Biome noise
		const float landScale = 2048.0f;
		const float landMinMult = 0.1f;
		const float landTransitionSharpness = 2.0f;
		const float landMountainBias = 0.2f; // -1 (flat) to 1 (mountains)

		// Mountain noise
		const float heightScale = 256.0f;
		const float heightWeight = 0.8f;
		const unsigned heightMaxHeight = 250;

		// Hill noise
		const float detailedScale = 32.0f;
		const float detailWeight = 1.0f - heightWeight;
		const unsigned detailMaxHeight = 100;

		// Trees
		const float treeDensity = 0.01f;//0.03f;

		// Interpolation grid size
		const float terrainInterpGrid = 4.0f;
	}
}