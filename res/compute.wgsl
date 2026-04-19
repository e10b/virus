
@group(0) @binding(0) var<uniform> position: vec3<f32>;
@group(0) @binding(1) var outTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> invViewProj: mat4x4<f32>;
@group(0) @binding(3) var<uniform> res: vec4<f32>;
@group(0) @binding(4) var topLevelGridTex: texture_3d<u32>;
@group(0) @binding(5) var topLevelSampler: sampler;
@group(0) @binding(6) var<storage, read_write> brickPool: array<u32>;
@group(0) @binding(7) var<storage, read_write> colorPalette: array<u32, 256>; // 256-color palette

@group(0) @binding(8) var owlTexture: texture_2d<f32>;
@group(0) @binding(9) var owlSampler: sampler;
@group(0) @binding(10) var<storage, read_write> sectorMap: array<u32>;
@group(0) @binding(11) var<storage, read_write> sectorCoords: array<vec4<i32>>;
@group(0) @binding(12) var<storage, read_write> sectorBounds: array<vec4<i32>, 2>;

// Hierarchical brickmap constants (all powers of 2 for efficient operations)
const TOP_LEVEL_SIZE = 64; // 64³ top-level grid (power of 2)
const BRICK_SIZE = 8; // 8³ bricks (power of 2)
const BRICKS_PER_SECTOR = 4; // 4³ bricks per sector
const SECTOR_SIZE = BRICK_SIZE * BRICKS_PER_SECTOR; // 32 voxels per sector
const SECTORS_PER_DIM = TOP_LEVEL_SIZE / BRICKS_PER_SECTOR; // 16 sectors per dimension
const WORLD_SIZE = TOP_LEVEL_SIZE * BRICK_SIZE; // 512³ world
const MAX_BRICK_POOL_SIZE = 524288u; // 512K bricks (power of 2)
const TOP_CACHE_DIM = 4; // 4x4x4 top-level tile in shared memory
const TOP_CACHE_COUNT = TOP_CACHE_DIM * TOP_CACHE_DIM * TOP_CACHE_DIM;
const ENABLE_VOXEL_AO = true;
const ENABLE_VOXEL_SHADOWS = true;

// Bit manipulation constants
const BRICK_INDEX_MASK = 0x7FFFFFFFu; // Bits 0-30
const IS_BRICK_FLAG = 0x80000000u;    // Bit 31

var<workgroup> topLevelCache: array<u32, TOP_CACHE_COUNT>;

// Get top-level grid index from world coordinates
fn getTopLevelIndex(pos: vec3<i32>) -> u32 {
    let topX = pos.x / BRICK_SIZE;
    let topY = pos.y / BRICK_SIZE;
    let topZ = pos.z / BRICK_SIZE;
    return u32(topX + topY * TOP_LEVEL_SIZE + topZ * TOP_LEVEL_SIZE * TOP_LEVEL_SIZE);
}

// Sample from the 3D texture using integer coordinates
fn sampleTopLevelGridRaw(topPos: vec3<i32>) -> u32 {
    let sample = textureLoad(topLevelGridTex, topPos, 0);
    return sample.r;
}

fn topCacheIndex(offset: vec3<i32>) -> i32 {
    return offset.x + offset.y * TOP_CACHE_DIM + offset.z * TOP_CACHE_DIM * TOP_CACHE_DIM;
}

fn sampleTopLevelGridCached(topPos: vec3<i32>, cacheOrigin: vec3<i32>) -> u32 {
    let offset = topPos - cacheOrigin;
    if (offset.x >= 0 && offset.x < TOP_CACHE_DIM &&
        offset.y >= 0 && offset.y < TOP_CACHE_DIM &&
        offset.z >= 0 && offset.z < TOP_CACHE_DIM) {
        return topLevelCache[topCacheIndex(offset)];
    }

    return sampleTopLevelGridRaw(topPos);
}

fn preloadTopLevelCache(cacheOrigin: vec3<i32>, linearLocalId: u32) {
    if (linearLocalId >= u32(TOP_CACHE_COUNT)) {
        return;
    }

    let idx = i32(linearLocalId);
    let ox = idx % TOP_CACHE_DIM;
    let oy = (idx / TOP_CACHE_DIM) % TOP_CACHE_DIM;
    let oz = idx / (TOP_CACHE_DIM * TOP_CACHE_DIM);
    let topPos = cacheOrigin + vec3<i32>(ox, oy, oz);
    topLevelCache[idx] = sampleTopLevelGridRaw(topPos);
}

// Get top-level grid entry from world coordinates  
fn getTopLevelEntry(pos: vec3<i32>) -> u32 {
    let topX = pos.x / BRICK_SIZE;
    let topY = pos.y / BRICK_SIZE;
    let topZ = pos.z / BRICK_SIZE;
    
    // Check bounds
    if (topX < 0 || topX >= TOP_LEVEL_SIZE ||
        topY < 0 || topY >= TOP_LEVEL_SIZE ||
        topZ < 0 || topZ >= TOP_LEVEL_SIZE) {
        return 0u;
    }
    
    return sampleTopLevelGridRaw(vec3<i32>(topX, topY, topZ));
}

// Check if a top-level grid cell contains any solid voxels
fn isTopLevelOccupied(topPos: vec3<i32>) -> bool {
    // Check bounds
    if (topPos.x < 0 || topPos.x >= TOP_LEVEL_SIZE ||
        topPos.y < 0 || topPos.y >= TOP_LEVEL_SIZE ||
        topPos.z < 0 || topPos.z >= TOP_LEVEL_SIZE) {
        return false;
    }

    let topEntry = sampleTopLevelGridRaw(topPos);
    
    // If it's a solid color, check if it's non-zero
    if ((topEntry & IS_BRICK_FLAG) == 0u) {
        return topEntry != 0u;
    }
    
    // If it's a brick, we need to check if it has any solid voxels
    // For now, assume if it's a brick, it has some content
    return true;
}

// Get top-level coordinates from voxel coordinates
fn getTopLevelCoords(voxelPos: vec3<i32>) -> vec3<i32> {
    let fx = i32(floor(f32(voxelPos.x) / f32(BRICK_SIZE)));
    let fy = i32(floor(f32(voxelPos.y) / f32(BRICK_SIZE)));
    let fz = i32(floor(f32(voxelPos.z) / f32(BRICK_SIZE)));
    return vec3<i32>(
        fx,
        fy,
        fz
    );
}

fn wrapCoord(value: i32, dim: i32) -> i32 {
    var wrapped = value % dim;
    if (wrapped < 0) {
        wrapped = wrapped + dim;
    }
    return wrapped;
}

fn getLocalTopLevelCoords(globalTopPos: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        wrapCoord(globalTopPos.x, TOP_LEVEL_SIZE),
        wrapCoord(globalTopPos.y, TOP_LEVEL_SIZE),
        wrapCoord(globalTopPos.z, TOP_LEVEL_SIZE)
    );
}
// Get sector coordinates from world coordinates
fn getSectorCoords(voxelPos: vec3<i32>) -> vec3<i32> {
    let fx = i32(floor(f32(voxelPos.x) / f32(SECTOR_SIZE)));
    let fy = i32(floor(f32(voxelPos.y) / f32(SECTOR_SIZE)));
    let fz = i32(floor(f32(voxelPos.z) / f32(SECTOR_SIZE)));
    return vec3<i32>(
        fx,
        fy,
        fz
    );
}

// Check if a sector contains any solid voxels
fn isSectorOccupied(sectorPos: vec3<i32>) -> bool {
    let localX = wrapCoord(sectorPos.x, SECTORS_PER_DIM);
    let localY = wrapCoord(sectorPos.y, SECTORS_PER_DIM);
    let localZ = wrapCoord(sectorPos.z, SECTORS_PER_DIM);
    let sectorIndex = u32(localX + localY * SECTORS_PER_DIM + localZ * SECTORS_PER_DIM * SECTORS_PER_DIM);
    let sectorMeta = sectorCoords[sectorIndex];
    if (sectorMeta.w == 0) {
        return false;
    }
    if (sectorMeta.x != sectorPos.x || sectorMeta.y != sectorPos.y || sectorMeta.z != sectorPos.z) {
        return false;
    }
    return sectorMap[sectorIndex] != 0u;
}
struct DDAHit {
t: f32,
voxel : vec3<i32>,
normal : vec3<f32>,
steps : i32,
};

fn morton3D_3bit(xIn: i32, yIn: i32, zIn: i32) -> u32 {
    let x = u32(xIn) & 0x7u;
    let y = u32(yIn) & 0x7u;
    let z = u32(zIn) & 0x7u;

    return ((x & 1u) << 0u) | ((y & 1u) << 1u) | ((z & 1u) << 2u) |
           ((x & 2u) << 2u) | ((y & 2u) << 3u) | ((z & 2u) << 4u) |
           ((x & 4u) << 4u) | ((y & 4u) << 5u) | ((z & 4u) << 6u);
}

fn brickHasAnyVoxel(brickIndex: u32) -> bool {
    if (brickIndex >= MAX_BRICK_POOL_SIZE) {
        return false;
    }

    let brickWordBase = brickIndex * 256u;
    for (var i = 0u; i < 256u; i = i + 1u) {
        if (brickPool[brickWordBase + i] != 0u) {
            return true;
        }
    }
    return false;
}

struct NodeExit {
    valid: bool,
    tExit: f32,
    axis: i32,
};

fn computeNodeExit(
    rayOrigin: vec3<f32>,
    rayDir: vec3<f32>,
    nodeMin: vec3<f32>,
    nodeMax: vec3<f32>,
    tCurrent: f32,
    tMax: f32
) -> NodeExit {
    var nodeExit = vec3<f32>(1e10);
    for (var j = 0u; j < 3u; j = j + 1u) {
        if (abs(rayDir[j]) > 1e-8) {
            let boundary = select(nodeMin[j], nodeMax[j], rayDir[j] > 0.0);
            nodeExit[j] = (boundary - rayOrigin[j]) / rayDir[j];
        }
    }

    let exitT = min(nodeExit.x, min(nodeExit.y, nodeExit.z));
    if (!(exitT > tCurrent && exitT <= tMax)) {
        return NodeExit(false, 0.0, 0);
    }

    var axis = 0;
    if (nodeExit.y <= nodeExit.x && nodeExit.y <= nodeExit.z) {
        axis = 1;
    } else if (nodeExit.z <= nodeExit.x && nodeExit.z <= nodeExit.y) {
        axis = 2;
    }

    return NodeExit(true, exitT, axis);
}

// Hierarchical DDA that uses top-level grid to skip empty regions
// Enhanced DDA with camera precision fix
fn hierarchicalVoxelDDA(
    rayOrigin: vec3<f32>,
    rayDir : vec3<f32>,
    gridMin : vec3<i32>,
    gridMax : vec3<i32>,
    cameraPos: vec3<f32>,
    cameraPosInt: vec3<i32>,
    cameraPosOffset: vec3<f32>,
    topCacheOrigin: vec3<i32>
) -> DDAHit {
    let gridMinRelative = vec3<f32>(gridMin - cameraPosInt) - cameraPosOffset;
    let gridMaxRelative = vec3<f32>(gridMax - cameraPosInt) - cameraPosOffset;
    let gridMinF = gridMinRelative;
    let gridMaxF = gridMaxRelative + vec3<f32>(1.0, 1.0, 1.0);

    // === 1. Intersect the grid bounding box ===
    var tMin = -1e10;
    var tMax = 1e10;

    for (var i = 0u; i < 3u; i = i + 1u) {
        let epsilon = 1e-8;
        let safeRayDir = select(rayDir[i], epsilon, abs(rayDir[i]) < epsilon);
        let invD = 1.0 / safeRayDir;

        var t0 = (gridMinF[i] - rayOrigin[i]) * invD;
        var t1 = (gridMaxF[i] - rayOrigin[i]) * invD;
        if (invD < 0.0) {
            let tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tMin = max(tMin, t0);
        tMax = min(tMax, t1);
    }

    if (tMax < tMin) {
        return DDAHit(-1.0, vec3<i32>(0), vec3<f32>(0), -1);
    }

    // === 2. Start at entry point ===
    var t = max(tMin, 0.0);
    if (t == tMin) {
        t += 1e-4; // Larger epsilon for better precision
    }
    var p = rayOrigin + t * rayDir;
    
    // PRECISION FIX: Calculate position relative to camera's integer voxel position
    let pWithOffset = p + cameraPosOffset;

    // Use double precision for voxel calculation to reduce jitter
    var voxel = vec3<i32>(
        clamp(i32(floor(pWithOffset.x + 0.5 * sign(rayDir.x) * 1e-6)) + cameraPosInt.x, gridMin.x, gridMax.x),
        clamp(i32(floor(pWithOffset.y + 0.5 * sign(rayDir.y) * 1e-6)) + cameraPosInt.y, gridMin.y, gridMax.y),
        clamp(i32(floor(pWithOffset.z + 0.5 * sign(rayDir.z) * 1e-6)) + cameraPosInt.z, gridMin.z, gridMax.z)
    );

    let step = vec3<i32>(
        select(-1, 1, rayDir.x >= 0.0),
        select(-1, 1, rayDir.y >= 0.0),
        select(-1, 1, rayDir.z >= 0.0)
    );

    let epsilon = 1e-6; // Reduced epsilon for better precision
    let safeRayDir = vec3<f32>(
        select(rayDir.x, epsilon, abs(rayDir.x) < epsilon),
        select(rayDir.y, epsilon, abs(rayDir.y) < epsilon),
        select(rayDir.z, epsilon, abs(rayDir.z) < epsilon)
    );
    let invDir = 1.0 / safeRayDir;

    // Pre-calculate DDA increments for efficiency with better precision
    var next = vec3<f32>(0.0);
    var delta = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i = i + 1u) {
        let voxelBoundary = f32(voxel[i]) + select(0.0, 1.0, step[i] > 0);
        let boundaryOffsetFromCameraInt = voxelBoundary - f32(cameraPosInt[i]);
        let boundaryRelativeToRayOrigin = boundaryOffsetFromCameraInt - cameraPosOffset[i];
        next[i] = (boundaryRelativeToRayOrigin - rayOrigin[i]) * invDir[i];
        delta[i] = abs(invDir[i]);
    }

    var normal = vec3<f32>(0.0);
    let maxSteps = 512;
    var stepCount = 0;
    var currentTopLevel = vec3<i32>(-1000); // Invalid to trigger first-enter evaluation
    var currentSector = vec3<i32>(-1000); // Invalid to trigger first-enter evaluation
    var currentTopEntry = 0u;

    for (var i = 0; i < maxSteps; i = i + 1) {
        stepCount = i;

        // Check if we're still in bounds
        if (all(voxel >= gridMin) && all(voxel <= gridMax)) {
            let newSector = getSectorCoords(voxel);
            if (any(newSector != currentSector)) {
                currentSector = newSector;
                if (!isSectorOccupied(currentSector)) {
                    let sectorMin = vec3<f32>(
                        f32(currentSector.x * SECTOR_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32(currentSector.y * SECTOR_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32(currentSector.z * SECTOR_SIZE - cameraPosInt.z) - cameraPosOffset.z);
                    let sectorMax = vec3<f32>(
                        f32((currentSector.x + 1) * SECTOR_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32((currentSector.y + 1) * SECTOR_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32((currentSector.z + 1) * SECTOR_SIZE - cameraPosInt.z) - cameraPosOffset.z);

                    let exitInfo = computeNodeExit(rayOrigin, rayDir, sectorMin, sectorMax, t, tMax);
                    if (exitInfo.valid) {
                        t = exitInfo.tExit;
                        p = rayOrigin + t * rayDir;
                        let pWithOffset2 = p + cameraPosOffset;
                        
                        voxel = vec3<i32>(
                            clamp(i32(floor(pWithOffset2.x + sign(rayDir.x) * 0.001)) + cameraPosInt.x, gridMin.x, gridMax.x),
                            clamp(i32(floor(pWithOffset2.y + sign(rayDir.y) * 0.001)) + cameraPosInt.y, gridMin.y, gridMax.y),
                            clamp(i32(floor(pWithOffset2.z + sign(rayDir.z) * 0.001)) + cameraPosInt.z, gridMin.z, gridMax.z));

                        // Snap axis exactly to the exited empty sector face.
                        if (exitInfo.axis == 0) {
                            voxel.x = currentSector.x * SECTOR_SIZE + select(0, SECTOR_SIZE, step.x > 0);
                            if (step.x < 0) { voxel.x -= 1; }
                            normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
                        }
                        else if (exitInfo.axis == 1) {
                            voxel.y = currentSector.y * SECTOR_SIZE + select(0, SECTOR_SIZE, step.y > 0);
                            if (step.y < 0) { voxel.y -= 1; }
                            normal = vec3<f32>(0.0, -f32(step.y), 0.0);
                        }
                        else {
                            voxel.z = currentSector.z * SECTOR_SIZE + select(0, SECTOR_SIZE, step.z > 0);
                            if (step.z < 0) { voxel.z -= 1; }
                            normal = vec3<f32>(0.0, 0.0, -f32(step.z));
                        }

                        // Recalculate DDA boundaries perfectly aligned
                        for (var j = 0u; j < 3u; j = j + 1u) {
                            let voxelBoundary2 = f32(voxel[j]) + select(0.0, 1.0, step[j] > 0);
                            let boundaryOffsetFromCameraInt2 = voxelBoundary2 - f32(cameraPosInt[j]);
                            let boundaryRelativeToRayOrigin2 = boundaryOffsetFromCameraInt2 - cameraPosOffset[j];
                            next[j] = (boundaryRelativeToRayOrigin2 - rayOrigin[j]) * invDir[j];
                        }
                        continue;
                    }
                }
            }

            let newTopLevel = getTopLevelCoords(voxel);

            // If we've moved to a new top-level cell, check if it's occupied
            if (any(newTopLevel != currentTopLevel)) {
                currentTopLevel = newTopLevel;
                let localTopLevel = getLocalTopLevelCoords(currentTopLevel);
                currentTopEntry = sampleTopLevelGridCached(localTopLevel, topCacheOrigin);
                var topIsEmpty = currentTopEntry == 0u;
                if (!topIsEmpty && (currentTopEntry & IS_BRICK_FLAG) != 0u) {
                    let topBrickIndex = currentTopEntry & BRICK_INDEX_MASK;
                    if (!brickHasAnyVoxel(topBrickIndex)) {
                        topIsEmpty = true;
                    }
                }

                // If top-level cell is not occupied, skip directly to top-level boundary
                if (topIsEmpty) {
                    // Calculate the top-level cell boundaries
                    let topMin = vec3<f32>(
                        f32(currentTopLevel.x * BRICK_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32(currentTopLevel.y * BRICK_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32(currentTopLevel.z * BRICK_SIZE - cameraPosInt.z) - cameraPosOffset.z);
                    let topMax = vec3<f32>(
                        f32((currentTopLevel.x + 1) * BRICK_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32((currentTopLevel.y + 1) * BRICK_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32((currentTopLevel.z + 1) * BRICK_SIZE - cameraPosInt.z) - cameraPosOffset.z);

                    let exitInfo = computeNodeExit(rayOrigin, rayDir, topMin, topMax, t, tMax);

                    if (exitInfo.valid) {
                        t = exitInfo.tExit;
                        p = rayOrigin + t * rayDir;
                        let pWithOffset3 = p + cameraPosOffset;

                        voxel = vec3<i32>(
                            clamp(i32(floor(pWithOffset3.x + sign(rayDir.x) * 0.001)) + cameraPosInt.x, gridMin.x, gridMax.x),
                            clamp(i32(floor(pWithOffset3.y + sign(rayDir.y) * 0.001)) + cameraPosInt.y, gridMin.y, gridMax.y),
                            clamp(i32(floor(pWithOffset3.z + sign(rayDir.z) * 0.001)) + cameraPosInt.z, gridMin.z, gridMax.z)
                        );

                        // Snap axis exactly to the exited empty top-level face.
                        if (exitInfo.axis == 0) {
                            voxel.x = currentTopLevel.x * BRICK_SIZE + select(0, BRICK_SIZE, step.x > 0);
                            if (step.x < 0) { voxel.x -= 1; }
                            normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
                        }
                        else if (exitInfo.axis == 1) {
                            voxel.y = currentTopLevel.y * BRICK_SIZE + select(0, BRICK_SIZE, step.y > 0);
                            if (step.y < 0) { voxel.y -= 1; }
                            normal = vec3<f32>(0.0, -f32(step.y), 0.0);
                        }
                        else {
                            voxel.z = currentTopLevel.z * BRICK_SIZE + select(0, BRICK_SIZE, step.z > 0);
                            if (step.z < 0) { voxel.z -= 1; }
                            normal = vec3<f32>(0.0, 0.0, -f32(step.z));
                        }
                        
                        // Recalculate DDA bounds perfectly
                        for (var j = 0u; j < 3u; j = j + 1u) {
                            let voxelBoundary3 = f32(voxel[j]) + select(0.0, 1.0, step[j] > 0);
                            let boundaryOffsetFromCameraInt3 = voxelBoundary3 - f32(cameraPosInt[j]);
                            let boundaryRelativeToRayOrigin3 = boundaryOffsetFromCameraInt3 - cameraPosOffset[j];
                            next[j] = (boundaryRelativeToRayOrigin3 - rayOrigin[j]) * invDir[j];
                        }

                        continue; // Skip to next iteration
                    }
                }
            }

            // Stackless fixed 3-level traversal: sector -> top-level -> brick voxel.
            // We already know current sector is occupied and currentTopEntry is cached.
            if ((currentTopEntry & IS_BRICK_FLAG) == 0u) {
                if (currentTopEntry != 0u) {
                    return DDAHit(t, voxel, normal, i);
                }
            } else {
                let brickIndex = currentTopEntry & BRICK_INDEX_MASK;
                if (brickIndex < MAX_BRICK_POOL_SIZE) {
                    let localX = wrapCoord(voxel.x, BRICK_SIZE);
                    let localY = wrapCoord(voxel.y, BRICK_SIZE);
                    let localZ = wrapCoord(voxel.z, BRICK_SIZE);

                    let voxelOffset = morton3D_3bit(localX, localY, localZ);
                    let voxelIndex = brickIndex * 512u + voxelOffset;
                    let wordIndex = voxelIndex / 2u;
                    let halfIndex = voxelIndex % 2u;
                    let word = brickPool[wordIndex];
                    let voxelData = (word >> (halfIndex * 16u)) & 0xFFFFu;

                    if (voxelData != 0u) {
                        return DDAHit(t, voxel, normal, i);
                    }
                }
            }
        }

        // Standard DDA step - use consistent epsilon to prevent jitter
        let eps = 1e-5; // Larger epsilon for better stability
        if (next.x <= next.y + eps && next.x <= next.z + eps) {
            voxel.x += step.x;
            t = next.x;
            next.x += delta.x;
            normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
        }
        else if (next.y <= next.z + eps) {
            voxel.y += step.y;
            t = next.y;
            next.y += delta.y;
            normal = vec3<f32>(0.0, -f32(step.y), 0.0);
        }
        else {
            voxel.z += step.z;
            t = next.z;
            next.z += delta.z;
            normal = vec3<f32>(0.0, 0.0, -f32(step.z));
        }

        if (t > tMax) {
            break;
        }
    }

    return DDAHit(-1.0, vec3<i32>(0), vec3<f32>(0), stepCount);
}

// Function to check if a voxel is solid based on the hierarchical storage
fn isVoxelSolid(voxelPos: vec3<i32>, topCacheOrigin: vec3<i32>) -> bool {
    // Check bounds
    if (voxelPos.y < 0) {
        return false;
    }

    let sectorPos = getSectorCoords(voxelPos);
    if (!isSectorOccupied(sectorPos)) {
        return false;
    }

    // Get top-level grid entry using the new texture-based method
    let topPos = getLocalTopLevelCoords(getTopLevelCoords(voxelPos));
    let topEntry = sampleTopLevelGridCached(topPos, topCacheOrigin);

    if ((topEntry & IS_BRICK_FLAG) == 0u) {
        // It's a solid color
        return topEntry != 0u;
    } else {
        // It's a brick - check individual voxel
        let brickIndex = topEntry & BRICK_INDEX_MASK;
        
        // Bounds check for brick index
        if (brickIndex >= MAX_BRICK_POOL_SIZE) {
            return false;
        }
        
        let localX = wrapCoord(voxelPos.x, BRICK_SIZE);
        let localY = wrapCoord(voxelPos.y, BRICK_SIZE);
        let localZ = wrapCoord(voxelPos.z, BRICK_SIZE);
        
        // Each brick has 512 uint16 voxels, packed 2 per u32 (= 256 u32 per brick)
        let voxelOffset = morton3D_3bit(localX, localY, localZ);
        let voxelIndex = brickIndex * 512u + voxelOffset;
        let wordIndex = voxelIndex / 2u;   // 2 uint16s per u32
        let halfIndex = voxelIndex % 2u;   // which uint16 within the u32 (0=low, 1=high)
        
        let word = brickPool[wordIndex];
        let voxelData = (word >> (halfIndex * 16u)) & 0xFFFFu;
        
        return voxelData != 0u;
    }
}

// Get voxel color from hierarchical storage
fn getVoxelColor(voxelPos: vec3<i32>, topCacheOrigin: vec3<i32>) -> vec4<f32> {
    // Check bounds
    if (voxelPos.y < 0) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta for out of bounds
    }

    let sectorPos = getSectorCoords(voxelPos);
    if (!isSectorOccupied(sectorPos)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Get top-level grid entry using the new texture-based method
    let topPos = getLocalTopLevelCoords(getTopLevelCoords(voxelPos));
    let topEntry = sampleTopLevelGridCached(topPos, topCacheOrigin);

    if ((topEntry & IS_BRICK_FLAG) == 0u) {
        // It's a solid color
        if (topEntry == 0u) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Transparent for empty
        }
        
        // Unpack RGB color from bits 0-23
        let r = f32((topEntry >> 16u) & 0xFFu) / 255.0;
        let g = f32((topEntry >> 8u) & 0xFFu) / 255.0;
        let b = f32(topEntry & 0xFFu) / 255.0;
        return vec4<f32>(r, g, b, 1.0);
    } else {
        // It's a brick - check individual voxel
        let brickIndex = topEntry & BRICK_INDEX_MASK;
        
        // Bounds check for brick index
        if (brickIndex >= MAX_BRICK_POOL_SIZE) {
            return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta for invalid brick
        }
        
        let localX = wrapCoord(voxelPos.x, BRICK_SIZE);
        let localY = wrapCoord(voxelPos.y, BRICK_SIZE);
        let localZ = wrapCoord(voxelPos.z, BRICK_SIZE);
        
        // Each brick has 512 uint16 voxels, packed 2 per u32 (= 256 u32 per brick)
        let voxelOffset = morton3D_3bit(localX, localY, localZ);
        let voxelIndex = brickIndex * 512u + voxelOffset;
        let wordIndex = voxelIndex / 2u;   // 2 uint16s per u32
        let halfIndex = voxelIndex % 2u;   // which uint16 within the u32 (0=low, 1=high)
        
        let word = brickPool[wordIndex];
        let voxelData = (word >> (halfIndex * 16u)) & 0xFFFFu;
        
        if (voxelData == 0u) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Transparent for empty
        }
        
        // Lower 8 bits = palette index (upper 8 bits reserved for future material data)
        let paletteIndex = voxelData & 0xFFu;
        
        // Look up color from palette
        let packedColor = colorPalette[paletteIndex];
        let r = f32((packedColor >> 16u) & 0xFFu) / 255.0;
        let g = f32((packedColor >> 8u) & 0xFFu) / 255.0;
        let b = f32(packedColor & 0xFFu) / 255.0;
        let a = f32((packedColor >> 24u) & 0xFFu) / 255.0;
        
        return vec4<f32>(r, g, b, a);
    }
}

// Multi-level heatmap (Viridis/Inferno blend)
fn viridis_inferno(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    // Simple but vibrant inferno-like color ramp
    let c1 = vec3<f32>(0.0, 0.0, 0.04);       // Black/Purple
    let c2 = vec3<f32>(0.5, 0.0, 0.5);        // Purple/Magenta
    let c3 = vec3<f32>(1.0, 0.5, 0.0);        // Orange
    let c4 = vec3<f32>(1.0, 1.0, 0.5);        // Yellow
    
    if (x < 0.33) {
        return mix(c1, c2, x / 0.33);
    } else if (x < 0.66) {
        return mix(c2, c3, (x - 0.33) / 0.33);
    } else {
        return mix(c3, c4, (x - 0.66) / 0.34);
    }
}

// Visualize ray steps with clear contrast - Heatmap version
fn visualizeRaySteps(steps: i32, maxSteps : i32) -> vec4<f32> {
    if (steps <= 0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let iters = f32(steps) / f32(maxSteps);
    let color = viridis_inferno(iters);
    return vec4<f32>(color, 1.0);
}

// Calculate shadow by casting a ray towards the light
fn calculateShadow(hitPos: vec3<f32>, normal : vec3<f32>, lightDir : vec3<f32>, gridMin : vec3<i32>, gridMax : vec3<i32>, cameraPos: vec3<f32>, cameraPosInt: vec3<i32>, cameraPosOffset: vec3<f32>, topCacheOrigin: vec3<i32>) -> f32 {
    let hitPosInt = vec3<i32>(hitPos);
    
    // Conservative boundary margin: only cast shadow rays if hit is well inside loaded grid
    let gridSize = gridMax - gridMin;
    let hitRelative = hitPosInt - gridMin;
    let margin = 8; // 8-voxel margin from boundaries
    
    // Check if far enough from all boundaries
    if (any(hitRelative < vec3<i32>(margin)) || any(hitRelative > gridSize - vec3<i32>(margin))) {
        return 1.0; // Near boundary: skip shadow ray, assume fully lit
    }
    
    // Offset the ray start position slightly along the normal to avoid self-intersection
    let shadowRayOriginWorld = hitPos + normal * 0.5;
    let shadowRayOriginRelative = shadowRayOriginWorld - cameraPos;

    // Cast shadow ray towards light using the main DDA function
    let shadowHit = hierarchicalVoxelDDA(shadowRayOriginRelative, lightDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset, topCacheOrigin);

    // Hierarchical DDA only returns positive t for solid hits.
    if (shadowHit.t > 0.0) {
        return 0.2; // Shadow intensity
    }

    return 1.0; // No shadow
}

// Calculate reflection color by casting a reflection ray
fn calculateReflection(hitPos: vec3<f32>, rayDir : vec3<f32>, normal : vec3<f32>, gridMin : vec3<i32>, gridMax : vec3<i32>, cameraPos: vec3<f32>, cameraPosInt: vec3<i32>, cameraPosOffset: vec3<f32>, topCacheOrigin: vec3<i32>) -> vec4<f32> {
    // Calculate reflection direction
    let reflectDir = reflect(rayDir, normal);

    // Offset the ray start position slightly along the normal to avoid self-intersection
    let reflectRayOriginWorld = hitPos + normal * 0.01;
    let reflectRayOriginRelative = reflectRayOriginWorld - cameraPos;

    // Cast reflection ray
    let reflectHit = hierarchicalVoxelDDA(reflectRayOriginRelative, reflectDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset, topCacheOrigin);

    if (reflectHit.t > 0.0) {
        let reflectedVoxelColor = getVoxelColor(reflectHit.voxel, topCacheOrigin);
        if (reflectedVoxelColor.a > 0.0) {
            // Apply simple lighting to reflected surface
            let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0));
            let reflectedDiffuse = max(0.3, dot(reflectHit.normal, lightDir));
            return vec4<f32>(reflectedVoxelColor.rgb * reflectedDiffuse, reflectedVoxelColor.a);
        }
    }

    // If no reflection hit, return sky color
    return vec4<f32>(0.5, 0.7, 1.0, 0.1); // Low alpha for subtle reflection
}

// Cheap voxel face AO: sample neighbors around the hit face.
fn calculateVoxelAO(voxelPos: vec3<i32>, normal: vec3<f32>, topCacheOrigin: vec3<i32>) -> f32 {
    let n = vec3<i32>(i32(round(normal.x)), i32(round(normal.y)), i32(round(normal.z)));

    var tangentA = vec3<i32>(1, 0, 0);
    var tangentB = vec3<i32>(0, 1, 0);
    if (abs(n.x) == 1) {
        tangentA = vec3<i32>(0, 1, 0);
        tangentB = vec3<i32>(0, 0, 1);
    } else if (abs(n.y) == 1) {
        tangentA = vec3<i32>(1, 0, 0);
        tangentB = vec3<i32>(0, 0, 1);
    } else {
        tangentA = vec3<i32>(1, 0, 0);
        tangentB = vec3<i32>(0, 1, 0);
    }

    // Sample one voxel outside the hit face to bias AO toward exposed areas.
    let faceCell = voxelPos + n;

    let sideOcc =
        f32(isVoxelSolid(faceCell + tangentA, topCacheOrigin)) +
        f32(isVoxelSolid(faceCell - tangentA, topCacheOrigin)) +
        f32(isVoxelSolid(faceCell + tangentB, topCacheOrigin)) +
        f32(isVoxelSolid(faceCell - tangentB, topCacheOrigin));

    let cornerOcc =
        f32(isVoxelSolid(faceCell + tangentA + tangentB, topCacheOrigin)) +
        f32(isVoxelSolid(faceCell + tangentA - tangentB, topCacheOrigin)) +
        f32(isVoxelSolid(faceCell - tangentA + tangentB, topCacheOrigin)) +
        f32(isVoxelSolid(faceCell - tangentA - tangentB, topCacheOrigin));

    // Side neighbors contribute more than corners.
    let occlusion = sideOcc * 0.12 + cornerOcc * 0.07;
    return clamp(1.0 - occlusion, 0.35, 1.0);
}

//@compute @workgroup_size(8, 8, 1)
//fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
//    let dims = textureDimensions(outTex);
//    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
//
//    // TEMPORARY: Render owl texture instead of voxels
//    let resolution = res.xy;
//    let pixelCenter = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
//    let uv = pixelCenter / resolution;
//    
//    // Get texture dimensions and convert UV to pixel coordinates
//    let texDims = textureDimensions(owlTexture);
//    let texCoords = vec2<i32>(
//        i32(uv.x * f32(texDims.x)),
//        i32(uv.y * f32(texDims.y))
//    );
//    
//    // Clamp coordinates to texture bounds
//    let clampedCoords = vec2<i32>(
//        clamp(texCoords.x, 0, i32(texDims.x) - 1),
//        clamp(texCoords.y, 0, i32(texDims.y) - 1)
//    );
//    
//    // Load the owl texture pixel (use textureLoad instead of textureSample for compute)
//    let owlColor = textureLoad(owlTexture, clampedCoords, 0);
//    
//    // Store the texture color directly
//    textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), owlColor);
//}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
    let dims = textureDimensions(outTex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let resolution = res.xy;
    // Use pixel center for ray generation with improved precision
    let pixelCenter = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let uv = (pixelCenter / resolution) * 2.0 - vec2<f32>(1.0, 1.0);
    let ndc = vec4<f32>(uv.x, -uv.y, 1.0, 1.0);

    // PRECISION FIX: Use camera-relative coordinates
    let cameraPos = position;
    let cameraPosInt = vec3<i32>(i32(floor(cameraPos.x)), i32(floor(cameraPos.y)), i32(floor(cameraPos.z)));
    let cameraPosOffset = cameraPos - vec3<f32>(cameraPosInt);
    let cameraTop = getTopLevelCoords(cameraPosInt);
    let cameraTopLocal = getLocalTopLevelCoords(cameraTop);
    let topCacheOrigin = vec3<i32>(
        clamp(cameraTopLocal.x - TOP_CACHE_DIM / 2, 0, TOP_LEVEL_SIZE - TOP_CACHE_DIM),
        clamp(cameraTopLocal.y - TOP_CACHE_DIM / 2, 0, TOP_LEVEL_SIZE - TOP_CACHE_DIM),
        clamp(cameraTopLocal.z - TOP_CACHE_DIM / 2, 0, TOP_LEVEL_SIZE - TOP_CACHE_DIM));

    let localLinearId = lid.x + lid.y * 8u + lid.z * 64u;
    preloadTopLevelCache(topCacheOrigin, localLinearId);
    workgroupBarrier();

    let rayOrigin = vec3<f32>(0.0, 0.0, 0.0);
    let viewPos = invViewProj * ndc;
    let rayDir = normalize(viewPos.xyz / viewPos.w);

    let boundsMinSector = sectorBounds[0];
    let boundsMaxSector = sectorBounds[1];
    if (boundsMinSector.w == 0) {
        textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.5, 0.7, 1.0, 1.0));
        return;
    }

    // Expand by one sector on each side for stable edge behavior.
    let sectorPad = 1;
    let gridMin = vec3<i32>(
        (boundsMinSector.x - sectorPad) * SECTOR_SIZE,
        (boundsMinSector.y - sectorPad) * SECTOR_SIZE,
        (boundsMinSector.z - sectorPad) * SECTOR_SIZE);
    let gridMax = vec3<i32>(
        (boundsMaxSector.x + sectorPad + 1) * SECTOR_SIZE - 1,
        (boundsMaxSector.y + sectorPad + 1) * SECTOR_SIZE - 1,
        (boundsMaxSector.z + sectorPad + 1) * SECTOR_SIZE - 1);

    let hit = hierarchicalVoxelDDA(rayOrigin, rayDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset, topCacheOrigin);

    var color : vec4<f32>;
    let fogEnabled = res.w > 0.0;
    let fogEnd = max(abs(res.w), 1.0);
    let fogStart = fogEnd * 0.35;
    let skyColor = vec3<f32>(0.5, 0.7, 1.0);

    // If we hit a voxel, return its color with shadows and reflections
    if (hit.t > 0.0) {
        let voxelColor = getVoxelColor(hit.voxel, topCacheOrigin);
        if (voxelColor.a > 0.0) {
            // Calculate hit position in world space
            let hitPos = rayOrigin + hit.t * rayDir;

            // Apply lighting based on normal
            let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0));
            let nDotL = dot(hit.normal, lightDir);
            
            var shadowFactor: f32 = 1.0;
            if (ENABLE_VOXEL_SHADOWS) {
                if (nDotL <= 0.0) {
                    // Face is pointing away from the light, it's automatically in shadow
                    shadowFactor = 0.2;
                } else {
                    // Regular per-pixel shadow tracing for faces pointing towards the light
                    shadowFactor = calculateShadow(hitPos + cameraPos, hit.normal, lightDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset, topCacheOrigin);
                }
            }

            // Toggle AO for debugging streaming/lighting artifacts.
            let ao = select(1.0, calculateVoxelAO(hit.voxel, hit.normal, topCacheOrigin), ENABLE_VOXEL_AO);
            let ambient = 0.26 * ao;
            let direct = max(0.0, nDotL) * shadowFactor;
            let light = ambient + direct * 0.82;
            let shadedColor = voxelColor.rgb * light;
            var finalColor = shadedColor;
            if (fogEnabled) {
                let fogFactor = smoothstep(fogStart, fogEnd, hit.t);
                finalColor = mix(finalColor, skyColor, fogFactor);
            }

            color = vec4<f32>(finalColor, 1.0);
        }
        else {
            // Show step visualization if ray entered the voxel grid but didn't hit anything
            if (hit.steps >= 0 && res.z > 0.5) {
                let maxSteps = 512;
                color = visualizeRaySteps(hit.steps, maxSteps);
            }
            else {
                // Ray missed the voxel grid - show blue sky
                color = vec4<f32>(0.5, 0.7, 1.0, 1.0);
            }
        }
    }
    else {
        // Show step visualization if ray entered the voxel grid but didn't hit anything
        if (hit.steps >= 0 && res.z > 0.5) {
            let maxSteps = 512;
            color = visualizeRaySteps(hit.steps, maxSteps);
        }
        else {
            // Ray missed the voxel grid - show blue sky
            color = vec4<f32>(0.5, 0.7, 1.0, 1.0);
        }
    }

    textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}