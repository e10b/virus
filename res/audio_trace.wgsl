@group(0) @binding(0) var<uniform> params: mat4x4<f32>;
@group(0) @binding(1) var<storage, read_write> outAudio: array<vec4<f32>, 6>;
@group(0) @binding(4) var topLevelGridTex: texture_3d<u32>;
@group(0) @binding(6) var<storage, read_write> brickPool: array<u32>;
@group(0) @binding(10) var<storage, read_write> sectorMap: array<u32>;
@group(0) @binding(11) var<storage, read_write> sectorCoords: array<vec4<i32>>;

const TOP_LEVEL_SIZE = 64;
const BRICK_SIZE = 8;
const BRICKS_PER_SECTOR = 4;
const SECTOR_SIZE = BRICK_SIZE * BRICKS_PER_SECTOR;
const SECTORS_PER_DIM = TOP_LEVEL_SIZE / BRICKS_PER_SECTOR;
const MAX_BRICK_POOL_SIZE = 524288u;
const BRICK_INDEX_MASK = 0x7FFFFFFFu;
const IS_BRICK_FLAG = 0x80000000u;

const REFLECTION_DIR_COUNT = 8;
const MAX_REFLECTIONS = 4;
const REFLECTION_DIR_COUNT_U = 8u;
const MAX_REFLECTIONS_U = 4u;
const MAX_BOUNCES = 3;
const MAX_BOUNCES_U = 3u;

struct HitInfo {
    hit: bool,
    distance: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    material: u32,
};

fn wrapCoord(value: i32, dim: i32) -> i32 {
    var wrapped = value % dim;
    if (wrapped < 0) {
        wrapped = wrapped + dim;
    }
    return wrapped;
}

fn floorDiv(value: i32, divisor: i32) -> i32 {
    return i32(floor(f32(value) / f32(divisor)));
}

fn getTopLevelCoords(voxelPos: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        floorDiv(voxelPos.x, BRICK_SIZE),
        floorDiv(voxelPos.y, BRICK_SIZE),
        floorDiv(voxelPos.z, BRICK_SIZE)
    );
}

fn getLocalTopLevelCoords(globalTopPos: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        wrapCoord(globalTopPos.x, TOP_LEVEL_SIZE),
        wrapCoord(globalTopPos.y, TOP_LEVEL_SIZE),
        wrapCoord(globalTopPos.z, TOP_LEVEL_SIZE)
    );
}

fn getSectorCoords(voxelPos: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        floorDiv(voxelPos.x, SECTOR_SIZE),
        floorDiv(voxelPos.y, SECTOR_SIZE),
        floorDiv(voxelPos.z, SECTOR_SIZE)
    );
}

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

fn morton3D_3bit(xIn: i32, yIn: i32, zIn: i32) -> u32 {
    let x = u32(xIn) & 0x7u;
    let y = u32(yIn) & 0x7u;
    let z = u32(zIn) & 0x7u;

    return ((x & 1u) << 0u) | ((y & 1u) << 1u) | ((z & 1u) << 2u) |
           ((x & 2u) << 2u) | ((y & 2u) << 3u) | ((z & 2u) << 4u) |
           ((x & 4u) << 4u) | ((y & 4u) << 5u) | ((z & 4u) << 6u);
}

fn voxelMaterial(voxelPos: vec3<i32>) -> u32 {
    if (voxelPos.y < 0) {
        return 0u;
    }

    let sectorPos = getSectorCoords(voxelPos);
    if (!isSectorOccupied(sectorPos)) {
        return 0u;
    }

    let topPos = getLocalTopLevelCoords(getTopLevelCoords(voxelPos));
    let topEntry = textureLoad(topLevelGridTex, topPos, 0).r;

    if ((topEntry & IS_BRICK_FLAG) == 0u) {
        if (topEntry == 0u) {
            return 0u;
        }
        // Legacy solid-color entries collapse to default stone-like material.
        return 4u;
    }

    let brickIndex = topEntry & BRICK_INDEX_MASK;
    if (brickIndex >= MAX_BRICK_POOL_SIZE) {
        return 0u;
    }

    let localX = wrapCoord(voxelPos.x, BRICK_SIZE);
    let localY = wrapCoord(voxelPos.y, BRICK_SIZE);
    let localZ = wrapCoord(voxelPos.z, BRICK_SIZE);

    let voxelOffset = morton3D_3bit(localX, localY, localZ);
    let voxelIndex = brickIndex * 512u + voxelOffset;
    let wordIndex = voxelIndex / 2u;
    let halfIndex = voxelIndex % 2u;

    let word = brickPool[wordIndex];
    let voxelData = (word >> (halfIndex * 16u)) & 0xFFFFu;
    if (voxelData == 0u) {
        return 0u;
    }

    return voxelData & 0xFFu;
}

fn materialAbsorption(material: u32) -> f32 {
    if (material == 1u) { return 0.52; }
    if (material == 2u) { return 0.34; }
    if (material == 3u) { return 0.48; }
    if (material == 4u) { return 0.18; }
    if (material == 5u) { return 0.50; }
    if (material == 6u) { return 0.24; }
    if (material == 7u) { return 0.38; }
    if (material == 9u) { return 0.32; }
    if (material == 10u) { return 0.22; }
    return 0.35;
}

fn distanceAttenuation(distance: f32) -> f32 {
    return 1.0 / (1.0 + 0.03 * distance + 0.0009 * distance * distance);
}

fn traceFirstHit(origin: vec3<f32>, direction: vec3<f32>, maxDistance: f32) -> HitInfo {
    var result = HitInfo(false, 0.0, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 0u);

    let dir = normalize(direction);
    let step = 0.75;
    var prevVoxel = vec3<i32>(floor(origin));

    var t = step;
    loop {
        if (t > maxDistance) {
            break;
        }
        let p = origin + dir * t;
        let voxel = vec3<i32>(floor(p));
        if (all(voxel == prevVoxel)) {
            t = t + step;
            continue;
        }

        let m = voxelMaterial(voxel);
        if (m != 0u) {
            let d = voxel - prevVoxel;
            var n = vec3<f32>(0.0, 1.0, 0.0);
            if (d.x != 0) {
                n = vec3<f32>(-f32(d.x), 0.0, 0.0);
            } else if (d.y != 0) {
                n = vec3<f32>(0.0, -f32(d.y), 0.0);
            } else if (d.z != 0) {
                n = vec3<f32>(0.0, 0.0, -f32(d.z));
            }

            result.hit = true;
            result.distance = t;
            result.position = p;
            result.normal = normalize(n);
            result.material = m;
            return result;
        }

        prevVoxel = voxel;
        t = t + step;
    }

    return result;
}

fn segmentOccluded(a: vec3<f32>, b: vec3<f32>) -> vec2<f32> {
    // x = occluded (0/1), y = blocking material
    let delta = b - a;
    let dist = length(delta);
    if (dist <= 0.001) {
        return vec2<f32>(0.0, 0.0);
    }

    let dir = delta / dist;
    let step = 0.75;
    var prevVoxel = vec3<i32>(floor(a));

    var t = step;
    loop {
        if (t >= dist) {
            break;
        }
        let p = a + dir * t;
        let voxel = vec3<i32>(floor(p));
        if (all(voxel == prevVoxel)) {
            t = t + step;
            continue;
        }

        let m = voxelMaterial(voxel);
        if (m != 0u) {
            return vec2<f32>(1.0, f32(m));
        }

        prevVoxel = voxel;
        t = t + step;
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    // outAudio[0] = vec4(directGain, directDistance, occludedFlag, reflectionCount)
    // outAudio[1..4] = vec4(delaySeconds, gain, material, alignment)
    // outAudio[5] reserved
    outAudio[0] = vec4<f32>(0.0);
    outAudio[1] = vec4<f32>(0.0);
    outAudio[2] = vec4<f32>(0.0);
    outAudio[3] = vec4<f32>(0.0);
    outAudio[4] = vec4<f32>(0.0);
    outAudio[5] = vec4<f32>(0.0);

    let sourcePos = params[0].xyz;
    let listenerPos = params[1].xyz;
    let maxDistance = params[1].w;
    let soundSpeed = params[2].x;

    let directVec = listenerPos - sourcePos;
    let directDistance = length(directVec);
    if (directDistance < 0.001 || directDistance > maxDistance) {
        return;
    }

    let occlusion = segmentOccluded(sourcePos, listenerPos);
    let directBase = distanceAttenuation(directDistance);
    var directGain = directBase;

    if (occlusion.x > 0.5) {
        let blocker = u32(occlusion.y + 0.5);
        let transmission = 1.0 - materialAbsorption(blocker);
        directGain = directBase * transmission * 0.22;
    }

    var reflectionDirs = array<vec3<f32>, REFLECTION_DIR_COUNT>(
        normalize(vec3<f32>(1.0, 0.2, 0.0)),
        normalize(vec3<f32>(-1.0, 0.2, 0.0)),
        normalize(vec3<f32>(0.0, 0.2, 1.0)),
        normalize(vec3<f32>(0.0, 0.2, -1.0)),
        normalize(vec3<f32>(1.0, 0.35, 1.0)),
        normalize(vec3<f32>(-1.0, 0.35, 1.0)),
        normalize(vec3<f32>(1.0, 0.35, -1.0)),
        normalize(vec3<f32>(-1.0, 0.35, -1.0))
    );

    var taps = array<vec4<f32>, MAX_REFLECTIONS>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0)
    );

    var i = 0u;
    loop {
        if (i >= REFLECTION_DIR_COUNT_U) {
            break;
        }

        var pathOrigin = sourcePos;
        var pathDir = reflectionDirs[i];
        var accumulatedDistance = 0.0;
        var pathEnergy = 1.0;

        var bounce = 0u;
        loop {
            if (bounce >= MAX_BOUNCES_U) {
                break;
            }

            let remainingDistance = maxDistance * 2.0 - accumulatedDistance;
            if (remainingDistance <= 0.0) {
                break;
            }

            let hit = traceFirstHit(pathOrigin, pathDir, remainingDistance);
            if (!hit.hit) {
                break;
            }

            accumulatedDistance = accumulatedDistance + hit.distance;
            let bouncePoint = hit.position + hit.normal * 0.15;

            let reflectivity = 1.0 - materialAbsorption(hit.material);
            pathEnergy = pathEnergy * reflectivity * 0.86;
            if (pathEnergy < 0.02) {
                break;
            }

            let toListener = listenerPos - bouncePoint;
            let segmentDistance = length(toListener);
            if (segmentDistance > 0.001) {
                let reflectedDir = reflect(pathDir, hit.normal);
                let alignment = dot(normalize(toListener), normalize(reflectedDir));
                if (alignment > 0.55) {
                    let blocked = segmentOccluded(bouncePoint, listenerPos);
                    if (blocked.x < 0.5) {
                        let totalDistance = accumulatedDistance + segmentDistance;
                        if (totalDistance > directDistance + 0.5 && totalDistance < maxDistance * 2.0) {
                            let gain = distanceAttenuation(totalDistance) * pathEnergy * (0.35 + 0.65 * alignment);
                            if (gain > 0.01) {
                                let delay = (totalDistance - directDistance) / max(soundSpeed, 1.0);
                                let tap = vec4<f32>(delay, gain, f32(hit.material), alignment);

                                var insertIdx = i32(MAX_REFLECTIONS);
                                var k = 0u;
                                loop {
                                    if (k >= MAX_REFLECTIONS_U) {
                                        break;
                                    }
                                    if (tap.y > taps[k].y) {
                                        insertIdx = i32(k);
                                        break;
                                    }
                                    k = k + 1u;
                                }

                                if (insertIdx < i32(MAX_REFLECTIONS)) {
                                    var shift = i32(MAX_REFLECTIONS) - 1;
                                    loop {
                                        if (shift <= insertIdx) {
                                            break;
                                        }
                                        taps[u32(shift)] = taps[u32(shift - 1)];
                                        shift = shift - 1;
                                    }
                                    taps[u32(insertIdx)] = tap;
                                }
                            }
                        }
                    }
                }
            }

            pathDir = reflect(pathDir, hit.normal);
            pathOrigin = bouncePoint + pathDir * 0.2;
            bounce = bounce + 1u;
        }

        i = i + 1u;
    }

    var count = 0.0;
    var r = 0u;
    loop {
        if (r >= MAX_REFLECTIONS_U) {
            break;
        }
        outAudio[r + 1] = taps[r];
        if (taps[r].y > 0.0) {
            count = count + 1.0;
        }
        r = r + 1u;
    }

    outAudio[0] = vec4<f32>(directGain, directDistance, occlusion.x, count);
}
