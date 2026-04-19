import sys

with open('res/compute.wgsl', 'r') as f:
    code = f.read()

# Replace the DDA signature
code = code.replace(
'''fn hierarchicalVoxelDDA(
    rayOrigin: vec3<f32>,
    rayDir : vec3<f32>,
    gridMin : vec3<i32>,
    gridMax : vec3<i32>
) -> DDAHit {
    let gridMinF = vec3<f32>(f32(gridMin.x), f32(gridMin.y), f32(gridMin.z));
    let gridMaxF = vec3<f32>(f32(gridMax.x + 1), f32(gridMax.y + 1), f32(gridMax.z + 1));''',
'''// Enhanced DDA with camera precision fix
fn hierarchicalVoxelDDA(
    rayOrigin: vec3<f32>,
    rayDir : vec3<f32>,
    gridMin : vec3<i32>,
    gridMax : vec3<i32>,
    cameraPos: vec3<f32>,
    cameraPosInt: vec3<i32>,
    cameraPosOffset: vec3<f32>
) -> DDAHit {
    let gridMinRelative = vec3<f32>(gridMin - cameraPosInt) - cameraPosOffset;
    let gridMaxRelative = vec3<f32>(gridMax - cameraPosInt) - cameraPosOffset;
    let gridMinF = gridMinRelative;
    let gridMaxF = gridMaxRelative + vec3<f32>(1.0, 1.0, 1.0);'''
)

# Replace the first voxel setup
code = code.replace(
'''    var p = rayOrigin + t * rayDir;

    // Use double precision for voxel calculation to reduce jitter
    var voxel = vec3<i32>(
        clamp(i32(floor(p.x + 0.5 * sign(rayDir.x) * 1e-6)), gridMin.x, gridMax.x),
        clamp(i32(floor(p.y + 0.5 * sign(rayDir.y) * 1e-6)), gridMin.y, gridMax.y),
        clamp(i32(floor(p.z + 0.5 * sign(rayDir.z) * 1e-6)), gridMin.z, gridMax.z)
    );''',
'''    var p = rayOrigin + t * rayDir;
    
    // PRECISION FIX: Calculate position relative to camera's integer voxel position
    let pWithOffset = p + cameraPosOffset;

    // Use double precision for voxel calculation to reduce jitter
    var voxel = vec3<i32>(
        clamp(i32(floor(pWithOffset.x + 0.5 * sign(rayDir.x) * 1e-6)) + cameraPosInt.x, gridMin.x, gridMax.x),
        clamp(i32(floor(pWithOffset.y + 0.5 * sign(rayDir.y) * 1e-6)) + cameraPosInt.y, gridMin.y, gridMax.y),
        clamp(i32(floor(pWithOffset.z + 0.5 * sign(rayDir.z) * 1e-6)) + cameraPosInt.z, gridMin.z, gridMax.z)
    );'''
)

# Replace the next[] calculation loops
# Loop 1
code = code.replace(
'''    for (var i = 0u; i < 3u; i = i + 1u) {
        let b = f32(voxel[i]) + select(0.0, 1.0, step[i] > 0);
        next[i] = (b - rayOrigin[i]) * invDir[i];
        delta[i] = abs(invDir[i]);
    }''',
'''    for (var i = 0u; i < 3u; i = i + 1u) {
        let voxelBoundary = f32(voxel[i]) + select(0.0, 1.0, step[i] > 0);
        let boundaryOffsetFromCameraInt = voxelBoundary - f32(cameraPosInt[i]);
        let boundaryRelativeToRayOrigin = boundaryOffsetFromCameraInt - cameraPosOffset[i];
        next[i] = (boundaryRelativeToRayOrigin - rayOrigin[i]) * invDir[i];
        delta[i] = abs(invDir[i]);
    }'''
)

# Replace sector min/max calculation
code = code.replace(
'''                    let sectorMin = vec3<f32>(
                        f32(currentSector.x * SECTOR_SIZE),
                        f32(currentSector.y * SECTOR_SIZE),
                        f32(currentSector.z * SECTOR_SIZE));
                    let sectorMax = vec3<f32>(
                        f32((currentSector.x + 1) * SECTOR_SIZE),
                        f32((currentSector.y + 1) * SECTOR_SIZE),
                        f32((currentSector.z + 1) * SECTOR_SIZE));''',
'''                    let sectorMin = vec3<f32>(
                        f32(currentSector.x * SECTOR_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32(currentSector.y * SECTOR_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32(currentSector.z * SECTOR_SIZE - cameraPosInt.z) - cameraPosOffset.z);
                    let sectorMax = vec3<f32>(
                        f32((currentSector.x + 1) * SECTOR_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32((currentSector.y + 1) * SECTOR_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32((currentSector.z + 1) * SECTOR_SIZE - cameraPosInt.z) - cameraPosOffset.z);'''
)

# Loop 2 inside sector skip
code = code.replace(
'''                        t = exitT + 1e-5;
                        p = rayOrigin + t * rayDir;
                        voxel = vec3<i32>(
                            clamp(i32(floor(p.x + 0.5 * sign(rayDir.x) * 1e-5)), gridMin.x, gridMax.x),
                            clamp(i32(floor(p.y + 0.5 * sign(rayDir.y) * 1e-5)), gridMin.y, gridMax.y),
                            clamp(i32(floor(p.z + 0.5 * sign(rayDir.z) * 1e-5)), gridMin.z, gridMax.z));
                        for (var j = 0u; j < 3u; j = j + 1u) {
                            let b = f32(voxel[j]) + select(0.0, 1.0, step[j] > 0);
                            next[j] = (b - rayOrigin[j]) * invDir[j];
                        }''',
'''                        t = exitT + 1e-5;
                        p = rayOrigin + t * rayDir;
                        let pWithOffset2 = p + cameraPosOffset;
                        voxel = vec3<i32>(
                            clamp(i32(floor(pWithOffset2.x + 0.5 * sign(rayDir.x) * 1e-5)) + cameraPosInt.x, gridMin.x, gridMax.x),
                            clamp(i32(floor(pWithOffset2.y + 0.5 * sign(rayDir.y) * 1e-5)) + cameraPosInt.y, gridMin.y, gridMax.y),
                            clamp(i32(floor(pWithOffset2.z + 0.5 * sign(rayDir.z) * 1e-5)) + cameraPosInt.z, gridMin.z, gridMax.z));
                        for (var j = 0u; j < 3u; j = j + 1u) {
                            let voxelBoundary2 = f32(voxel[j]) + select(0.0, 1.0, step[j] > 0);
                            let boundaryOffsetFromCameraInt2 = voxelBoundary2 - f32(cameraPosInt[j]);
                            let boundaryRelativeToRayOrigin2 = boundaryOffsetFromCameraInt2 - cameraPosOffset[j];
                            next[j] = (boundaryRelativeToRayOrigin2 - rayOrigin[j]) * invDir[j];
                        }'''
)

# Replace top-level min/max calculation
code = code.replace(
'''                    let topMin = vec3<f32>(f32(currentTopLevel.x * BRICK_SIZE),
                        f32(currentTopLevel.y * BRICK_SIZE),
                        f32(currentTopLevel.z * BRICK_SIZE));
                    let topMax = vec3<f32>(f32((currentTopLevel.x + 1) * BRICK_SIZE),
                        f32((currentTopLevel.y + 1) * BRICK_SIZE),
                        f32((currentTopLevel.z + 1) * BRICK_SIZE));''',
'''                    let topMin = vec3<f32>(
                        f32(currentTopLevel.x * BRICK_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32(currentTopLevel.y * BRICK_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32(currentTopLevel.z * BRICK_SIZE - cameraPosInt.z) - cameraPosOffset.z);
                    let topMax = vec3<f32>(
                        f32((currentTopLevel.x + 1) * BRICK_SIZE - cameraPosInt.x) - cameraPosOffset.x,
                        f32((currentTopLevel.y + 1) * BRICK_SIZE - cameraPosInt.y) - cameraPosOffset.y,
                        f32((currentTopLevel.z + 1) * BRICK_SIZE - cameraPosInt.z) - cameraPosOffset.z);'''
)

# Loop 3 inside top-level skip
code = code.replace(
'''                        t = exitT + 1e-5; // Add small offset to ensure we cross the boundary
                        p = rayOrigin + t * rayDir;

                        // Clamp to ensure we stay within bounds with directional bias
                        voxel = vec3<i32>(
                            clamp(i32(floor(p.x + 0.5 * sign(rayDir.x) * 1e-5)), gridMin.x, gridMax.x),
                            clamp(i32(floor(p.y + 0.5 * sign(rayDir.y) * 1e-5)), gridMin.y, gridMax.y),
                            clamp(i32(floor(p.z + 0.5 * sign(rayDir.z) * 1e-5)), gridMin.z, gridMax.z)
                        );

                        // Recalculate DDA state with better precision
                        for (var j = 0u; j < 3u; j = j + 1u) {
                            let b = f32(voxel[j]) + select(0.0, 1.0, step[j] > 0);
                            next[j] = (b - rayOrigin[j]) * invDir[j];
                        }''',
'''                        t = exitT + 1e-5; // Add small offset to ensure we cross the boundary
                        p = rayOrigin + t * rayDir;
                        let pWithOffset3 = p + cameraPosOffset;

                        // Clamp to ensure we stay within bounds with directional bias
                        voxel = vec3<i32>(
                            clamp(i32(floor(pWithOffset3.x + 0.5 * sign(rayDir.x) * 1e-5)) + cameraPosInt.x, gridMin.x, gridMax.x),
                            clamp(i32(floor(pWithOffset3.y + 0.5 * sign(rayDir.y) * 1e-5)) + cameraPosInt.y, gridMin.y, gridMax.y),
                            clamp(i32(floor(pWithOffset3.z + 0.5 * sign(rayDir.z) * 1e-5)) + cameraPosInt.z, gridMin.z, gridMax.z)
                        );

                        // Recalculate DDA state with better precision
                        for (var j = 0u; j < 3u; j = j + 1u) {
                            let voxelBoundary3 = f32(voxel[j]) + select(0.0, 1.0, step[j] > 0);
                            let boundaryOffsetFromCameraInt3 = voxelBoundary3 - f32(cameraPosInt[j]);
                            let boundaryRelativeToRayOrigin3 = boundaryOffsetFromCameraInt3 - cameraPosOffset[j];
                            next[j] = (boundaryRelativeToRayOrigin3 - rayOrigin[j]) * invDir[j];
                        }'''
)

# Update the main function
old_main = '''    let rayOrigin = position;
    let worldPos = invViewProj * ndc;
    let rayDir = normalize(worldPos.xyz / worldPos.w - rayOrigin);

    let gridMin = vec3<i32>(0, 0, 0);
    let gridMax = vec3<i32>(WORLD_SIZE - 1, WORLD_SIZE - 1, WORLD_SIZE - 1);

    let hit = hierarchicalVoxelDDA(rayOrigin, rayDir, gridMin, gridMax);'''

new_main = '''    // PRECISION FIX: Use camera-relative coordinates
    let cameraPos = position;
    let cameraPosInt = vec3<i32>(i32(floor(cameraPos.x)), i32(floor(cameraPos.y)), i32(floor(cameraPos.z)));
    let cameraPosOffset = cameraPos - vec3<f32>(cameraPosInt);

    let rayOrigin = vec3<f32>(0.0, 0.0, 0.0);
    let viewPos = invViewProj * ndc;
    let rayDir = normalize(viewPos.xyz / viewPos.w);

    let gridMin = vec3<i32>(0, 0, 0);
    let gridMax = vec3<i32>(WORLD_SIZE - 1, WORLD_SIZE - 1, WORLD_SIZE - 1);

    let hit = hierarchicalVoxelDDA(rayOrigin, rayDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset);'''
code = code.replace(old_main, new_main)

# Update calculateShadow/calculateReflection calls too
old_shadow = '''fn calculateShadow(hitPos: vec3<f32>, normal : vec3<f32>, lightDir : vec3<f32>, gridMin : vec3<i32>, gridMax : vec3<i32>) -> f32 {
    // Offset the ray start position slightly along the normal to avoid self-intersection
    let shadowRayOrigin = hitPos + normal * 0.01;

    // Cast shadow ray towards light
    let shadowHit = hierarchicalVoxelDDA(shadowRayOrigin, lightDir, gridMin, gridMax);'''

new_shadow = '''fn calculateShadow(hitPos: vec3<f32>, normal : vec3<f32>, lightDir : vec3<f32>, gridMin : vec3<i32>, gridMax : vec3<i32>, cameraPos: vec3<f32>, cameraPosInt: vec3<i32>, cameraPosOffset: vec3<f32>) -> f32 {
    // Offset the ray start position slightly along the normal to avoid self-intersection
    let shadowRayOriginWorld = hitPos + normal * 0.01;
    let shadowRayOriginRelative = shadowRayOriginWorld - cameraPos;

    // Cast shadow ray towards light
    let shadowHit = hierarchicalVoxelDDA(shadowRayOriginRelative, lightDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset);'''
code = code.replace(old_shadow, new_shadow)

old_reflect = '''fn calculateReflection(hitPos: vec3<f32>, rayDir : vec3<f32>, normal : vec3<f32>, gridMin : vec3<i32>, gridMax : vec3<i32>) -> vec4<f32> {
    // Calculate reflection direction
    let reflectDir = reflect(rayDir, normal);

    // Offset the ray start position slightly along the normal to avoid self-intersection
    let reflectRayOrigin = hitPos + normal * 0.01;

    // Cast reflection ray
    let reflectHit = hierarchicalVoxelDDA(reflectRayOrigin, reflectDir, gridMin, gridMax);'''

new_reflect = '''fn calculateReflection(hitPos: vec3<f32>, rayDir : vec3<f32>, normal : vec3<f32>, gridMin : vec3<i32>, gridMax : vec3<i32>, cameraPos: vec3<f32>, cameraPosInt: vec3<i32>, cameraPosOffset: vec3<f32>) -> vec4<f32> {
    // Calculate reflection direction
    let reflectDir = reflect(rayDir, normal);

    // Offset the ray start position slightly along the normal to avoid self-intersection
    let reflectRayOriginWorld = hitPos + normal * 0.01;
    let reflectRayOriginRelative = reflectRayOriginWorld - cameraPos;

    // Cast reflection ray
    let reflectHit = hierarchicalVoxelDDA(reflectRayOriginRelative, reflectDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset);'''
code = code.replace(old_reflect, new_reflect)

# Update calls to shadow/reflection in main
code = code.replace(
    'let shadowFactor = calculateShadow(hitPos, hit.normal, lightDir, gridMin, gridMax);',
    'let shadowFactor = calculateShadow(hitPos, hit.normal, lightDir, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset);'
)
code = code.replace(
    'let reflectionColor = calculateReflection(hitPos, rayDir, hit.normal, gridMin, gridMax);',
    'let reflectionColor = calculateReflection(hitPos, rayDir, hit.normal, gridMin, gridMax, cameraPos, cameraPosInt, cameraPosOffset);'
)

with open('res/compute.wgsl', 'w') as f:
    f.write(code)

print("Applied precision fixes to wgsl!")
