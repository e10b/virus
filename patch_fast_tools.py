import sys

with open('example/compute.h', 'r') as f:
    text = f.read()

old_logic = """                if (rCube > 0) {
                    for (int dx = -rCube; dx <= rCube; dx++) {
                        for (int dy = -rCube; dy <= rCube; dy++) {
                            for (int dz = -rCube; dz <= rCube; dz++) {
                                if (add) setVoxel(center.x + dx, center.y + dy, center.z + dz, 232);
                                else clearVoxel(center.x + dx, center.y + dy, center.z + dz);
                                updateGPUBuffers(center.x + dx, center.y + dy, center.z + dz);
                            }
                        }
                    }
                } else if (rSphere > 0) {
                    int rSq = rSphere * rSphere;
                    for (int dx = -rSphere; dx <= rSphere; dx++) {
                        for (int dy = -rSphere; dy <= rSphere; dy++) {
                            for (int dz = -rSphere; dz <= rSphere; dz++) {
                                if (dx*dx + dy*dy + dz*dz <= rSq) {
                                    if (add) setVoxel(center.x + dx, center.y + dy, center.z + dz, 232);
                                    else clearVoxel(center.x + dx, center.y + dy, center.z + dz);
                                    updateGPUBuffers(center.x + dx, center.y + dy, center.z + dz);
                                }
                            }
                        }
                    }
                } else if (toolMode == 1) { // 1x1x1
                    if (add) setVoxel(center.x, center.y, center.z, 232);
                    else clearVoxel(center.x, center.y, center.z);
                    updateGPUBuffers(center.x, center.y, center.z);
                }"""

new_logic = """                if (rCube > 0) {
                    for (int dx = -rCube; dx <= rCube; dx++) {
                        for (int dy = -rCube; dy <= rCube; dy++) {
                            for (int dz = -rCube; dz <= rCube; dz++) {
                                if (add) setVoxel(center.x + dx, center.y + dy, center.z + dz, 232);
                                else clearVoxel(center.x + dx, center.y + dy, center.z + dz);
                            }
                        }
                    }
                    int minTopX = std::max(0, (center.x - rCube) / brickSize);
                    int maxTopX = std::min(topLevelSize - 1, (center.x + rCube) / brickSize);
                    int minTopY = std::max(0, (center.y - rCube) / brickSize);
                    int maxTopY = std::min(topLevelSize - 1, (center.y + rCube) / brickSize);
                    int minTopZ = std::max(0, (center.z - rCube) / brickSize);
                    int maxTopZ = std::min(topLevelSize - 1, (center.z + rCube) / brickSize);
                    for(int tz=minTopZ; tz<=maxTopZ; ++tz) {
                        for(int ty=minTopY; ty<=maxTopY; ++ty) {
                            for(int tx=minTopX; tx<=maxTopX; ++tx) {
                                pendingUpdates.insert(tx + ty * topLevelSize + tz * topLevelSize * topLevelSize);
                            }
                        }
                    }
                } else if (rSphere > 0) {
                    int rSq = rSphere * rSphere;
                    for (int dx = -rSphere; dx <= rSphere; dx++) {
                        for (int dy = -rSphere; dy <= rSphere; dy++) {
                            for (int dz = -rSphere; dz <= rSphere; dz++) {
                                if (dx*dx + dy*dy + dz*dz <= rSq) {
                                    if (add) setVoxel(center.x + dx, center.y + dy, center.z + dz, 232);
                                    else clearVoxel(center.x + dx, center.y + dy, center.z + dz);
                                }
                            }
                        }
                    }
                    int minTopX = std::max(0, (center.x - rSphere) / brickSize);
                    int maxTopX = std::min(topLevelSize - 1, (center.x + rSphere) / brickSize);
                    int minTopY = std::max(0, (center.y - rSphere) / brickSize);
                    int maxTopY = std::min(topLevelSize - 1, (center.y + rSphere) / brickSize);
                    int minTopZ = std::max(0, (center.z - rSphere) / brickSize);
                    int maxTopZ = std::min(topLevelSize - 1, (center.z + rSphere) / brickSize);
                    for(int tz=minTopZ; tz<=maxTopZ; ++tz) {
                        for(int ty=minTopY; ty<=maxTopY; ++ty) {
                            for(int tx=minTopX; tx<=maxTopX; ++tx) {
                                pendingUpdates.insert(tx + ty * topLevelSize + tz * topLevelSize * topLevelSize);
                            }
                        }
                    }
                } else if (toolMode == 1) { // 1x1x1
                    if (add) setVoxel(center.x, center.y, center.z, 232);
                    else clearVoxel(center.x, center.y, center.z);
                    updateGPUBuffers(center.x, center.y, center.z);
                }"""

if old_logic in text:
    text = text.replace(old_logic, new_logic)
    with open('example/compute.h', 'w') as f:
        f.write(text)
    print("Patched successfully!")
else:
    print("Could not find TargetContent!")
