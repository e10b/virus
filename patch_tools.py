import sys

with open('example/compute.h', 'r') as f:
    text = f.read()

old_logic = """            if (add) {
                if (isSolid) {
                    if (lastEmpty.x >= 0 && lastEmpty.x < worldSize &&
                        lastEmpty.y >= 0 && lastEmpty.y < worldSize &&
                        lastEmpty.z >= 0 && lastEmpty.z < worldSize) {

                        glm::ivec3 center = lastEmpty;
                        if (toolMode == 1) {
                            setVoxel(center.x, center.y, center.z, 232);
                            updateGPUBuffers(center.x, center.y, center.z);
                        } else if (toolMode == 2) {
                            for (int dx = -3; dx <= 4; dx++) {
                                for (int dy = -3; dy <= 4; dy++) {
                                    for (int dz = -3; dz <= 4; dz++) {
                                        setVoxel(center.x + dx, center.y + dy, center.z + dz, 232);
                                        updateGPUBuffers(center.x + dx, center.y + dy, center.z + dz);
                                    }
                                }
                            }
                        } else if (toolMode == 3) {
                            for (int dx = -8; dx <= 8; dx++) {
                                for (int dy = -8; dy <= 8; dy++) {
                                    for (int dz = -8; dz <= 8; dz++) {
                                        if (dx*dx + dy*dy + dz*dz <= 64) {
                                            setVoxel(center.x + dx, center.y + dy, center.z + dz, 232);
                                            updateGPUBuffers(center.x + dx, center.y + dy, center.z + dz);
                                        }
                                    }
                                }
                            }
                        }
                        return true;
                    }
                    return false;
                }
            }
            else {
                if (isSolid) {
                    glm::ivec3 center = voxel;
                    if (toolMode == 1) {
                        clearVoxel(center.x, center.y, center.z);
                        updateGPUBuffers(center.x, center.y, center.z);
                    } else if (toolMode == 2) {
                        for (int dx = -3; dx <= 4; dx++) {
                            for (int dy = -3; dy <= 4; dy++) {
                                for (int dz = -3; dz <= 4; dz++) {
                                    clearVoxel(center.x + dx, center.y + dy, center.z + dz);
                                    updateGPUBuffers(center.x + dx, center.y + dy, center.z + dz);
                                }
                            }
                        }
                    } else if (toolMode == 3) {
                        for (int dx = -8; dx <= 8; dx++) {
                            for (int dy = -8; dy <= 8; dy++) {
                                for (int dz = -8; dz <= 8; dz++) {
                                    if (dx*dx + dy*dy + dz*dz <= 64) {
                                        clearVoxel(center.x + dx, center.y + dy, center.z + dz);
                                        updateGPUBuffers(center.x + dx, center.y + dy, center.z + dz);
                                    }
                                }
                            }
                        }
                    }
                    return true;
                }
            }"""

new_logic = """            if ((add && isSolid && lastEmpty.x >= 0 && lastEmpty.x < worldSize &&
                 lastEmpty.y >= 0 && lastEmpty.y < worldSize &&
                 lastEmpty.z >= 0 && lastEmpty.z < worldSize) || 
                (!add && isSolid)) {
                
                glm::ivec3 center = add ? lastEmpty : voxel;
                int rCube = 0;
                int rSphere = 0;
                
                if (toolMode == 1) {} // 1x1x1 handled separately below
                else if (toolMode == 2) { rCube = 4; } // ~8x8x8
                else if (toolMode == 3) { rSphere = 8; }
                else if (toolMode == 5) { rCube = 16; } // 32x32x32
                else if (toolMode == 6) { rSphere = 16; }
                else if (toolMode == 7) { rCube = 64; } // 128x128x128
                else if (toolMode == 8) { rSphere = 64; }
                
                if (rCube > 0) {
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
                }
                
                return true;
            }"""

if old_logic in text:
    text = text.replace(old_logic, new_logic)
    with open('example/compute.h', 'w') as f:
        f.write(text)
    print("Patched successfully!")
else:
    print("Could not find TargetContent!")
