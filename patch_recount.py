import sys

with open('example/compute.h', 'r') as f:
    text = f.read()

old_logic = """        for (auto& t : threads) {
            t.join();
        }
        
        updateAllSectors();"""

new_logic = """        for (auto& t : threads) {
            t.join();
        }
        
        size_t allocatedCount = currentBrickPoolSize.load();
        std::vector<std::thread> recountThreads;
        for (unsigned int t = 0; t < numThreads; ++t) {
            recountThreads.push_back(std::thread([&, t]() {
                size_t startIdx = (allocatedCount * t) / numThreads;
                size_t endIdx = (allocatedCount * (t + 1)) / numThreads;
                for (size_t i = startIdx; i < endIdx; ++i) {
                    if (brickMetadata[i].inUse) {
                        uint16_t count = 0;
                        for (int j = 0; j < 512; ++j) {
                            if (brickPool[i * 512 + j] != 0) count++;
                        }
                        brickMetadata[i].nonEmptyCount = count;
                    }
                }
            }));
        }
        for (auto& rt : recountThreads) {
            rt.join();
        }
        
        updateAllSectors();"""

if old_logic in text:
    text = text.replace(old_logic, new_logic)
    with open('example/compute.h', 'w') as f:
        f.write(text)
    print("Patched recount successfully!")
else:
    print("Could not find TargetContent!")
