#include "color_palette.h"

#include <algorithm>
#include <cstdio>

void VoxelCompute::initializeTerrainPalette() {
    colorPalette.colorCount = 256;
    for (int i = 0; i < 256; ++i) {
        colorPalette.colors[i] = 0xFF000000;
    }

    colorPalette.colors[1] = 0xFFE0D3A2;
    colorPalette.colors[2] = 0xFF6FA343;
    colorPalette.colors[3] = 0xFF8A6A4A;
    colorPalette.colors[4] = 0xFF6B7078;
    colorPalette.colors[5] = 0xFFD9DDE3;
    colorPalette.colors[6] = 0xFF4E5A63;
    colorPalette.colors[7] = 0xFF89B55A;
    colorPalette.colors[PLAYER_PLACED_MATERIAL] = 0xFF2E6BFF;
    colorPalette.colors[VoxelCompute::SOUND_VOXEL_MATERIAL] = 0xFFAA00FF;  // Purple
}

void VoxelCompute::buildColorPalette(const std::set<uint32_t>& uniqueColors) {
    if (uniqueColors.empty()) {
        colorPalette.colorCount = 1;
        colorPalette.colors[0] = 0xFF808080;
        return;
    }

    struct ColorBox {
        std::vector<uint32_t> colors;

        uint32_t getAverageColor() const {
            if (colors.empty()) return 0;
            uint64_t rSum = 0, gSum = 0, bSum = 0;
            for (uint32_t c : colors) {
                rSum += (c >> 16) & 0xFF;
                gSum += (c >> 8) & 0xFF;
                bSum += c & 0xFF;
            }
            uint8_t r = rSum / colors.size();
            uint8_t g = gSum / colors.size();
            uint8_t b = bSum / colors.size();
            return 0xFF000000 | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
        }

        void split(ColorBox& box1, ColorBox& box2) {
            if (colors.size() <= 1) {
                box1.colors = colors;
                return;
            }

            uint32_t rMin = 255, rMax = 0, gMin = 255, gMax = 0, bMin = 255, bMax = 0;
            for (uint32_t c : colors) {
                uint8_t r = (c >> 16) & 0xFF;
                uint8_t g = (c >> 8) & 0xFF;
                uint8_t b = c & 0xFF;
                rMin = std::min(rMin, (uint32_t)r);
                rMax = std::max(rMax, (uint32_t)r);
                gMin = std::min(gMin, (uint32_t)g);
                gMax = std::max(gMax, (uint32_t)g);
                bMin = std::min(bMin, (uint32_t)b);
                bMax = std::max(bMax, (uint32_t)b);
            }

            uint32_t rRange = rMax - rMin;
            uint32_t gRange = gMax - gMin;
            uint32_t bRange = bMax - bMin;

            std::vector<uint32_t> sorted = colors;
            if (rRange >= gRange && rRange >= bRange) {
                std::sort(sorted.begin(), sorted.end(), [](uint32_t a, uint32_t b) {
                    return ((a >> 16) & 0xFF) < ((b >> 16) & 0xFF);
                });
            } else if (gRange >= bRange) {
                std::sort(sorted.begin(), sorted.end(), [](uint32_t a, uint32_t b) {
                    return ((a >> 8) & 0xFF) < ((b >> 8) & 0xFF);
                });
            } else {
                std::sort(sorted.begin(), sorted.end(), [](uint32_t a, uint32_t b) {
                    return (a & 0xFF) < (b & 0xFF);
                });
            }

            size_t mid = sorted.size() / 2;
            box1.colors.assign(sorted.begin(), sorted.begin() + mid);
            box2.colors.assign(sorted.begin() + mid, sorted.end());
        }
    };

    std::vector<ColorBox> boxes(1);
    boxes[0].colors.assign(uniqueColors.begin(), uniqueColors.end());

    while ((int)boxes.size() < 256 && (int)boxes.size() < (int)uniqueColors.size()) {
        size_t maxIdx = 0;
        size_t maxSize = boxes[0].colors.size();
        for (size_t i = 1; i < boxes.size(); ++i) {
            if (boxes[i].colors.size() > maxSize) {
                maxSize = boxes[i].colors.size();
                maxIdx = i;
            }
        }

        ColorBox newBox;
        boxes[maxIdx].split(boxes[maxIdx], newBox);
        if (!newBox.colors.empty()) {
            boxes.push_back(newBox);
        } else {
            break;
        }
    }

    colorPalette.colorCount = std::min(256, (int)boxes.size());
    for (int i = 0; i < colorPalette.colorCount; ++i) {
        colorPalette.colors[i] = boxes[i].getAverageColor();
    }

    printf("Built palette with %d colors from %zu unique colors\n", colorPalette.colorCount, uniqueColors.size());
}

uint8_t VoxelCompute::findClosestPaletteIndex(uint32_t color) const {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    uint32_t bestDist = 0xFFFFFFFF;
    uint8_t bestIdx = 0;

    for (int i = 0; i < colorPalette.colorCount; ++i) {
        uint32_t p = colorPalette.colors[i];
        uint8_t pr = (p >> 16) & 0xFF;
        uint8_t pg = (p >> 8) & 0xFF;
        uint8_t pb = p & 0xFF;

        int dr = (int)r - (int)pr;
        int dg = (int)g - (int)pg;
        int db = (int)b - (int)pb;
        uint32_t dist = dr * dr + dg * dg + db * db;

        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }

    return bestIdx;
}

namespace compute_layers {

void initializeTerrainPalette(VoxelCompute& volume) {
    volume.initializeTerrainPalette();
}

void buildColorPalette(VoxelCompute& volume, const std::set<uint32_t>& uniqueColors) {
    volume.buildColorPalette(uniqueColors);
}

uint8_t findClosestPaletteIndex(const VoxelCompute& volume, uint32_t color) {
    return volume.findClosestPaletteIndex(color);
}

} // namespace compute_layers
