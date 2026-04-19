#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <map>
#include <memory>
#include "itexturemanager.h"
#include "objloader.h"

// Forward declaration for texture sampling
extern "C" {
    unsigned char* stbi_load(char const* filename, int* x, int* y, int* channels_in_file, int desired_channels);
    void stbi_image_free(void* retval_from_stbi_load);
    const char* stbi_failure_reason(void);
}

class Voxelizer {
private:
    // Note: Old texture caching system removed - now using TextureManager
    
    // Calculate UV coordinates for a point within a triangle using barycentric coordinates
    static glm::vec2 calculateUVForPoint(const glm::vec3& point, 
                                         const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                                         const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2) {
        // Calculate barycentric coordinates
        glm::vec3 v0v1 = v1 - v0;
        glm::vec3 v0v2 = v2 - v0;
        glm::vec3 v0p = point - v0;
        
        float dot00 = glm::dot(v0v2, v0v2);
        float dot01 = glm::dot(v0v2, v0v1);
        float dot02 = glm::dot(v0v2, v0p);
        float dot11 = glm::dot(v0v1, v0v1);
        float dot12 = glm::dot(v0v1, v0p);
        
        float denom = dot00 * dot11 - dot01 * dot01;
        if (abs(denom) < 1e-6) {
            // Degenerate triangle, return center UV
            return (uv0 + uv1 + uv2) / 3.0f;
        }
        
        float invDenom = 1.0f / denom;
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
        float w = 1.0f - u - v;
        
        // Interpolate UV coordinates using barycentric coordinates
        return w * uv0 + u * uv1 + v * uv2;
    }

    // Median Cut Algorithm for color quantization
    struct ColorBox {
        std::vector<uint32_t> colors;
        
        uint32_t getAverageColor() const {
            if (colors.empty()) return 0;
            uint32_t rSum = 0, gSum = 0, bSum = 0;
            for (uint32_t c : colors) {
                rSum += (c >> 16) & 0xFF;
                gSum += (c >> 8) & 0xFF;
                bSum += c & 0xFF;
            }
            uint8_t r = rSum / colors.size();
            uint8_t g = gSum / colors.size();
            uint8_t b = bSum / colors.size();
            return ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
        }
        
        void split(ColorBox& box1, ColorBox& box2) {
            if (colors.size() <= 1) {
                box1.colors = colors;
                return;
            }
            
            // Find axis with largest range
            uint32_t rMin = 255, rMax = 0;
            uint32_t gMin = 255, gMax = 0;
            uint32_t bMin = 255, bMax = 0;
            
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
            
            // Sort by the axis with largest range
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
            
            // Split at median
            size_t mid = sorted.size() / 2;
            box1.colors.assign(sorted.begin(), sorted.begin() + mid);
            box2.colors.assign(sorted.begin() + mid, sorted.end());
        }
    };
    
    static void medianCutQuantize(std::vector<ColorBox>& boxes, int targetColors) {
        while ((int)boxes.size() < targetColors) {
            // Find box with most colors to split
            size_t maxIdx = 0;
            size_t maxSize = boxes[0].colors.size();
            for (size_t i = 1; i < boxes.size(); ++i) {
                if (boxes[i].colors.size() > maxSize) {
                    maxSize = boxes[i].colors.size();
                    maxIdx = i;
                }
            }
            
            // Split the box
            ColorBox newBox;
            boxes[maxIdx].split(boxes[maxIdx], newBox);
            if (newBox.colors.size() > 0) {
                boxes.push_back(newBox);
            } else {
                break;
            }
        }
    }


public:
    // Pack color into voxel data (RGBA8 format in uint32_t)
    static uint32_t packColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        return (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
    }
    
    // Note: Texture cleanup now handled by TextureManager

    static void voxelizeModel(const std::vector<Triangle>& triangles, 
                             uint32_t* voxelData, 
                             int sizeX, int sizeY, int sizeZ,
                             ITextureManager& textureManager) {
        
        // Clear voxel data
        size_t voxelCount = sizeX * sizeY * sizeZ;
        memset(voxelData, 0, voxelCount * sizeof(uint32_t));
        
        if (triangles.empty()) {
            printf("No triangles to voxelize!\n");
            return;
        }
        
        // Find bounding box of the model
        glm::vec3 minBounds = triangles[0].v0;
        glm::vec3 maxBounds = triangles[0].v0;
        
        for (const Triangle& tri : triangles) {
            minBounds = glm::min(minBounds, tri.v0);
            minBounds = glm::min(minBounds, tri.v1);
            minBounds = glm::min(minBounds, tri.v2);
            
            maxBounds = glm::max(maxBounds, tri.v0);
            maxBounds = glm::max(maxBounds, tri.v1);
            maxBounds = glm::max(maxBounds, tri.v2);
        }
        
        printf("Model bounds: min(%.2f, %.2f, %.2f) max(%.2f, %.2f, %.2f)\n",
               minBounds.x, minBounds.y, minBounds.z,
               maxBounds.x, maxBounds.y, maxBounds.z);
        
        // Calculate scale to fit model in voxel grid
        glm::vec3 modelSize = maxBounds - minBounds;
        float maxDimension = std::max({modelSize.x, modelSize.y, modelSize.z});

        // Use full grid capacity minus small margin for safety
        float scale = (std::min({ sizeX, sizeY, sizeZ }) * 0.95f) / maxDimension;

        // Center the model in the voxel grid - proper centering formula
        glm::vec3 modelCenter = (minBounds + maxBounds) * 0.5f;
        glm::vec3 gridCenter = glm::vec3(sizeX - 1, sizeY - 1, sizeZ - 1) * 0.5f;
        glm::vec3 offset = gridCenter - modelCenter * scale;

        printf("Scale: %.6f, GridCenter: (%.2f, %.2f, %.2f), ModelCenter: (%.2f, %.2f, %.2f)\n", 
               scale, gridCenter.x, gridCenter.y, gridCenter.z, modelCenter.x, modelCenter.y, modelCenter.z);
        printf("Offset: (%.2f, %.2f, %.2f)\n", offset.x, offset.y, offset.z);
        
        // Voxelize each triangle
        int voxelizedTriangles = 0;
        int debugVoxelCount = 0; // For debug output limiting
        for (const Triangle& tri : triangles) {
            // Transform triangle vertices to voxel space
            glm::vec3 v0 = tri.v0 * scale + offset;
            glm::vec3 v1 = tri.v1 * scale + offset;
            glm::vec3 v2 = tri.v2 * scale + offset;
            
            // Debug: print the first triangle's info
            if (voxelizedTriangles == 0) {
                printf("First triangle: fallback color(%.3f, %.3f, %.3f), textureIndex=%u\n", 
                       tri.color.r, tri.color.g, tri.color.b, tri.textureIndex);
            }
            
            // Find bounding box of the triangle in voxel space
            glm::vec3 triMin = glm::min(v0, glm::min(v1, v2));
            glm::vec3 triMax = glm::max(v0, glm::max(v1, v2));
            
            // Clamp to grid bounds
            int minX = std::max(0, (int)std::floor(triMin.x));
            int minY = std::max(0, (int)std::floor(triMin.y));
            int minZ = std::max(0, (int)std::floor(triMin.z));
            int maxX = std::min(sizeX - 1, (int)std::ceil(triMax.x));
            int maxY = std::min(sizeY - 1, (int)std::ceil(triMax.y));
            int maxZ = std::min(sizeZ - 1, (int)std::ceil(triMax.z));
            
            // Test each voxel in the bounding box
            for (int z = minZ; z <= maxZ; ++z) {
                for (int y = minY; y <= maxY; ++y) {
                    for (int x = minX; x <= maxX; ++x) {
                        // Test if voxel center is inside or near the triangle
                        glm::vec3 voxelCenter = glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f);
                        
                        if (isPointInTriangle(voxelCenter, v0, v1, v2) ||
                            distanceToTriangle(voxelCenter, v0, v1, v2) < 0.7f) { // Slightly expand for better coverage
                            
                            uint32_t voxelColor;
                            
                            // If triangle has texture, sample it at the voxel's UV coordinates
                            if (tri.textureIndex > 0) {
                                glm::vec2 voxelUV = calculateUVForPoint(voxelCenter, v0, v1, v2, tri.uv0, tri.uv1, tri.uv2);
                                glm::vec3 sampledColor = textureManager.sampleTexture(tri.textureIndex, voxelUV);
                                
                                uint8_t r = (uint8_t)(sampledColor.r * 255.0f);
                                uint8_t g = (uint8_t)(sampledColor.g * 255.0f);
                                uint8_t b = (uint8_t)(sampledColor.b * 255.0f);
                                voxelColor = packColor(r, g, b);
                                
                                // Debug: print first few UV samples only
                                if (voxelizedTriangles == 0 && debugVoxelCount < 2) {
                                    printf("  Voxel (%d,%d,%d) UV(%.3f,%.3f) -> color(%.3f,%.3f,%.3f)\n", 
                                           x, y, z, voxelUV.x, voxelUV.y, sampledColor.r, sampledColor.g, sampledColor.b);
                                }
                            } else {
                                // No texture, use material color
                                uint8_t r = (uint8_t)(tri.color.r * 255.0f);
                                uint8_t g = (uint8_t)(tri.color.g * 255.0f);
                                uint8_t b = (uint8_t)(tri.color.b * 255.0f);
                                voxelColor = packColor(r, g, b);
                            }
                            
                            size_t idx = x + y * sizeX + z * sizeX * sizeY;
                            voxelData[idx] = voxelColor;
                            
                            if (voxelizedTriangles == 0 && debugVoxelCount < 3) {
                                debugVoxelCount++;
                            }
                        }
                    }
                }
            }
            voxelizedTriangles++;
        }
        
        printf("Voxelized %d triangles\n", voxelizedTriangles);
        
        // Count solid voxels
        int solidVoxels = 0;
        for (size_t i = 0; i < voxelCount; ++i) {
            if (voxelData[i] != 0) solidVoxels++;
        }
        printf("Created %d solid voxels out of %zu total\n", solidVoxels, voxelCount);
    }

private:
    // Check if point is inside triangle using barycentric coordinates
    static bool isPointInTriangle(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
        glm::vec3 v0v1 = v1 - v0;
        glm::vec3 v0v2 = v2 - v0;
        glm::vec3 v0p = p - v0;
        
        float dot00 = glm::dot(v0v2, v0v2);
        float dot01 = glm::dot(v0v2, v0v1);
        float dot02 = glm::dot(v0v2, v0p);
        float dot11 = glm::dot(v0v1, v0v1);
        float dot12 = glm::dot(v0v1, v0p);
        
        float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
        
        return (u >= 0) && (v >= 0) && (u + v <= 1);
    }
    
    // Calculate distance from point to triangle
    static float distanceToTriangle(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
        glm::vec3 edge0 = v1 - v0;
        glm::vec3 edge1 = v2 - v0;
        glm::vec3 v0p = p - v0;
        
        float a = glm::dot(edge0, edge0);
        float b = glm::dot(edge0, edge1);
        float c = glm::dot(edge1, edge1);
        float d = glm::dot(edge0, v0p);
        float e = glm::dot(edge1, v0p);
        
        float det = a * c - b * b;
        float s = b * e - c * d;
        float t = b * d - a * e;
        
        if (s + t < det) {
            if (s < 0.0f) {
                if (t < 0.0f) {
                    // region 4
                    if (d < 0.0f) {
                        t = 0.0f;
                        s = -d >= a ? 1.0f : -d / a;
                    } else {
                        s = 0.0f;
                        t = e >= 0.0f ? 0.0f : (-e >= c ? 1.0f : -e / c);
                    }
                } else {
                    // region 3
                    s = 0.0f;
                    t = e >= 0.0f ? 0.0f : (-e >= c ? 1.0f : -e / c);
                }
            } else if (t < 0.0f) {
                // region 5
                t = 0.0f;
                s = d >= 0.0f ? 0.0f : (-d >= a ? 1.0f : -d / a);
            } else {
                // region 0
                float invDet = 1.0f / det;
                s *= invDet;
                t *= invDet;
            }
        } else {
            if (s < 0.0f) {
                // region 2
                float tmp0 = b + d;
                float tmp1 = c + e;
                if (tmp1 > tmp0) {
                    float numer = tmp1 - tmp0;
                    float denom = a - 2 * b + c;
                    s = numer >= denom ? 1.0f : numer / denom;
                    t = 1.0f - s;
                } else {
                    t = 0.0f;
                    s = tmp0 <= 0.0f ? 1.0f : (d >= 0.0f ? 0.0f : -d / a);
                }
            } else if (t < 0.0f) {
                // region 6
                float tmp0 = b + e;
                float tmp1 = a + d;
                if (tmp1 > tmp0) {
                    float numer = tmp1 - tmp0;
                    float denom = a - 2 * b + c;
                    t = numer >= denom ? 1.0f : numer / denom;
                    s = 1.0f - t;
                } else {
                    s = 0.0f;
                    t = tmp0 <= 0.0f ? 1.0f : (e >= 0.0f ? 0.0f : -e / c);
                }
            } else {
                // region 1
                float numer = c + e - b - d;
                if (numer <= 0.0f) {
                    s = 0.0f;
                } else {
                    float denom = a - 2 * b + c;
                    s = numer >= denom ? 1.0f : numer / denom;
                }
                t = 1.0f - s;
            }
        }
        
        glm::vec3 closest = v0 + s * edge0 + t * edge1;
        return glm::length(p - closest);
    }
};
