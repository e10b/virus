#pragma once

#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <glm/glm.hpp>
#include "itexturemanager.h"

// Forward declarations for stb_image functions
extern "C" {
    unsigned char* stbi_load(char const* filename, int* x, int* y, int* channels_in_file, int desired_channels);
    unsigned char* stbi_load_from_memory(unsigned char const* buffer, int len, int* x, int* y, int* channels_in_file, int desired_channels);
    void stbi_image_free(void* retval_from_stbi_load);
    const char* stbi_failure_reason(void);
}

struct Triangle {
    glm::vec3 v0, v1, v2;
    glm::vec2 uv0, uv1, uv2; // UV coordinates for each vertex
    glm::vec3 color; // RGB color (0.0 to 1.0)
    uint32_t textureIndex; // Index into the texture array (0 = no texture)
    
    Triangle() : color(1.0f, 0.0f, 0.0f), uv0(0.0f), uv1(0.0f), uv2(0.0f), textureIndex(0) {} // Default to red, no texture
};

// Compile-time verification that Triangle has textureIndex member
static_assert(sizeof(Triangle) > 0, "Triangle struct must be defined");
static_assert(std::is_same_v<decltype(Triangle{}.textureIndex), uint32_t>, "Triangle must have uint32_t textureIndex member");

class ObjLoader {
private:
    struct Material {
        std::string name;
        glm::vec3 diffuse; // Kd - diffuse color
        glm::vec3 ambient; // Ka - ambient color
        std::string diffuseTexture; // map_Kd (filename only)
        std::string fullTexturePath; // Full resolved path to texture
        
        Material() : diffuse(1.0f, 0.0f, 0.0f), ambient(1.0f, 1.0f, 1.0f) {} // Default red diffuse, white ambient
    };
    
    // Generate a color from material name hash
    static glm::vec3 generateColorFromName(const std::string& name) {
        // Simple hash function
        size_t hash = 0;
        for (char c : name) {
            hash = hash * 31 + c;
        }
        
        // Convert hash to RGB values
        float r = ((hash & 0xFF) / 255.0f);
        float g = (((hash >> 8) & 0xFF) / 255.0f);
        float b = (((hash >> 16) & 0xFF) / 255.0f);
        
        // Ensure colors aren't too dark
        r = 0.3f + r * 0.7f;
        g = 0.3f + g * 0.7f;
        b = 0.3f + b * 0.7f;
        
        return glm::vec3(r, g, b);
    }
    
    // Note: Old sampleTextureAtUV function removed - now using TextureManager

    // Load texture and get average color
    static glm::vec3 getAverageColorFromTexture(const std::string& texturePath) {
        printf("Attempting to load texture: '%s'\n", texturePath.c_str());
        
        int width, height, channels;
        unsigned char* data = stbi_load(texturePath.c_str(), &width, &height, &channels, 3); // Force RGB
        
        if (!data) {
            printf("Failed to load texture: %s\n", texturePath.c_str());
            printf("  stbi_load returned NULL\n");
            printf("  stbi error: %s\n", stbi_failure_reason());
            return glm::vec3(0.7f, 0.7f, 0.7f); // Default gray
        }
        
        printf("Successfully loaded texture: %dx%d, %d channels\n", width, height, channels);
        
        // Calculate average color
        long long totalR = 0, totalG = 0, totalB = 0;
        int totalPixels = width * height;
        
        for (int i = 0; i < totalPixels * 3; i += 3) {
            totalR += data[i];
            totalG += data[i + 1];
            totalB += data[i + 2];
        }
        
        stbi_image_free(data);
        
        // Convert to [0,1] range
        float avgR = (float)totalR / (totalPixels * 255.0f);
        float avgG = (float)totalG / (totalPixels * 255.0f);
        float avgB = (float)totalB / (totalPixels * 255.0f);
        
        printf("Texture '%s' average color: (%.3f, %.3f, %.3f)\n", texturePath.c_str(), avgR, avgG, avgB);
        
        return glm::vec3(avgR, avgG, avgB);
    }
    
    static std::vector<Material> loadMTL(const std::string& mtlPath, const std::string& baseDirectory = "") {
        std::vector<Material> materials;
        
        FILE* file = fopen(mtlPath.c_str(), "r");
        if (!file) {
            printf("Failed to open MTL file: %s\n", mtlPath.c_str());
            return materials;
        }
        
        Material currentMaterial;
        bool hasMaterial = false;
        
        char line[1024];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "newmtl ", 7) == 0) {
                // Save previous material if it exists
                if (hasMaterial) {
                    // Resolve full texture path if texture exists
                    if (!currentMaterial.diffuseTexture.empty()) {
                        std::string fullTexturePath = currentMaterial.diffuseTexture;
                        // Handle relative paths that start with "./"
                        if (fullTexturePath.substr(0, 2) == "./") {
                            fullTexturePath = baseDirectory + fullTexturePath.substr(2);
                        } else if (fullTexturePath[0] != '/' && fullTexturePath[0] != '\\' && fullTexturePath.find(':') == std::string::npos) {
                            // Relative path without "./"
                            fullTexturePath = baseDirectory + fullTexturePath;
                        }
                        currentMaterial.fullTexturePath = fullTexturePath;
                        printf("Resolved texture path: '%s'\n", fullTexturePath.c_str());
                        
                        // For materials without explicit Kd, use white as fallback when there's a texture
                        if (currentMaterial.diffuse == glm::vec3(1.0f, 0.0f, 0.0f)) {
                            currentMaterial.diffuse = glm::vec3(1.0f, 1.0f, 1.0f); // White - let texture provide color
                            printf("Material '%s' has texture but no Kd - using white fallback\n", currentMaterial.name.c_str());
                        }
                    } else if (currentMaterial.diffuse == glm::vec3(1.0f, 0.0f, 0.0f)) {
                        // No texture and no explicit diffuse color - generate color from name
                        currentMaterial.diffuse = generateColorFromName(currentMaterial.name);
                    }
                    materials.push_back(currentMaterial);
                }
                
                // Start new material
                currentMaterial = Material();
                char materialName[256];
                sscanf(line, "newmtl %s", materialName);
                currentMaterial.name = std::string(materialName);
                hasMaterial = true;
            }
            else if (strncmp(line, "Kd ", 3) == 0) {
                // Diffuse color
                float r, g, b;
                if (sscanf(line, "Kd %f %f %f", &r, &g, &b) == 3) {
                    currentMaterial.diffuse = glm::vec3(r, g, b);
                }
            }
            else if (strncmp(line, "Ka ", 3) == 0) {
                // Ambient color
                float r, g, b;
                if (sscanf(line, "Ka %f %f %f", &r, &g, &b) == 3) {
                    currentMaterial.ambient = glm::vec3(r, g, b);
                }
            }
            else if (strncmp(line, "map_Kd ", 7) == 0) {
                // Diffuse texture map
                char texturePath[256];
                if (sscanf(line, "map_Kd %s", texturePath) == 1) {
                    currentMaterial.diffuseTexture = std::string(texturePath);
                }
            }
        }
        
        // Save last material
        if (hasMaterial) {
            // Resolve full texture path if texture exists
            if (!currentMaterial.diffuseTexture.empty()) {
                std::string fullTexturePath = currentMaterial.diffuseTexture;
                // Handle relative paths that start with "./"
                if (fullTexturePath.substr(0, 2) == "./") {
                    fullTexturePath = baseDirectory + fullTexturePath.substr(2);
                } else if (fullTexturePath[0] != '/' && fullTexturePath[0] != '\\' && fullTexturePath.find(':') == std::string::npos) {
                    // Relative path without "./"
                    fullTexturePath = baseDirectory + fullTexturePath;
                }
                currentMaterial.fullTexturePath = fullTexturePath;
                printf("Resolved texture path: '%s'\n", fullTexturePath.c_str());
                
                // For materials without explicit Kd, use white as fallback when there's a texture
                if (currentMaterial.diffuse == glm::vec3(1.0f, 0.0f, 0.0f)) {
                    currentMaterial.diffuse = glm::vec3(1.0f, 1.0f, 1.0f); // White - let texture provide color
                    printf("Material '%s' has texture but no Kd - using white fallback\n", currentMaterial.name.c_str());
                }
            } else if (currentMaterial.diffuse == glm::vec3(1.0f, 0.0f, 0.0f)) {
                // No texture and no explicit diffuse color - generate color from name
                currentMaterial.diffuse = generateColorFromName(currentMaterial.name);
            }
            materials.push_back(currentMaterial);
        }
        
        fclose(file);
        printf("Loaded %zu materials from %s\n", materials.size(), mtlPath.c_str());
        
        // Debug: print loaded materials
        for (const Material& mat : materials) {
            printf("  Material '%s': diffuse(%.2f, %.2f, %.2f), texture='%s', fullPath='%s'\n", 
                   mat.name.c_str(), mat.diffuse.r, mat.diffuse.g, mat.diffuse.b, 
                   mat.diffuseTexture.c_str(), mat.fullTexturePath.c_str());
        }
        
        return materials;
    }

public:
    static std::vector<Triangle> loadOBJ(const std::string& filepath, ITextureManager& textureManager) {
        std::vector<Triangle> triangles;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec2> texCoords;
        std::vector<Material> materials;
        
        // Current material being used
        glm::vec3 currentColor(1.0f, 0.0f, 0.0f); // Default red
        uint32_t currentTextureIndex = 0; // Current texture index (0 = no texture)
        
        FILE* file = fopen(filepath.c_str(), "r");
        if (!file) {
            printf("Failed to open OBJ file: %s\n", filepath.c_str());
            return triangles;
        }
        
        // Extract directory path for MTL file loading
        std::string directory = filepath;
        size_t lastSlash = directory.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            directory = directory.substr(0, lastSlash + 1);
        } else {
            directory = "";
        }
        
        char line[1024];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "v ", 2) == 0) {
                // Vertex position
                float x, y, z;
                sscanf(line, "v %f %f %f", &x, &y, &z);
                vertices.push_back(glm::vec3(x, y, z));
            }
            else if (strncmp(line, "vn ", 3) == 0) {
                // Vertex normal
                float x, y, z;
                sscanf(line, "vn %f %f %f", &x, &y, &z);
                normals.push_back(glm::vec3(x, y, z));
            }
            else if (strncmp(line, "vt ", 3) == 0) {
                // Texture coordinate
                float u, v;
                sscanf(line, "vt %f %f", &u, &v);
                texCoords.push_back(glm::vec2(u, v));
            }
            else if (strncmp(line, "mtllib ", 7) == 0) {
                // Material library
                char mtlFile[256];
                sscanf(line, "mtllib %s", mtlFile);
                std::string mtlPath = directory + std::string(mtlFile);
                materials = loadMTL(mtlPath, directory);
            }
            else if (strncmp(line, "usemtl ", 7) == 0) {
                // Use material
                char materialName[256];
                sscanf(line, "usemtl %s", materialName);
                
                // Find material by name
                bool found = false;
                for (const Material& mat : materials) {
                    if (mat.name == std::string(materialName)) {
                        currentColor = mat.diffuse;
                        currentTextureIndex = textureManager.getTextureIndex(mat.fullTexturePath);
                        printf("🎨 MATERIAL '%s': color(%.2f, %.2f, %.2f), textureIndex=%u, path='%s'\n", 
                               materialName, currentColor.r, currentColor.g, currentColor.b, currentTextureIndex, mat.fullTexturePath.c_str());
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    printf("Warning: Material '%s' not found, using default color\n", materialName);
                    currentTextureIndex = 0;
                }
            }
            else if (strncmp(line, "f ", 2) == 0) {
                // Face (triangle)
                int v1, v2, v3;
                int vt1, vt2, vt3;
                int vn1, vn2, vn3;
                
                // Try different face formats
                if (sscanf(line, "f %d/%d/%d %d/%d/%d %d/%d/%d", 
                          &v1, &vt1, &vn1, &v2, &vt2, &vn2, &v3, &vt3, &vn3) == 9) {
                    // Format: v/vt/vn
                } else if (sscanf(line, "f %d//%d %d//%d %d//%d", 
                                 &v1, &vn1, &v2, &vn2, &v3, &vn3) == 6) {
                    // Format: v//vn
                } else if (sscanf(line, "f %d/%d %d/%d %d/%d", 
                                 &v1, &vt1, &v2, &vt2, &v3, &vt3) == 6) {
                    // Format: v/vt
                } else if (sscanf(line, "f %d %d %d", &v1, &v2, &v3) == 3) {
                    // Format: v
                } else {
                    continue; // Skip unsupported face format
                }
                
                // Convert to 0-based indexing
                v1--; v2--; v3--;
                vt1--; vt2--; vt3--;
                
                if (v1 >= 0 && v1 < vertices.size() &&
                    v2 >= 0 && v2 < vertices.size() &&
                    v3 >= 0 && v3 < vertices.size()) {
                    Triangle tri;
                    tri.v0 = vertices[v1];
                    tri.v1 = vertices[v2];
                    tri.v2 = vertices[v3];
                    
                    // Set UV coordinates if available
                    if (vt1 >= 0 && vt1 < texCoords.size()) tri.uv0 = texCoords[vt1];
                    if (vt2 >= 0 && vt2 < texCoords.size()) tri.uv1 = texCoords[vt2];
                    if (vt3 >= 0 && vt3 < texCoords.size()) tri.uv2 = texCoords[vt3];
                    
                    tri.color = currentColor; // Assign current material color (fallback)
                    tri.textureIndex = currentTextureIndex; // Store texture index for sampling
                    
                    // Debug: Print first few triangles
                    static int triCount = 0;
                    if (triCount < 5) {
                        printf("🔺 TRIANGLE %d: textureIndex=%u, color(%.3f,%.3f,%.3f)\n", 
                               triCount, tri.textureIndex, tri.color.r, tri.color.g, tri.color.b);
                        triCount++;
                    }
                    
                    triangles.push_back(tri);
                }
            }
        }
        
        fclose(file);
        printf("Loaded %zu triangles from %s\n", triangles.size(), filepath.c_str());
        return triangles;
    }
};
