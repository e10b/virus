#pragma once

#include <glm/glm.hpp>
#include <string>

// Interface for texture management
class ITextureManager {
public:
    virtual ~ITextureManager() = default;
    virtual uint32_t getTextureIndex(const std::string& texturePath) = 0;
    virtual uint32_t getTextureFromMemory(const unsigned char* data, int length, const std::string& name) = 0;
    virtual glm::vec3 sampleTexture(uint32_t textureIndex, const glm::vec2& uv) const = 0;
};
