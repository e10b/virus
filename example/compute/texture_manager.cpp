#include "texture_manager.h"

#include <cmath>
#include <cstdio>

VoxelCompute::TextureData::TextureData() : data(nullptr), width(0), height(0), channels(0) {}

VoxelCompute::TextureData::~TextureData() {
    if (data) {
        stbi_image_free(data);
        data = nullptr;
    }
}

uint32_t VoxelCompute::TextureManager::getTextureFromMemory(const unsigned char* data, int length, const std::string& name) {
    auto it = pathToIndex.find(name);
    if (it != pathToIndex.end()) return it->second;

    auto texData = std::make_unique<TextureData>();
    texData->data = stbi_load_from_memory(data, length, &texData->width, &texData->height, &texData->channels, 3);
    if (!texData->data) {
        printf("❌ FAILED TO LOAD EMBEDDED TEXTURE: '%s'\n", name.c_str());
        return 0;
    }

    textureArray.push_back(std::move(texData));
    uint32_t newIndex = static_cast<uint32_t>(textureArray.size());
    pathToIndex[name] = newIndex;

    printf("✅ LOADED EMBEDDED TEXTURE: '%s' (%dx%d)\n", name.c_str(),
           textureArray.back()->width, textureArray.back()->height);
    return newIndex;
}

uint32_t VoxelCompute::TextureManager::getTextureIndex(const std::string& texturePath) {
    if (texturePath.empty()) {
        printf("⚪ EMPTY TEXTURE PATH -> index 0\n");
        return 0;
    }

    auto it = pathToIndex.find(texturePath);
    if (it != pathToIndex.end()) {
        printf("♻️  REUSING TEXTURE '%s' -> index %u\n", texturePath.c_str(), it->second);
        return it->second;
    }

    printf("🔄 LOADING NEW TEXTURE: '%s'\n", texturePath.c_str());
    auto texData = std::make_unique<TextureData>();
    texData->data = stbi_load(texturePath.c_str(), &texData->width, &texData->height, &texData->channels, 3);
    if (!texData->data) {
        printf("❌ FAILED TO LOAD TEXTURE: '%s'\n", texturePath.c_str());
        printf("   stbi error: %s\n", stbi_failure_reason());
        return 0;
    }

    textureArray.push_back(std::move(texData));
    uint32_t newIndex = static_cast<uint32_t>(textureArray.size());
    pathToIndex[texturePath] = newIndex;

    printf("✅ LOADED TEXTURE '%s' at index %u (%dx%d)\n",
           texturePath.c_str(), newIndex, textureArray[newIndex - 1]->width, textureArray[newIndex - 1]->height);

    return newIndex;
}

glm::vec3 VoxelCompute::TextureManager::sampleTexture(uint32_t textureIndex, const glm::vec2& uv) const {
    if (textureIndex == 0 || textureIndex > textureArray.size() || !textureArray[textureIndex - 1]->data) {
        return glm::vec3(-1.0f);
    }

    const TextureData* tex = textureArray[textureIndex - 1].get();

    float u = uv.x - std::floor(uv.x);
    float v = 1.0f - (uv.y - std::floor(uv.y));

    int x = int(u * tex->width) % tex->width;
    int y = int(v * tex->height) % tex->height;

    int index = (y * tex->width + x) * 3;
    glm::vec3 color(
        tex->data[index] / 255.0f,
        tex->data[index + 1] / 255.0f,
        tex->data[index + 2] / 255.0f);

    return color;
}

size_t VoxelCompute::TextureManager::getTextureCount() const {
    return textureArray.size();
}

const VoxelCompute::TextureData* VoxelCompute::TextureManager::getTextureData(uint32_t textureIndex) const {
    if (textureIndex == 0 || textureIndex > textureArray.size()) {
        return nullptr;
    }
    return textureArray[textureIndex - 1].get();
}

void VoxelCompute::TextureManager::printMemoryUsage() const {
    size_t totalMemory = 0;
    printf("🖼️  TEXTURE ARRAY STATUS:\n");
    for (size_t i = 0; i < textureArray.size(); ++i) {
        if (textureArray[i]->data) {
            size_t texMemory = textureArray[i]->width * textureArray[i]->height * 3;
            totalMemory += texMemory;
            printf("  [%zu->%zu] %dx%d (%.1f KB)\n", i + 1, i, textureArray[i]->width, textureArray[i]->height, texMemory / 1024.0f);
        } else {
            printf("  [%zu->%zu] INVALID (no data)\n", i + 1, i);
        }
    }
    printf("📊 TOTAL: %zu textures, %.2f MB\n",
           textureArray.size(), totalMemory / (1024.0f * 1024.0f));
}

wgfx::Texture VoxelCompute::createOwlGPUTexture() {
    std::string owlTexturePath = std::string(RESOURCE_DIR) + "/owl.jpg";

    printf("🦉 Loading owl texture for GPU: '%s'\n", owlTexturePath.c_str());
    wgfx::Texture owlTexture = wgfx::loadTexture(owlTexturePath.c_str());
    printf("✅ Owl texture loaded for GPU rendering\n");

    return owlTexture;
}

glm::vec3 VoxelCompute::sampleTexture(const std::string& texturePath, const glm::vec2& uv) {
    uint32_t textureIndex = textureManager.getTextureIndex(texturePath);
    return textureManager.sampleTexture(textureIndex, uv);
}

namespace compute_layers {

uint32_t getTextureIndex(TextureManager& manager, const std::string& texturePath) {
    return manager.getTextureIndex(texturePath);
}

glm::vec3 sampleTexture(const TextureManager& manager, uint32_t textureIndex, const glm::vec2& uv) {
    return manager.sampleTexture(textureIndex, uv);
}

} // namespace compute_layers
