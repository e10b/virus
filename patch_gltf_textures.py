import sys

with open('example/gltfloader.h', 'r') as f:
    text = f.read()

text = text.replace(
    'static void processNode(const fastgltf::Asset& asset, size_t nodeIndex, const glm::mat4& parentTransform, std::vector<Triangle>& outTriangles) {',
    'static void processNode(const fastgltf::Asset& asset, size_t nodeIndex, const glm::mat4& parentTransform, std::vector<Triangle>& outTriangles, ITextureManager& textureManager) {'
)

old_mat = """                glm::vec3 matColor(0.8f);
                if (primitive.materialIndex.has_value()) {
                    auto& material = asset.materials[primitive.materialIndex.value()];
                    matColor = glm::vec3(
                        material.pbrData.baseColorFactor[0],
                        material.pbrData.baseColorFactor[1],
                        material.pbrData.baseColorFactor[2]
                    );
                }

                // Process triangles
                size_t numTriangles = indicesAccessor.count / 3;
                size_t outBase = outTriangles.size();
                outTriangles.resize(outBase + numTriangles);

                // Initialize all to standard empty state
                for (size_t i = 0; i < numTriangles; ++i) {
                    outTriangles[outBase + i].color = matColor;
                    outTriangles[outBase + i].textureIndex = 0;
                }"""

new_mat = """                uint32_t currentTextureIndex = 0;
                glm::vec3 matColor(0.8f);
                if (primitive.materialIndex.has_value()) {
                    auto& material = asset.materials[primitive.materialIndex.value()];
                    matColor = glm::vec3(
                        material.pbrData.baseColorFactor[0],
                        material.pbrData.baseColorFactor[1],
                        material.pbrData.baseColorFactor[2]
                    );
                    
                    if (material.pbrData.baseColorTexture.has_value()) {
                        size_t texIndex = material.pbrData.baseColorTexture.value().textureIndex;
                        auto& texture = asset.textures[texIndex];
                        if (texture.imageIndex.has_value()) {
                            auto& image = asset.images[texture.imageIndex.value()];
                            std::string texName = image.name.empty() ? "tex" + std::to_string(texIndex) : std::string(image.name);
                            if (auto* array = std::get_if<fastgltf::sources::Array>(&image.data)) {
                                currentTextureIndex = textureManager.getTextureFromMemory(
                                    reinterpret_cast<const unsigned char*>(array->bytes.data()), 
                                    array->bytes.size(), texName);
                            } else if (auto* bufferViewSource = std::get_if<fastgltf::sources::BufferView>(&image.data)) {
                                auto& bufferView = asset.bufferViews[bufferViewSource->bufferViewIndex];
                                auto& buffer = asset.buffers[bufferView.bufferIndex];
                                if (auto* bufferArray = std::get_if<fastgltf::sources::Array>(&buffer.data)) {
                                    currentTextureIndex = textureManager.getTextureFromMemory(
                                        reinterpret_cast<const unsigned char*>(bufferArray->bytes.data()) + bufferView.byteOffset, 
                                        bufferView.byteLength, texName);
                                }
                            }
                        }
                    }
                }

                // Process triangles
                size_t numTriangles = indicesAccessor.count / 3;
                size_t outBase = outTriangles.size();
                outTriangles.resize(outBase + numTriangles);

                // Initialize all to standard empty state
                for (size_t i = 0; i < numTriangles; ++i) {
                    outTriangles[outBase + i].color = matColor;
                    outTriangles[outBase + i].textureIndex = currentTextureIndex;
                }"""

text = text.replace(old_mat, new_mat)

text = text.replace(
    'processNode(asset, childIdx, transform, outTriangles);',
    'processNode(asset, childIdx, transform, outTriangles, textureManager);'
)

text = text.replace(
    'processNode(asset, nodeIndex, glm::mat4(1.0f), triangles);',
    'processNode(asset, nodeIndex, glm::mat4(1.0f), triangles, textureManager);'
)

with open('example/gltfloader.h', 'w') as f:
    f.write(text)
print("Patched GLTF loader textures correctly!")
