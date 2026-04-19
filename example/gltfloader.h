#pragma once
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include "objloader.h"
#include <iostream>
#include <filesystem>
#include <variant>

class GLTFLoader {
private:
    static glm::mat4 getNodeTransform(const fastgltf::Node& node) {
        if (auto* matrix = std::get_if<fastgltf::math::fmat4x4>(&node.transform)) {
            glm::mat4 m;
            memcpy(glm::value_ptr(m), matrix->data(), 16 * sizeof(float));
            return m;
        }
        
        glm::mat4 m(1.0f);
        if (auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
            m = glm::translate(m, glm::vec3(trs->translation[0], trs->translation[1], trs->translation[2]));
            glm::quat q(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]); // w,x,y,z
            m *= glm::mat4_cast(q);
            m = glm::scale(m, glm::vec3(trs->scale[0], trs->scale[1], trs->scale[2]));
        }
        return m;
    }

    static void processNode(const fastgltf::Asset& asset, size_t nodeIndex, const glm::mat4& parentTransform, std::vector<Triangle>& outTriangles, ITextureManager& textureManager) {
        auto& node = asset.nodes[nodeIndex];
        glm::mat4 transform = parentTransform * getNodeTransform(node);

        if (node.meshIndex.has_value()) {
            auto& mesh = asset.meshes[node.meshIndex.value()];
            for (auto& primitive : mesh.primitives) {
                if (!primitive.indicesAccessor.has_value()) continue;

                auto posIt = primitive.findAttribute("POSITION");
                if (posIt == primitive.attributes.end()) continue;

                auto& posAccessor = asset.accessors[posIt->accessorIndex];
                auto& indicesAccessor = asset.accessors[primitive.indicesAccessor.value()];

                std::vector<glm::vec3> positions(posAccessor.count);
                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAccessor, [&](glm::vec3 pos, size_t idx) {
                    glm::vec4 wpos = transform * glm::vec4(pos, 1.0f);
                    positions[idx] = glm::vec3(wpos) / wpos.w;
                });

                std::vector<glm::vec2> uvs;
                auto uvIt = primitive.findAttribute("TEXCOORD_0");
                if (uvIt != primitive.attributes.end()) {
                    auto& uvAccessor = asset.accessors[uvIt->accessorIndex];
                    uvs.resize(uvAccessor.count);
                    fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, uvAccessor, [&](glm::vec2 uv, size_t idx) {
                        // Obj pipeline automatically flips V in voxel_volume.h, so we cancel this out by pre-flipping.
                        uvs[idx] = glm::vec2(uv.x, 1.0f - uv.y);
                    });
                }

                uint32_t currentTextureIndex = 0;
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
                }

                fastgltf::iterateAccessorWithIndex<uint32_t>(asset, indicesAccessor, [&](uint32_t vIdx, size_t iIdx) {
                    size_t triIdx = outBase + (iIdx / 3);
                    size_t vertIdx = iIdx % 3;

                    if (vertIdx == 0) outTriangles[triIdx].v0 = positions[vIdx];
                    else if (vertIdx == 1) outTriangles[triIdx].v1 = positions[vIdx];
                    else outTriangles[triIdx].v2 = positions[vIdx];

                    if (!uvs.empty()) {
                        if (vertIdx == 0) outTriangles[triIdx].uv0 = uvs[vIdx];
                        else if (vertIdx == 1) outTriangles[triIdx].uv1 = uvs[vIdx];
                        else outTriangles[triIdx].uv2 = uvs[vIdx];
                    }
                });
            }
        }

        for (auto childIdx : node.children) {
            processNode(asset, childIdx, transform, outTriangles, textureManager);
        }
    }

public:
    static std::vector<Triangle> loadGLTF(const std::string& filepath, ITextureManager& textureManager) {
        std::vector<Triangle> triangles;

        fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization);
        constexpr auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages;

        auto dataResult = fastgltf::GltfDataBuffer::FromPath(filepath);
        if (dataResult.error() != fastgltf::Error::None) {
            printf("Failed to load generic GLTF %s\n", filepath.c_str());
            return triangles;
        }

        fastgltf::GltfDataBuffer& buffer = dataResult.get();
        auto assetResult = parser.loadGltf(buffer, std::filesystem::path(filepath).parent_path(), options);
        if (assetResult.error() != fastgltf::Error::None) {
            printf("fastgltf parser err! Error: %s\n", fastgltf::getErrorMessage(assetResult.error()).data());
            return triangles;
        }

        auto& asset = assetResult.get();

        if (asset.scenes.empty()) return triangles;

        auto& scene = asset.scenes[asset.defaultScene.value_or(0)];
        for (auto nodeIndex : scene.nodeIndices) {
            processNode(asset, nodeIndex, glm::mat4(1.0f), triangles, textureManager);
        }

        printf("GLTF Load complete! Exported %zu triangles.\n", triangles.size());
        return triangles;
    }
};
