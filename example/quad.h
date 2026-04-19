#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "context.h"
#include "imgui.h"
#include "wgfx.h"

struct QuantumState {
    int n = 2;
    int l = 1;
    int m = 0;
    int sampleCount = 100000;

    void clamp() {
        n = std::clamp(n, 1, 30);
        l = std::clamp(l, 0, n - 1);
        m = std::clamp(m, -l, l);
    }
};

struct ClipState {
    glm::vec3 origin = glm::vec3(0.0f);
    int removedOctant = 1;
};

class OrbitCamera {
public:
    glm::vec3 target = glm::vec3(0.0f);
    float radius = 50.0f;
    float azimuth = 0.0f;
    float elevation = 3.14159265358979323846f / 2.0f;
    float orbitSpeed = 0.01f;
    float zoomSpeed = 10.0f;

    glm::vec3 position() const {
        float e = glm::clamp(elevation, 0.01f, 3.14159265358979323846f - 0.01f);
        return glm::vec3(
            radius * std::sin(e) * std::cos(azimuth),
            radius * std::cos(e),
            radius * std::sin(e) * std::sin(azimuth)
        );
    }

    void process(float dt, bool allowMouseCapture, float wheelDelta) {
        float mx = 0.0f;
        float my = 0.0f;

        if (allowMouseCapture) {
            Uint32 mask = SDL_GetMouseState(&mx, &my);
            bool draggingNow = (mask & SDL_BUTTON_LMASK) != 0 || (mask & SDL_BUTTON_MMASK) != 0;
            if (draggingNow) {
                if (!dragging_) {
                    dragging_ = true;
                    lastX_ = mx;
                    lastY_ = my;
                } else {
                    float dx = mx - lastX_;
                    float dy = my - lastY_;
                    azimuth += dx * orbitSpeed;
                    elevation -= dy * orbitSpeed;
                    elevation = glm::clamp(elevation, 0.01f, 3.14159265358979323846f - 0.01f);
                    lastX_ = mx;
                    lastY_ = my;
                }
            } else {
                dragging_ = false;
            }
        }

        if (wheelDelta != 0.0f) {
            radius -= wheelDelta * zoomSpeed;
        }

        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        if (keys[SDL_SCANCODE_EQUALS]) radius -= zoomSpeed * dt * 20.0f;
        if (keys[SDL_SCANCODE_MINUS]) radius += zoomSpeed * dt * 20.0f;
        radius = std::max(radius, 1.0f);
    }

private:
    bool dragging_ = false;
    float lastX_ = 0.0f;
    float lastY_ = 0.0f;
};

class Quad {
public:
    static Quad& Instance() {
        static Quad instance;
        return instance;
    }

    wgfx::Pipeline* pipeline = nullptr;

    void render(float dt) {
        processShortcuts();
        advanceSimulation(dt);

        ImGuiIO& io = ImGui::GetIO();
        float wheel = Context::Instance().consumeWheelDelta();
        camera_.process(dt, !io.WantCaptureMouse, wheel);

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 2000.0f);
        glm::mat4 view = glm::lookAt(camera_.position(), camera_.target, glm::vec3(0, 1, 0));
        gpuState_.viewProj = projection * view;

        gpuState_.clipOrigin = glm::vec4(clip_.origin, intensityRange_);
        gpuState_.quantum = glm::vec4(
            static_cast<float>(quantum_.n),
            static_cast<float>(quantum_.l),
            static_cast<float>(quantum_.m),
            static_cast<float>(colorMode_)
        );
        gpuState_.render = glm::vec4(
            static_cast<float>(clip_.removedOctant),
            intensityScale_,
            simulationTime_,
            sampleSeed_
        );

        pipeline->updateUniform(stateUniform_, reinterpret_cast<const float*>(&gpuState_));

        ibo_->indexCount = static_cast<uint32_t>(quantum_.sampleCount);
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    void drawImGuiPanel() {
        ImGui::Begin("Orbital Controls");
        ImGui::Text("GPU orbital evaluation (instant n/l/m updates)");

        int n = quantum_.n;
        int l = quantum_.l;
        int m = quantum_.m;
        int count = quantum_.sampleCount;
        int colorMode = colorMode_;
        int octant = clip_.removedOctant;
        float origin[3] = { clip_.origin.x, clip_.origin.y, clip_.origin.z };
        int oldN = quantum_.n;
        int oldL = quantum_.l;
        int oldM = quantum_.m;
        float flowSpeed = flowSpeed_;
        float intensityRange = intensityRange_;

        ImGui::SliderInt("n", &n, 1, 30);
        l = std::clamp(l, 0, n - 1);
        ImGui::SliderInt("l", &l, 0, std::max(0, n - 1));
        m = std::clamp(m, -l, l);
        ImGui::SliderInt("m", &m, -std::max(1, l), std::max(1, l));
        ImGui::SliderInt("samples", &count, 1000, kMaxParticles);

        ImGui::Combo("colorspace", &colorMode, "Inferno\0Magma\0Plasma\0Viridis\0Cividis\0Turbo\0Gray\0Fire\0Cyan-Magenta\0");

        ImGui::Separator();
        ImGui::Text("Selected octant is removed");
        ImGui::SliderFloat3("origin", origin, -200.0f, 200.0f, "%.2f");
        ImGui::SliderInt("removed octant", &octant, 1, 8);

        ImGui::SliderFloat("intensity scale", &intensityScale_, 0.1f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("intensity range", &intensityRange, 0.05f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("flow speed", &flowSpeed, 0.0f, 8.0f, "%.2f");

        quantum_.n = n;
        quantum_.l = l;
        quantum_.m = m;
        quantum_.sampleCount = std::clamp(count, 1000, kMaxParticles);
        quantum_.clamp();
        colorMode_ = std::clamp(colorMode, 0, 8);
        flowSpeed_ = std::max(0.0f, flowSpeed);
        intensityRange_ = std::clamp(intensityRange, 0.01f, 10.0f);

        if (oldN != quantum_.n || oldL != quantum_.l || oldM != quantum_.m) {
            sampleSeed_ += 1.0f;
        }

        clip_.origin = glm::vec3(origin[0], origin[1], origin[2]);
        clip_.removedOctant = std::clamp(octant, 1, 8);

        if (ImGui::Button("Reshuffle sample set")) {
            sampleSeed_ += 1.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset motion phase")) {
            simulationTime_ = 0.0f;
        }

        ImGui::Text("Default removes octant 1 at origin (0,0,0)");
        ImGui::End();
    }

private:
    struct alignas(16) GpuOrbitalState {
        glm::mat4 viewProj;
        glm::vec4 clipOrigin;
        glm::vec4 quantum;
        glm::vec4 render;
    };

    static constexpr int kMaxParticles = 250000;

    std::unique_ptr<wgfx::VertexBuffer> vbo_;
    std::unique_ptr<wgfx::IndexBuffer> ibo_;
    wgfx::Uniform* stateUniform_ = nullptr;

    OrbitCamera camera_;
    QuantumState quantum_;
    ClipState clip_;

    int colorMode_ = 0;
    float intensityScale_ = 1.0f;
    float intensityRange_ = 0.35f;
    float sampleSeed_ = 1.0f;
    float simulationTime_ = 0.0f;
    float flowSpeed_ = 1.0f;

    GpuOrbitalState gpuState_{};

    bool prevW_ = false;
    bool prevS_ = false;
    bool prevE_ = false;
    bool prevD_ = false;
    bool prevR_ = false;
    bool prevF_ = false;
    bool prevT_ = false;
    bool prevG_ = false;

    Quad() {
        pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "atoms_sphere.wgsl").c_str()));

        stateUniform_ = wgfx::createUniform(0, sizeof(GpuOrbitalState), reinterpret_cast<const float*>(&gpuState_));
        pipeline->setUniform(stateUniform_);

        pipeline->targets = 1;
        pipeline->useDepth = false;

        initBuffers();
        pipeline->init(vbo_.get());
    }

    Quad(const Quad&) = delete;
    void operator=(const Quad&) = delete;

    void initBuffers() {
        std::vector<float> points(static_cast<size_t>(kMaxParticles) * 3, 0.0f);

        std::vector<uint32_t> indices(static_cast<size_t>(kMaxParticles));
        for (uint32_t i = 0; i < static_cast<uint32_t>(kMaxParticles); ++i) {
            indices[i] = i;
        }

        vbo_.reset(wgfx::createVertexBuffer(points));
        vbo_->setTopology(PrimitiveTopology::PointList);
        vbo_->setAttribute(0, wgfx::vec3f, 0);

        ibo_.reset(wgfx::createIndexBuffer(indices));
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    void processShortcuts() {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) {
            return;
        }

        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        bool w = keys[SDL_SCANCODE_W];
        bool s = keys[SDL_SCANCODE_S];
        bool e = keys[SDL_SCANCODE_E];
        bool d = keys[SDL_SCANCODE_D];
        bool r = keys[SDL_SCANCODE_R];
        bool f = keys[SDL_SCANCODE_F];
        bool t = keys[SDL_SCANCODE_T];
        bool g = keys[SDL_SCANCODE_G];

        int oldN = quantum_.n;
        int oldL = quantum_.l;
        int oldM = quantum_.m;

        if (w && !prevW_) quantum_.n += 1;
        if (s && !prevS_) quantum_.n -= 1;
        if (e && !prevE_) quantum_.l += 1;
        if (d && !prevD_) quantum_.l -= 1;
        if (r && !prevR_) quantum_.m += 1;
        if (f && !prevF_) quantum_.m -= 1;
        if (t && !prevT_) quantum_.sampleCount += 10000;
        if (g && !prevG_) quantum_.sampleCount -= 10000;

        quantum_.sampleCount = std::clamp(quantum_.sampleCount, 1000, kMaxParticles);
        quantum_.clamp();

        if (oldN != quantum_.n || oldL != quantum_.l || oldM != quantum_.m) {
            sampleSeed_ += 1.0f;
        }

        prevW_ = w;
        prevS_ = s;
        prevE_ = e;
        prevD_ = d;
        prevR_ = r;
        prevF_ = f;
        prevT_ = t;
        prevG_ = g;
    }

    void advanceSimulation(float dt) {
        simulationTime_ += std::max(dt, 0.0f) * flowSpeed_;
    }
};
