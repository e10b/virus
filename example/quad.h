#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "context.h"
#include "imgui.h"
#include "slater_vmc.h"
#include "wgfx.h"

struct QuantumState {
    int n = 9;
    int l = 3;
    int m = 1;
    int sampleCount = 100000;

    void clamp() {
        n = std::clamp(n, 1, 30);
        l = std::clamp(l, 0, n - 1);
        if (l == 14) {
            l = 13;
        }
        m = std::clamp(m, -l, l);
    }
};

struct ClipState {
    glm::vec3 origin = glm::vec3(0.0f);
    int removedOctant = 1;
};

class OrbitCamera {
public:
    glm::vec3 target = glm::vec3(-40.050f, 1.792f, 21.270f);
    float radius = 548.800f;
    float azimuth = 1.051212f;   // 60.23 deg
    float elevation = 1.378810f; // 79.00 deg
    float orbitSpeed = 0.01f;
    float zoomSpeed = 10.0f;

    glm::vec3 position() const {
        float e = glm::clamp(elevation, 0.01f, 3.14159265358979323846f - 0.01f);
        return target + glm::vec3(
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

        const bool vmcActive = vmcMode_ && vmcConfigured_;

        if (vmcActive) {
            vmc_.setWalkerCount(vmcWalkerCount_);
            vmc_.setParameters(vmcStepSize_, vmcZetaScale_, vmcJastrowBeta_);
            vmc_.setMaxCloudPoints(static_cast<size_t>(quantum_.sampleCount));
            vmc_.step(vmcSweepsPerFrame_, vmcThermalizationSweeps_, vmcMeasureEvery_);
            rebuildVmcPointBuffer();
            updateCameraForVmc();
        } else {
            if (orbitalBufferDirty_) {
                rebuildOrbitalAttributeBuffer();
                orbitalBufferDirty_ = false;
            }
        }

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
            vmcActive ? -1.0f : static_cast<float>(quantum_.n),
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

        if (vmcActive) {
            ibo_->indexCount = static_cast<uint32_t>(std::clamp(vmcDrawCount_, 1, kMaxParticles));
        } else {
            ibo_->indexCount = static_cast<uint32_t>(quantum_.sampleCount);
        }
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    void drawImGuiPanel() {
        ImGui::Begin("Orbital Controls");
        ImGui::Text("GPU orbital evaluation (instant n/l/m updates)");

        bool applyConfigFromEnter = ImGui::InputText(
            "electron config",
            configInput_,
            sizeof(configInput_),
            ImGuiInputTextFlags_EnterReturnsTrue
        );
        ImGui::SameLine();
        if (ImGui::Button("Apply config") || applyConfigFromEnter) {
            applyElectronConfiguration(configInput_);
        }
        ImGui::SameLine();
        if (ImGui::Button("Single orbital mode")) {
            vmcMode_ = false;
            useConfiguration_ = false;
            configMessage_ = "Single-orbital mode active";
            configError_.clear();
            orbitalBufferDirty_ = true;
        }

        ImGui::TextWrapped("Examples: 1s^1 (Hydrogen), 1s^2 2s^2 2p^2 (Carbon)");
        if (!configMessage_.empty()) {
            ImGui::TextWrapped("%s", configMessage_.c_str());
        }
        if (!configError_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", configError_.c_str());
        }
        ImGui::Separator();

        ImGui::Checkbox("Many-body Slater VMC mode", &vmcMode_);
        if (vmcMode_ && !vmcConfigured_) {
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.25f, 1.0f), "VMC is enabled but not configured. Click Apply config first.");
        }
        ImGui::SliderInt("VMC sweeps/frame", &vmcSweepsPerFrame_, 1, 48);
        ImGui::SliderInt("VMC thermalization", &vmcThermalizationSweeps_, 0, 16);
        ImGui::SliderInt("VMC measure every", &vmcMeasureEvery_, 1, 8);
        ImGui::SliderInt("VMC walkers", &vmcWalkerCount_, 1, 512);
        ImGui::SliderFloat("VMC step size", &vmcStepSize_, 0.03f, 1.5f, "%.3f");
        ImGui::SliderFloat("VMC zeta scale", &vmcZetaScale_, 0.2f, 3.0f, "%.3f");
        ImGui::SliderFloat("Jastrow beta", &vmcJastrowBeta_, 0.0f, 1.5f, "%.3f");
        ImGui::Checkbox("VMC camera auto-center", &vmcAutoCenterCamera_);
        ImGui::SameLine();
        if (ImGui::Button("Center camera now")) {
            centerCameraOnVmcCloud(true);
        }
        if (ImGui::Button("Reset walkers")) {
            vmc_.resetWalkers(static_cast<uint32_t>(sampleSeed_ * 7919.0f) ^ 0xa341316cu);
            vmc_.clearPointCloud();
            sampleSeed_ += 1.0f;
            centerCameraOnVmcCloud(true);
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear VMC cloud")) {
            vmc_.clearPointCloud();
            vmcDrawCount_ = 0;
        }
        const SlaterVMC::Stats vmcStats = vmc_.stats();
        ImGui::Text("VMC active walkers: %d", vmc_.walkerCount());
        ImGui::Text("VMC acceptance: %.2f%%", vmcStats.acceptance * 100.0);
        ImGui::Text("VMC <E_L>: %.6f Ha (%d measurements)", vmcStats.avgEnergy, vmcStats.measurements);
        ImGui::Separator();

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

        constexpr int kMaxN = 30;
        ImGui::SliderInt("n", &n, 1, kMaxN);
        ImGui::SliderInt("l", &l, 0, kMaxN - 1);
        if (l >= n) {
            n = std::min(kMaxN, l + 1);
            l = std::min(l, n - 1);
        }
        m = std::clamp(m, -l, l);
        ImGui::SliderInt("m", &m, -std::max(1, l), std::max(1, l));
        ImGui::SliderInt("samples", &count, 1000, kMaxParticles);
        ImGui::Text("Constraint: l < n (n auto-increases when needed)");

        ImGui::Combo("colorspace", &colorMode, "Inferno\0Magma\0Plasma\0Viridis\0Cividis\0Turbo\0Gray\0Fire\0Cyan-Magenta\0Phase Velocity\0Stationary Phase\0");
        if (colorMode == 9) {
            ImGui::TextWrapped("Phase Velocity: Colors cycle through the spectrum; faster local color cycling indicates higher local momentum/energy.");
        }
        if (colorMode == 10) {
            ImGui::TextWrapped("Stationary Phase: The orbital phase rotates coherently over time (global e^{-iEt/hbar} style evolution).");
        }

        ImGui::Separator();
        ImGui::Text("Selected octant is removed");
        ImGui::SliderFloat3("origin", origin, -200.0f, 200.0f, "%.2f");
        ImGui::SliderInt("removed octant", &octant, 1, 8);

        ImGui::SliderFloat("intensity scale", &intensityScale_, 0.1f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("intensity range", &intensityRange, 0.05f, 50.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("flow speed", &flowSpeed, 0.0f, 8.0f, "%.2f");

        quantum_.n = n;
        quantum_.l = l;
        quantum_.m = m;
        quantum_.sampleCount = std::clamp(count, 1000, kMaxParticles);
        quantum_.clamp();
        colorMode_ = std::clamp(colorMode, 0, 10);
        flowSpeed_ = std::max(0.0f, flowSpeed);
        intensityRange_ = std::clamp(intensityRange, 0.01f, 50.0f);
        vmc_.setMaxCloudPoints(static_cast<size_t>(quantum_.sampleCount));

        if (!useConfiguration_) {
            orbitalBufferDirty_ = true;
        }

        if (oldN != quantum_.n || oldL != quantum_.l || oldM != quantum_.m) {
            sampleSeed_ += 1.0f;
            orbitalBufferDirty_ = true;
        }

        clip_.origin = glm::vec3(origin[0], origin[1], origin[2]);
        clip_.removedOctant = std::clamp(octant, 1, 8);

        if (ImGui::Button("Reshuffle sample set")) {
            sampleSeed_ += 1.0f;
            orbitalBufferDirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset motion phase")) {
            simulationTime_ = 0.0f;
        }

        if (ImGui::CollapsingHeader("Debug Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            const glm::vec3 camPos = camera_.position();
            const float azimuthDeg = camera_.azimuth * 57.29577951308232f;
            const float elevationDeg = camera_.elevation * 57.29577951308232f;
            ImGui::Text("position: (%.3f, %.3f, %.3f)", camPos.x, camPos.y, camPos.z);
            ImGui::Text("target:   (%.3f, %.3f, %.3f)", camera_.target.x, camera_.target.y, camera_.target.z);
            ImGui::Text("rotation: azimuth %.2f deg, elevation %.2f deg", azimuthDeg, elevationDeg);
            ImGui::Text("radius: %.3f", camera_.radius);
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

    int colorMode_ = 3;
    float intensityScale_ = 18.44f;
    float intensityRange_ = 0.337f;
    float sampleSeed_ = 1.0f;
    float simulationTime_ = 0.0f;
    float flowSpeed_ = 8.0f;

    bool useConfiguration_ = false;
    bool orbitalBufferDirty_ = true;
    std::vector<SpinOrbitalState> electronOrbitals_;
    std::vector<float> orbitalAttribData_;
    std::vector<float> vmcUploadData_;
    char configInput_[256] = "1s^1";
    std::string configMessage_;
    std::string configError_;

    SlaterVMC vmc_;
    bool vmcMode_ = true;
    bool vmcConfigured_ = false;
    int vmcSweepsPerFrame_ = 12;
    int vmcThermalizationSweeps_ = 2;
    int vmcMeasureEvery_ = 2;
    int vmcWalkerCount_ = 64;
    float vmcStepSize_ = 0.35f;
    float vmcZetaScale_ = 1.0f;
    float vmcJastrowBeta_ = 0.2f;
    int vmcDrawCount_ = 0;
    bool vmcAutoCenterCamera_ = true;

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
        orbitalAttribData_.assign(static_cast<size_t>(kMaxParticles) * 3, 0.0f);
        fillSingleOrbitalAttribData();

        std::vector<uint32_t> indices(static_cast<size_t>(kMaxParticles));
        for (uint32_t i = 0; i < static_cast<uint32_t>(kMaxParticles); ++i) {
            indices[i] = i;
        }

        vbo_.reset(wgfx::createVertexBuffer(orbitalAttribData_));
        vbo_->setTopology(PrimitiveTopology::PointList);
        vbo_->setAttribute(0, wgfx::vec3f, 0);

        ibo_.reset(wgfx::createIndexBuffer(indices));
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    static int orbitalLetterToL(char c) {
        static const std::string letters = "spdfghiklmnoqrtuvwxyz";
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const size_t pos = letters.find(c);
        if (pos == std::string::npos) {
            return -1;
        }
        return static_cast<int>(pos);
    }

    static const char* elementNameFromZ(int z) {
        static constexpr std::array<const char*, 31> kNames = {
            "Unknown",
            "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron",
            "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon",
            "Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus",
            "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
            "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese",
            "Iron", "Cobalt", "Nickel", "Copper", "Zinc"
        };
        if (z >= 1 && z < static_cast<int>(kNames.size())) {
            return kNames[static_cast<size_t>(z)];
        }
        return "Custom atom";
    }

    bool parseElectronConfiguration(const std::string& config, std::vector<SpinOrbitalState>& outStates, std::string& err) const {
        outStates.clear();
        err.clear();

        std::stringstream ss(config);
        std::string token;
        while (ss >> token) {
            size_t i = 0;
            while (i < token.size() && std::isdigit(static_cast<unsigned char>(token[i])) != 0) {
                ++i;
            }
            if (i == 0 || i >= token.size()) {
                err = "Invalid token: '" + token + "'";
                return false;
            }

            int n = std::stoi(token.substr(0, i));
            int l = orbitalLetterToL(token[i]);
            if (l < 0) {
                err = "Unknown orbital letter in token: '" + token + "'";
                return false;
            }
            ++i;

            if (n < 1 || n > 30) {
                err = "n must be in [1, 30] in token: '" + token + "'";
                return false;
            }
            if (l >= n) {
                err = "Need l < n in token: '" + token + "'";
                return false;
            }
            if (l > 13) {
                err = "l > 13 is not supported in this renderer yet";
                return false;
            }

            if (i < token.size() && token[i] == '^') {
                ++i;
            }

            int occupancy = 1;
            if (i < token.size()) {
                const std::string occText = token.substr(i);
                if (occText.empty() ||
                    std::all_of(occText.begin(), occText.end(), [](char ch) {
                        return std::isdigit(static_cast<unsigned char>(ch)) != 0;
                    }) == false) {
                    err = "Invalid occupancy in token: '" + token + "'";
                    return false;
                }
                occupancy = std::stoi(occText);
            }

            const int capacity = 2 * (2 * l + 1);
            if (occupancy < 1 || occupancy > capacity) {
                err = "Occupancy exceeds shell capacity in token: '" + token + "'";
                return false;
            }

            int assigned = 0;
            for (int m = -l; m <= l && assigned < occupancy; ++m) {
                outStates.push_back({ n, l, m, +1 });
                ++assigned;
            }
            for (int m = -l; m <= l && assigned < occupancy; ++m) {
                outStates.push_back({ n, l, m, -1 });
                ++assigned;
            }
        }

        if (outStates.empty()) {
            err = "Configuration is empty";
            return false;
        }
        return true;
    }

    void fillSingleOrbitalAttribData() {
        for (int i = 0; i < kMaxParticles; ++i) {
            const size_t base = static_cast<size_t>(i) * 3;
            orbitalAttribData_[base + 0] = static_cast<float>(quantum_.n);
            orbitalAttribData_[base + 1] = static_cast<float>(quantum_.l);
            orbitalAttribData_[base + 2] = static_cast<float>(quantum_.m);
        }
    }

    void rebuildOrbitalAttributeBuffer() {
        if (!vbo_) {
            return;
        }

        if (!useConfiguration_ || electronOrbitals_.empty()) {
            fillSingleOrbitalAttribData();
            vbo_->write(orbitalAttribData_);
            return;
        }

        const int sampleCount = std::clamp(quantum_.sampleCount, 1000, kMaxParticles);
        std::vector<int> map(static_cast<size_t>(sampleCount));
        for (int i = 0; i < sampleCount; ++i) {
            map[static_cast<size_t>(i)] = i % static_cast<int>(electronOrbitals_.size());
        }

        std::mt19937 rng(static_cast<uint32_t>(sampleSeed_ * 997.0f) ^ 0x9e3779b9u);
        std::shuffle(map.begin(), map.end(), rng);

        for (int i = 0; i < kMaxParticles; ++i) {
            int orbitalIndex = 0;
            if (i < sampleCount) {
                orbitalIndex = map[static_cast<size_t>(i)];
            }
            const SpinOrbitalState& st = electronOrbitals_[static_cast<size_t>(orbitalIndex)];
            const size_t base = static_cast<size_t>(i) * 3;
            orbitalAttribData_[base + 0] = static_cast<float>(st.n);
            orbitalAttribData_[base + 1] = static_cast<float>(st.l);
            orbitalAttribData_[base + 2] = static_cast<float>(st.m);
        }

        vbo_->write(orbitalAttribData_);
    }

    void rebuildVmcPointBuffer() {
        if (!vbo_) {
            return;
        }

        const std::vector<glm::vec3>& cloud = vmc_.pointCloud();
        const std::vector<glm::vec3>& source = cloud.empty() ? vmc_.positions() : cloud;
        vmcDrawCount_ = std::min<int>(static_cast<int>(source.size()), kMaxParticles);

        if (vmcDrawCount_ <= 0) {
            vmcUploadData_.assign({ 0.0f, 0.0f, 0.0f });
            vmcDrawCount_ = 1;
            vbo_->write(vmcUploadData_);
            return;
        }

        vmcUploadData_.resize(static_cast<size_t>(vmcDrawCount_) * 3);
        for (int i = 0; i < vmcDrawCount_; ++i) {
            const glm::vec3& p = source[static_cast<size_t>(i)];
            const size_t base = static_cast<size_t>(i) * 3;
            vmcUploadData_[base + 0] = p.x;
            vmcUploadData_[base + 1] = p.y;
            vmcUploadData_[base + 2] = p.z;
        }

        vbo_->write(vmcUploadData_);
    }

    void centerCameraOnVmcCloud(bool immediate) {
        const std::vector<glm::vec3>& cloud = vmc_.pointCloud();
        const std::vector<glm::vec3>& source = cloud.empty() ? vmc_.positions() : cloud;
        if (source.empty()) {
            return;
        }

        const size_t maxSamples = std::min<size_t>(source.size(), 4000);
        const size_t stride = std::max<size_t>(1, source.size() / maxSamples);

        glm::vec3 centroid(0.0f);
        size_t count = 0;
        for (size_t i = 0; i < source.size(); i += stride) {
            centroid += source[i];
            ++count;
            if (count >= maxSamples) {
                break;
            }
        }
        if (count == 0) {
            return;
        }
        centroid /= static_cast<float>(count);

        float maxDist = 0.0f;
        for (size_t i = 0; i < source.size(); i += stride) {
            maxDist = std::max(maxDist, glm::length(source[i] - centroid));
        }

        const float desiredRadius = std::clamp(maxDist * 4.0f + 8.0f, 8.0f, 320.0f);
        const float blend = immediate ? 1.0f : 0.08f;
        camera_.target = glm::mix(camera_.target, centroid, blend);
        camera_.radius = glm::mix(camera_.radius, desiredRadius, blend);
    }

    void updateCameraForVmc() {
        if (!vmcAutoCenterCamera_) {
            return;
        }
        centerCameraOnVmcCloud(false);
    }

    void applyElectronConfiguration(const std::string& text) {
        std::vector<SpinOrbitalState> parsed;
        std::string parseError;
        if (!parseElectronConfiguration(text, parsed, parseError)) {
            configError_ = parseError;
            configMessage_.clear();
            return;
        }

        electronOrbitals_ = std::move(parsed);
        useConfiguration_ = true;
        configError_.clear();

        const int z = static_cast<int>(electronOrbitals_.size());
        configMessage_ = "Loaded " + std::to_string(z) + " electron(s): " + elementNameFromZ(z);

        vmc_.setNucleusCharge(z);
        vmc_.setWalkerCount(vmcWalkerCount_);
        vmc_.setParameters(vmcStepSize_, vmcZetaScale_, vmcJastrowBeta_);
        std::string vmcErr;
        if (!vmc_.configure(electronOrbitals_, vmcErr)) {
            configError_ = "VMC setup failed: " + vmcErr;
            vmcConfigured_ = false;
            return;
        }
        vmcConfigured_ = true;
        vmcMode_ = true;
        vmc_.setMaxCloudPoints(static_cast<size_t>(quantum_.sampleCount));
        vmc_.resetWalkers(static_cast<uint32_t>(sampleSeed_ * 7919.0f) ^ 0x9e3779b9u);
        centerCameraOnVmcCloud(true);

        quantum_.n = electronOrbitals_.front().n;
        quantum_.l = electronOrbitals_.front().l;
        quantum_.m = electronOrbitals_.front().m;
        quantum_.clamp();

        sampleSeed_ += 1.0f;
        orbitalBufferDirty_ = true;
    }

    void processShortcuts() {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) {
            return;
        }

        if (vmcMode_) {
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
