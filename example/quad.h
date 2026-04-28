#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
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
        if (renderPath_ == RenderPath::Path2D) {
            render2d(dt);
            return;
        }

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
        int pathIndex = static_cast<int>(renderPath_);
        if (ImGui::Combo("path", &pathIndex, "orbital\0"
                                               "2d\0")) {
            renderPath_ = static_cast<RenderPath>(std::clamp(pathIndex, 0, 1));
            if (renderPath_ == RenderPath::Path2D) {
                pipeline = pipeline2d_;
                pipeline->setVertexBuffer(vbo2d_.get());
                pipeline->setIndexBuffer(ibo2d_.get());
            } else {
                pipeline = pipelineOrbital_;
                pipeline->setVertexBuffer(vbo_.get());
                pipeline->setIndexBuffer(ibo_.get());
            }
        }

        if (renderPath_ == RenderPath::Path2D) {
            ImGui::Text("2D Hydrogen orbital visualizer");
            int n = quantum_.n;
            int l = quantum_.l;
            int m = quantum_.m;
            int colorMode = colorMode_;
            int twoDMode = twoDUseTdse_ ? 1 : 0;
            int potentialType = static_cast<int>(tdsePotentialType_);
            int gridSize = tdseGridSize_;
            const int oldN = quantum_.n;
            const int oldL = quantum_.l;
            const int oldM = quantum_.m;
            const int oldGrid = tdseGridSize_;
            const Potential2dType oldPotentialType = tdsePotentialType_;
            const float oldPotentialStrength = tdsePotentialStrength_;
            const float oldSquareHalfWidth = tdseSquareHalfWidth_;
            const float oldDomain = tdseDomainHalfExtent_;
            const float oldBigRadius = tdseCircleRadiusBig_;
            const float oldSmallRadius = tdseCircleRadiusSmall_;
            const float oldAnnIn = tdseAnnulusInner_;
            const float oldAnnOut = tdseAnnulusOuter_;
            const float oldBarrierHalfWidth = tdseBarrierHalfWidth_;
            const float oldSlitCenter = tdseSlitCenterOffset_;
            const float oldSlitHalfH = tdseSlitHalfHeight_;
            const glm::vec2 oldPacketCenter = tdsePacketCenter_;
            const glm::vec2 oldPacketMomentum = tdsePacketMomentum_;
            const bool oldUseAbsorbing = tdseUseAbsorbingBoundary_;
            const float oldAbsorbWidth = tdseAbsorbWidth_;
            const float oldAbsorbStrength = tdseAbsorbStrength_;
            constexpr int kMaxN = 30;
            ImGui::Combo("2d mode", &twoDMode, "Analytic orbital\0TDSE FDTD\0");
            ImGui::SliderInt("n", &n, 1, kMaxN);
            ImGui::SliderInt("l", &l, 0, kMaxN - 1);
            if (l >= n) {
                n = std::min(kMaxN, l + 1);
                l = std::min(l, n - 1);
            }
            m = std::clamp(m, -l, l);
            ImGui::SliderInt("m", &m, -std::max(1, l), std::max(1, l));
            ImGui::Text("Constraint: l < n (n auto-increases when needed)");
            ImGui::Combo("colorspace", &colorMode, "Inferno\0Magma\0Plasma\0Viridis\0Cividis\0Turbo\0Gray\0Fire\0Cyan-Magenta\0Phase Velocity\0Stationary Phase\0");
            ImGui::SliderFloat("2d zoom", &twoDZoom_, 1e-6f, 1e6f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d thickness", &twoDThickness_, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d slice z", &twoDSliceZ_, -8.0f, 8.0f, "%.2f");
            ImGui::Checkbox("2d integrate depth", &twoDIntegrateDepth_);
            ImGui::SliderFloat("2d intensity scale", &intensityScale_, 0.1f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d intensity range", &intensityRange_, 0.05f, 50.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d phase speed", &twoDPhaseSpeed_, 0.0f, 8.0f, "%.2f");

            twoDUseTdse_ = (twoDMode == 1);
            if (twoDUseTdse_) {
                ImGui::Separator();
                ImGui::Text("Time-dependent Schrodinger (explicit FDTD)");
                ImGui::Text("Use WASD to move the packet center");
                ImGui::Combo("potential", &potentialType, "Square well\0Circle well (big)\0Circle well (small)\0Double slit well\0Big circle small circle (annulus)\0");
                ImGui::SliderFloat("potential strength", &tdsePotentialStrength_, 0.0f, 16.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("square half-width", &tdseSquareHalfWidth_, 0.2f, 10.0f, "%.2f");
                ImGui::SliderFloat("big circle radius", &tdseCircleRadiusBig_, 0.3f, 20.0f, "%.2f");
                ImGui::SliderFloat("small circle radius", &tdseCircleRadiusSmall_, 0.2f, 10.0f, "%.2f");
                ImGui::SliderFloat("annulus inner radius", &tdseAnnulusInner_, 0.2f, 12.0f, "%.2f");
                ImGui::SliderFloat("annulus outer radius", &tdseAnnulusOuter_, 0.3f, 20.0f, "%.2f");
                ImGui::SliderFloat("double-slit barrier half-width", &tdseBarrierHalfWidth_, 0.05f, 1.2f, "%.3f");
                ImGui::SliderFloat("double-slit offset", &tdseSlitCenterOffset_, 0.0f, 6.0f, "%.2f");
                ImGui::SliderFloat("double-slit half-height", &tdseSlitHalfHeight_, 0.05f, 2.5f, "%.2f");
                ImGui::SliderFloat("domain half-size", &tdseDomainHalfExtent_, 4.0f, 40.0f, "%.2f");
                ImGui::SliderInt("fdtd grid", &gridSize, 64, 320);
                ImGui::Combo("integrator", &tdseIntegrator_, "Euler\0Crank-Nicolson\0");
                ImGui::SliderInt("substeps/frame", &tdseSubstepsPerFrame_, 1, 32);
                ImGui::SliderFloat("dt", &tdseDt_, 1e-6f, 2e-3f, "%.6f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat2("packet center", glm::value_ptr(tdsePacketCenter_), -40.0f, 40.0f, "%.2f");
                ImGui::SliderFloat2("packet momentum", glm::value_ptr(tdsePacketMomentum_), -8.0f, 8.0f, "%.2f");
                ImGui::SliderFloat("WASD move speed", &tdsePacketMoveSpeed_, 0.5f, 25.0f, "%.2f");
                ImGui::Checkbox("absorbing boundary", &tdseUseAbsorbingBoundary_);
                if (tdseUseAbsorbingBoundary_) {
                    ImGui::SliderFloat("absorb width", &tdseAbsorbWidth_, 0.4f, 16.0f, "%.2f");
                    ImGui::SliderFloat("absorb strength", &tdseAbsorbStrength_, 0.5f, 40.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                }
                ImGui::Checkbox("overlay potential", &tdseOverlayPotential_);
                if (ImGui::Button("Reset TDSE wavefunction")) {
                    tdseWaveNeedsReset_ = true;
                }
                tdsePotentialType_ = static_cast<Potential2dType>(std::clamp(potentialType, 0, 4));
                tdseGridSize_ = std::clamp(gridSize, 64, 320);
                if (tdseGridSize_ != oldGrid) {
                    resizeTdseBuffers();
                }
                if (tdsePotentialType_ != oldPotentialType ||
                    tdsePotentialStrength_ != oldPotentialStrength ||
                    tdseSquareHalfWidth_ != oldSquareHalfWidth ||
                    tdseDomainHalfExtent_ != oldDomain ||
                    tdseCircleRadiusBig_ != oldBigRadius ||
                    tdseCircleRadiusSmall_ != oldSmallRadius ||
                    tdseAnnulusInner_ != oldAnnIn ||
                    tdseAnnulusOuter_ != oldAnnOut ||
                    tdseBarrierHalfWidth_ != oldBarrierHalfWidth ||
                    tdseSlitCenterOffset_ != oldSlitCenter ||
                    tdseSlitHalfHeight_ != oldSlitHalfH) {
                    tdsePotentialDirty_ = true;
                    tdseWaveNeedsReset_ = true;
                }
                if (tdsePacketCenter_ != oldPacketCenter || tdsePacketMomentum_ != oldPacketMomentum) {
                    tdseWaveNeedsReset_ = true;
                }
                if (tdseUseAbsorbingBoundary_ != oldUseAbsorbing ||
                    tdseAbsorbWidth_ != oldAbsorbWidth ||
                    tdseAbsorbStrength_ != oldAbsorbStrength) {
                    tdseWaveNeedsReset_ = true;
                }
            }

            quantum_.n = n;
            quantum_.l = l;
            quantum_.m = m;
            quantum_.clamp();
            colorMode_ = std::clamp(colorMode, 0, 10);
            if (oldN != quantum_.n || oldL != quantum_.l || oldM != quantum_.m) {
                tdseWaveNeedsReset_ = true;
            }
            ImGui::End();
            return;
        }

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
    static constexpr float kPi = 3.14159265358979323846f;
    static constexpr int kMaxTdseGrid = 320;

    enum class Potential2dType : int {
        SquareWell = 0,
        CircleWellBig = 1,
        CircleWellSmall = 2,
        DoubleSlitWell = 3,
        AnnularWell = 4
    };

    enum class RenderPath : int {
        Orbital = 0,
        Path2D = 1
    };

    struct alignas(16) GpuOrbitalState {
        glm::mat4 viewProj;
        glm::vec4 clipOrigin;
        glm::vec4 quantum;
        glm::vec4 render;
    };
    struct alignas(16) Gpu2dState {
        glm::vec4 orbital; // x:n, y:l, z:m, w:colorMode
        glm::vec4 tuning;  // x:intensityScale, y:intensityRange, z:zoom, w:thickness
        glm::vec4 render;  // x:time, y:aspect, z:phaseSpeed, w:mode(0:analytic,1:tdse)
        glm::vec4 pan;     // x:panX, y:panY, z:sliceZ, w:integrateDepth(0/1)
        glm::vec4 tdse;    // x:gridSize, y:domainHalfExtent, z:potentialOverlay, w:reserved
    };

    static constexpr int kMaxParticles = 250000;

    std::unique_ptr<wgfx::VertexBuffer> vbo_;
    std::unique_ptr<wgfx::IndexBuffer> ibo_;
    wgfx::Uniform* stateUniform_ = nullptr;
    std::unique_ptr<wgfx::VertexBuffer> vbo2d_;
    std::unique_ptr<wgfx::IndexBuffer> ibo2d_;
    wgfx::Uniform* stateUniform2d_ = nullptr;
    wgfx::Uniform* tdseStorage_ = nullptr;
    wgfx::Pipeline* pipelineOrbital_ = nullptr;
    wgfx::Pipeline* pipeline2d_ = nullptr;
    RenderPath renderPath_ = RenderPath::Orbital;

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
    Gpu2dState gpu2dState_{};
    float twoDTime_ = 0.0f;
    float twoDZoom_ = 1.0f;
    float twoDThickness_ = 3.2f;
    float twoDPhaseSpeed_ = 1.0f;
    float twoDSliceZ_ = 0.0f;
    bool twoDIntegrateDepth_ = false;
    glm::vec2 twoDPan_ = glm::vec2(0.0f);
    bool twoDDragging_ = false;
    glm::vec2 twoDLastMouse_ = glm::vec2(0.0f);

    bool twoDUseTdse_ = false;
    int tdseGridSize_ = 192;
    float tdseDomainHalfExtent_ = 22.0f;
    float tdseDt_ = 0.00018f;
    int tdseSubstepsPerFrame_ = 8;
    int tdseIntegrator_ = 1;
    Potential2dType tdsePotentialType_ = Potential2dType::SquareWell;
    float tdsePotentialStrength_ = 4.0f;
    float tdseSquareHalfWidth_ = 3.2f;
    float tdseCircleRadiusBig_ = 6.0f;
    float tdseCircleRadiusSmall_ = 2.6f;
    float tdseAnnulusInner_ = 2.4f;
    float tdseAnnulusOuter_ = 6.8f;
    float tdseBarrierHalfWidth_ = 0.32f;
    float tdseSlitCenterOffset_ = 1.7f;
    float tdseSlitHalfHeight_ = 0.55f;
    glm::vec2 tdsePacketCenter_ = glm::vec2(-4.8f, 0.0f);
    glm::vec2 tdsePacketMomentum_ = glm::vec2(3.2f, 0.0f);
    float tdsePacketMoveSpeed_ = 8.0f;
    bool tdseUseAbsorbingBoundary_ = true;
    float tdseAbsorbWidth_ = 5.0f;
    float tdseAbsorbStrength_ = 14.0f;
    bool tdseOverlayPotential_ = false;
    bool tdsePotentialDirty_ = true;
    bool tdseWaveNeedsReset_ = true;
    int tdseNormCounter_ = 0;

    std::vector<float> tdseReal_;
    std::vector<float> tdseImag_;
    std::vector<float> tdseNextReal_;
    std::vector<float> tdseNextImag_;
    std::vector<float> tdseRhsReal_;
    std::vector<float> tdseRhsImag_;
    std::vector<float> tdsePotential_;
    std::vector<float> tdseUpload_;

    bool prevW_ = false;
    bool prevS_ = false;
    bool prevE_ = false;
    bool prevD_ = false;
    bool prevR_ = false;
    bool prevF_ = false;
    bool prevT_ = false;
    bool prevG_ = false;

    Quad() {
        pipelineOrbital_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "atoms_sphere.wgsl").c_str()));
        pipeline = pipelineOrbital_;

        stateUniform_ = wgfx::createUniform(0, sizeof(GpuOrbitalState), reinterpret_cast<const float*>(&gpuState_));
        pipelineOrbital_->setUniform(stateUniform_);

        pipelineOrbital_->targets = 1;
        pipelineOrbital_->useDepth = false;

        initBuffers();
        pipelineOrbital_->init(vbo_.get());

        pipeline2d_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "circle_2d.wgsl").c_str()));
        stateUniform2d_ = wgfx::createUniform(0, sizeof(Gpu2dState), reinterpret_cast<const float*>(&gpu2dState_));
        pipeline2d_->setUniform(stateUniform2d_);
        resizeTdseBuffers();
        tdseStorage_ = wgfx::createStorage(
            1,
            static_cast<size_t>(kMaxTdseGrid) * static_cast<size_t>(kMaxTdseGrid) * 4 * sizeof(float),
            nullptr,
            true);
        pipeline2d_->uniforms.setStorage(tdseStorage_);
        pipeline2d_->targets = 1;
        pipeline2d_->useDepth = false;
        init2dBuffers();
        pipeline2d_->init(vbo2d_.get());
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

        ibo_.reset(wgfx::createIndexBuffer(std::vector<uint32_t>(indices)));
        pipelineOrbital_->setVertexBuffer(vbo_.get());
        pipelineOrbital_->setIndexBuffer(ibo_.get());
    }

    void init2dBuffers() {
        const std::vector<float> vertices = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        const std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };

        vbo2d_.reset(wgfx::createVertexBuffer(vertices));
        vbo2d_->setTopology(PrimitiveTopology::TriangleList);
        vbo2d_->setAttribute(0, wgfx::vec3f, 0);

        ibo2d_.reset(wgfx::createIndexBuffer(indices));
        pipeline2d_->setVertexBuffer(vbo2d_.get());
        pipeline2d_->setIndexBuffer(ibo2d_.get());
    }

    static float factorialIntCpu(int v) {
        if (v <= 1) {
            return 1.0f;
        }
        float out = 1.0f;
        for (int i = 2; i <= v; ++i) {
            out *= static_cast<float>(i);
        }
        return out;
    }

    static float associatedLaguerreCpu(int k, int alpha, float x) {
        if (k <= 0) {
            return 1.0f;
        }
        float lm2 = 1.0f;
        float lm1 = 1.0f + static_cast<float>(alpha) - x;
        if (k == 1) {
            return lm1;
        }
        float l = lm1;
        for (int j = 2; j <= k; ++j) {
            l = ((static_cast<float>(2 * j - 1 + alpha) - x) * lm1 - static_cast<float>(j - 1 + alpha) * lm2) / static_cast<float>(j);
            lm2 = lm1;
            lm1 = l;
        }
        return l;
    }

    static float associatedLegendreCpu(int l, int mAbs, float x) {
        float pmm = 1.0f;
        if (mAbs > 0) {
            const float somx2 = std::sqrt(std::max(0.0f, (1.0f - x) * (1.0f + x)));
            float fact = 1.0f;
            for (int j = 1; j <= mAbs; ++j) {
                pmm = pmm * (-fact) * somx2;
                fact += 2.0f;
            }
        }
        if (l == mAbs) {
            return pmm;
        }
        float pm1m = x * static_cast<float>(2 * mAbs + 1) * pmm;
        if (l == mAbs + 1) {
            return pm1m;
        }
        float pll = pm1m;
        for (int ll = mAbs + 2; ll <= l; ++ll) {
            pll = ((static_cast<float>(2 * ll - 1) * x * pm1m) - (static_cast<float>(ll + mAbs - 1) * pmm)) / static_cast<float>(ll - mAbs);
            pmm = pm1m;
            pm1m = pll;
        }
        return pll;
    }

    float orbitalWavefunctionSliceReal(float x, float y, int n, int l, int m) const {
        const float r = std::sqrt(x * x + y * y);
        const float theta = std::acos(std::clamp(0.0f / std::max(r, 1e-6f), -1.0f, 1.0f));
        const float rho = 2.0f * r / std::max(static_cast<float>(n), 1.0f);
        const int k = n - l - 1;
        const int alpha = 2 * l + 1;
        const float laguerre = associatedLaguerreCpu(std::max(k, 0), alpha, rho);
        const float nn = std::max(static_cast<float>(n), 1.0f);
        const float num = factorialIntCpu(std::max(n - l - 1, 0));
        const float den = std::max(factorialIntCpu(std::max(n + l, 0)), 1e-8f);
        const float norm = std::pow(2.0f / nn, 3.0f) * num / (2.0f * nn * den);
        const float radial = std::sqrt(std::max(norm, 0.0f)) * std::exp(-rho * 0.5f) * std::pow(std::max(rho, 1e-6f), static_cast<float>(l)) * laguerre;

        const float xLeg = std::cos(theta);
        const int mAbs = std::abs(m);
        const float plm = associatedLegendreCpu(l, mAbs, xLeg);
        const float angNum = factorialIntCpu(std::max(l - mAbs, 0));
        const float angDen = std::max(factorialIntCpu(std::max(l + mAbs, 0)), 1e-8f);
        const float yNorm = (static_cast<float>(2 * l + 1) / (4.0f * kPi)) * (angNum / angDen);
        const float angularAmp = std::sqrt(std::max(yNorm, 0.0f)) * plm;

        const float phi = std::atan2(y, x);
        const float phase = static_cast<float>(m) * phi;
        return radial * angularAmp * std::cos(phase);
    }

    void resizeTdseBuffers() {
        tdseGridSize_ = std::clamp(tdseGridSize_, 64, kMaxTdseGrid);
        const size_t cellCount = static_cast<size_t>(tdseGridSize_) * static_cast<size_t>(tdseGridSize_);
        tdseReal_.assign(cellCount, 0.0f);
        tdseImag_.assign(cellCount, 0.0f);
        tdseNextReal_.assign(cellCount, 0.0f);
        tdseNextImag_.assign(cellCount, 0.0f);
        tdseRhsReal_.assign(cellCount, 0.0f);
        tdseRhsImag_.assign(cellCount, 0.0f);
        tdsePotential_.assign(cellCount, 0.0f);
        tdseUpload_.assign(cellCount * 4, 0.0f);
        tdsePotentialDirty_ = true;
        tdseWaveNeedsReset_ = true;
    }

    void rebuildTdsePotential() {
        if (tdsePotential_.empty()) {
            return;
        }

        const int n = tdseGridSize_;
        const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        const float strength = std::max(tdsePotentialStrength_, 0.0f);
        const float halfWidth = std::clamp(tdseSquareHalfWidth_, 0.1f, domain * 0.95f);
        const float radiusBig = std::clamp(tdseCircleRadiusBig_, 0.2f, domain * 0.95f);
        const float radiusSmall = std::clamp(tdseCircleRadiusSmall_, 0.2f, domain * 0.95f);
        const float annIn = std::clamp(tdseAnnulusInner_, 0.1f, domain * 0.9f);
        const float annOut = std::clamp(tdseAnnulusOuter_, annIn + 0.1f, domain * 0.95f);
        const float barrierHalfWidth = std::clamp(tdseBarrierHalfWidth_, 0.06f, domain * 0.2f);
        const float slitCenter = std::clamp(tdseSlitCenterOffset_, 0.0f, domain * 0.8f);
        const float slitHalfH = std::clamp(tdseSlitHalfHeight_, 0.05f, domain * 0.4f);

        for (int iy = 0; iy < n; ++iy) {
            const float y = -domain + static_cast<float>(iy) * dx;
            for (int ix = 0; ix < n; ++ix) {
                const float x = -domain + static_cast<float>(ix) * dx;
                float v = 0.0f;
                const float r = std::sqrt(x * x + y * y);

                switch (tdsePotentialType_) {
                case Potential2dType::SquareWell:
                    if (std::abs(x) <= halfWidth && std::abs(y) <= halfWidth) {
                        v = -strength;
                    }
                    break;
                case Potential2dType::CircleWellBig:
                    if (r <= radiusBig) {
                        v = -strength;
                    }
                    break;
                case Potential2dType::CircleWellSmall:
                    if (r <= radiusSmall) {
                        v = -strength;
                    }
                    break;
                case Potential2dType::DoubleSlitWell: {
                    const float boxHalf = std::max(halfWidth * 1.8f, 1.0f);
                    if (std::abs(x) <= boxHalf && std::abs(y) <= boxHalf) {
                        v = -0.5f * strength;
                    }
                    const bool inBarrier = (std::abs(x) <= barrierHalfWidth) && (std::abs(y) <= boxHalf * 0.95f);
                    const bool inSlitA = std::abs(y - slitCenter) <= slitHalfH;
                    const bool inSlitB = std::abs(y + slitCenter) <= slitHalfH;
                    if (inBarrier && !(inSlitA || inSlitB)) {
                        v = 2.0f * strength;
                    }
                    break;
                }
                case Potential2dType::AnnularWell: {
                    if (r >= annIn && r <= annOut) {
                        v = -strength;
                    }
                    break;
                }
                }

                tdsePotential_[static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix)] = v;
            }
        }

        tdsePotentialDirty_ = false;
    }

    void resetTdseWavefunction() {
        if (tdsePotentialDirty_) {
            rebuildTdsePotential();
        }

        const int n = tdseGridSize_;
        const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        const glm::vec2 center(
            std::clamp(tdsePacketCenter_.x, -domain * 0.95f, domain * 0.95f),
            std::clamp(tdsePacketCenter_.y, -domain * 0.95f, domain * 0.95f));
        tdsePacketCenter_ = center;
        const glm::vec2 momentum = tdsePacketMomentum_;
        const float sigma = std::max(domain * 0.12f, 0.2f);
        const float inv2Sigma2 = 1.0f / std::max(2.0f * sigma * sigma, 1e-6f);

        float norm = 0.0f;
        for (int iy = 0; iy < n; ++iy) {
            const float y = -domain + static_cast<float>(iy) * dx;
            for (int ix = 0; ix < n; ++ix) {
                const float x = -domain + static_cast<float>(ix) * dx;
                const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                const float dxp = x - center.x;
                const float dyp = y - center.y;
                const float env = std::exp(-(dxp * dxp + dyp * dyp) * inv2Sigma2);
                const float phase = momentum.x * dxp + momentum.y * dyp;
                tdseReal_[idx] = env * std::cos(phase);
                tdseImag_[idx] = env * std::sin(phase);
                norm += (tdseReal_[idx] * tdseReal_[idx] + tdseImag_[idx] * tdseImag_[idx]) * dx * dx;
            }
        }

        if (norm > 1e-12f) {
            const float inv = 1.0f / std::sqrt(norm);
            for (size_t i = 0; i < tdseReal_.size(); ++i) {
                tdseReal_[i] *= inv;
                tdseImag_[i] *= inv;
            }
        }

        tdseWaveNeedsReset_ = false;
        tdseNormCounter_ = 0;
        twoDTime_ = 0.0f;
    }

void stepTdseSimulation() {
        if (!twoDUseTdse_) {
            return;
        }
        if (tdsePotentialDirty_) {
            rebuildTdsePotential();
        }
        if (tdseWaveNeedsReset_) {
            resetTdseWavefunction();
        }

        const int n = tdseGridSize_;
        if (n < 8 || tdseReal_.size() != static_cast<size_t>(n) * static_cast<size_t>(n)) {
            return;
        }

        const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        // Mikaberidze's implementation assumes dx = 1.0 (grid units) for the Laplacian.
        // We set invDx2 = 1.0f here to perfectly match the stability and wave dynamics of the JS version.
        const float invDx2 = 1.0f; // previously: 1.0f / std::max(dx * dx, 1e-8f);
        const float dt = std::clamp(tdseDt_, 1e-6f, 2e-3f);
        const int steps = std::clamp(tdseSubstepsPerFrame_, 1, 32);
        const float absorbWidth = std::clamp(tdseAbsorbWidth_, 0.1f, domain * 0.9f);
        const float absorbStrength = std::max(tdseAbsorbStrength_, 0.0f);

        for (int step = 0; step < steps; ++step) {
            if (tdseIntegrator_ == 0) {
                // Euler (Explicit)
                for (int iy = 1; iy < n - 1; ++iy) {
                    for (int ix = 1; ix < n - 1; ++ix) {
                        const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                        const size_t left = idx - 1;
                        const size_t right = idx + 1;
                        const size_t down = idx - static_cast<size_t>(n);
                        const size_t up = idx + static_cast<size_t>(n);

                        const float lapI = (tdseImag_[left] + tdseImag_[right] + tdseImag_[down] + tdseImag_[up] - 4.0f * tdseImag_[idx]) * invDx2;
                        const float lapR = (tdseReal_[left] + tdseReal_[right] + tdseReal_[down] + tdseReal_[up] - 4.0f * tdseReal_[idx]) * invDx2;
                        const float v = tdsePotential_[idx];

                        tdseNextReal_[idx] = tdseReal_[idx] + dt * (-0.5f * lapI + v * tdseImag_[idx]);
                        tdseNextImag_[idx] = tdseImag_[idx] + dt * (0.5f * lapR - v * tdseReal_[idx]);
                    }
                }
            } else {
                // Crank-Nicolson
                // 1) Explicit RHS
                for (int iy = 1; iy < n - 1; ++iy) {
                    for (int ix = 1; ix < n - 1; ++ix) {
                        const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                        const size_t left = idx - 1;
                        const size_t right = idx + 1;
                        const size_t down = idx - static_cast<size_t>(n);
                        const size_t up = idx + static_cast<size_t>(n);

                        const float lapI = (tdseImag_[left] + tdseImag_[right] + tdseImag_[down] + tdseImag_[up] - 4.0f * tdseImag_[idx]) * invDx2;
                        const float lapR = (tdseReal_[left] + tdseReal_[right] + tdseReal_[down] + tdseReal_[up] - 4.0f * tdseReal_[idx]) * invDx2;
                        const float v = tdsePotential_[idx];

                        tdseRhsReal_[idx] = tdseReal_[idx] + 0.5f * dt * (-0.5f * lapI + v * tdseImag_[idx]);
                        tdseRhsImag_[idx] = tdseImag_[idx] + 0.5f * dt * (0.5f * lapR - v * tdseReal_[idx]);
                    }
                }

                // 2) Fixed-point iteration
                const int CN_ITERS = 10;
                for (int iter = 0; iter < CN_ITERS; ++iter) {
                    for (int iy = 1; iy < n - 1; ++iy) {
                        for (int ix = 1; ix < n - 1; ++ix) {
                            const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                            const size_t left = idx - 1;
                            const size_t right = idx + 1;
                            const size_t down = idx - static_cast<size_t>(n);
                            const size_t up = idx + static_cast<size_t>(n);

                            const float lapI = (tdseImag_[left] + tdseImag_[right] + tdseImag_[down] + tdseImag_[up] - 4.0f * tdseImag_[idx]) * invDx2;
                            const float lapR = (tdseReal_[left] + tdseReal_[right] + tdseReal_[down] + tdseReal_[up] - 4.0f * tdseReal_[idx]) * invDx2;
                            const float v = tdsePotential_[idx];

                            tdseNextReal_[idx] = tdseRhsReal_[idx] + 0.5f * dt * (-0.5f * lapI + v * tdseImag_[idx]);
                            tdseNextImag_[idx] = tdseRhsImag_[idx] + 0.5f * dt * (0.5f * lapR - v * tdseReal_[idx]);
                        }
                    }

                    if (iter < CN_ITERS - 1) {
                        tdseReal_.swap(tdseNextReal_);
                        tdseImag_.swap(tdseNextImag_);
                    }
                }
            }

            // Boundary zeroing
            for (int i = 0; i < n; ++i) {
                const size_t top = static_cast<size_t>(i);
                const size_t bottom = static_cast<size_t>(n - 1) * static_cast<size_t>(n) + static_cast<size_t>(i);
                const size_t left = static_cast<size_t>(i) * static_cast<size_t>(n);
                const size_t right = left + static_cast<size_t>(n - 1);
                tdseNextReal_[top] = tdseNextReal_[top + static_cast<size_t>(n)];
                tdseNextImag_[top] = tdseNextImag_[top + static_cast<size_t>(n)];
                tdseNextReal_[bottom] = tdseNextReal_[bottom - static_cast<size_t>(n)];
                tdseNextImag_[bottom] = tdseNextImag_[bottom - static_cast<size_t>(n)];
                tdseNextReal_[left] = tdseNextReal_[left + 1];
                tdseNextImag_[left] = tdseNextImag_[left + 1];
                tdseNextReal_[right] = tdseNextReal_[right - 1];
                tdseNextImag_[right] = tdseNextImag_[right - 1];
            }

            if (tdseUseAbsorbingBoundary_) {
                for (int iy = 0; iy < n; ++iy) {
                    for (int ix = 0; ix < n; ++ix) {
                        const float edgeCells = static_cast<float>(std::min(std::min(ix, n - 1 - ix), std::min(iy, n - 1 - iy)));
                        const float edgeDist = edgeCells * dx;
                        if (edgeDist < absorbWidth) {
                            const float s = 1.0f - (edgeDist / absorbWidth);
                            const float damping = std::exp(-absorbStrength * s * s * dt);
                            const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                            tdseNextReal_[idx] *= damping;
                            tdseNextImag_[idx] *= damping;
                        }
                    }
                }
            }

            tdseReal_.swap(tdseNextReal_);
            tdseImag_.swap(tdseNextImag_);
        }

        ++tdseNormCounter_;
        if (tdseNormCounter_ >= 10) {
            float norm = 0.0f;
            for (size_t i = 0; i < tdseReal_.size(); ++i) {
                // Mikaberidze integrates without dx*dx scaling
                norm += (tdseReal_[i] * tdseReal_[i] + tdseImag_[i] * tdseImag_[i]); // previously: * dx * dx
            }
            if (norm > 1e-12f) {
                const float inv = 1.0f / std::sqrt(norm);
                for (size_t i = 0; i < tdseReal_.size(); ++i) {
                    tdseReal_[i] *= inv;
                    tdseImag_[i] *= inv;
                }
            }
            tdseNormCounter_ = 0;
        }
    }

    void uploadTdseField() {
        if (!tdseStorage_ || tdseUpload_.empty() || tdsePotential_.empty()) {
            return;
        }
        float maxAbsPotential = 0.0f;
        for (float v : tdsePotential_) {
            maxAbsPotential = std::max(maxAbsPotential, std::abs(v));
        }
        const float invMaxAbsPotential = (maxAbsPotential > 1e-6f) ? (1.0f / maxAbsPotential) : 0.0f;

        for (size_t i = 0; i < tdseReal_.size(); ++i) {
            const float re = tdseReal_[i];
            const float im = tdseImag_[i];
            const float rho = re * re + im * im;
            float phase01 = std::atan2(im, re) / (2.0f * kPi);
            if (phase01 < 0.0f) {
                phase01 += 1.0f;
            }
            const float pot01 = 0.5f + 0.5f * tdsePotential_[i] * invMaxAbsPotential;

            const size_t base = i * 4;
            tdseUpload_[base + 0] = rho;
            tdseUpload_[base + 1] = phase01;
            tdseUpload_[base + 2] = pot01;
            tdseUpload_[base + 3] = 1.0f;
        }

        pipeline2d_->uniforms.updateStorageBuffer(tdseStorage_, tdseUpload_.data(), tdseUpload_.size() * sizeof(float));
    }

    void render2d(float dt) {
        twoDTime_ += std::max(dt, 0.0f);

        if (twoDUseTdse_) {
            stepTdseSimulation();
            uploadTdseField();
        }

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        process2dNavigation(width, height, aspect, dt);

        gpu2dState_.orbital = glm::vec4(
            static_cast<float>(quantum_.n),
            static_cast<float>(quantum_.l),
            static_cast<float>(quantum_.m),
            static_cast<float>(colorMode_));
        gpu2dState_.tuning = glm::vec4(
            std::max(intensityScale_, 0.001f),
            std::max(intensityRange_, 0.001f),
            std::max(twoDZoom_, 1e-6f),
            std::max(twoDThickness_, 0.01f));
        gpu2dState_.render = glm::vec4(twoDTime_, aspect, std::max(twoDPhaseSpeed_, 0.0f), twoDUseTdse_ ? 1.0f : 0.0f);
        gpu2dState_.pan = glm::vec4(twoDPan_.x, twoDPan_.y, twoDSliceZ_, twoDIntegrateDepth_ ? 1.0f : 0.0f);
        gpu2dState_.tdse = glm::vec4(
            static_cast<float>(tdseGridSize_),
            std::max(tdseDomainHalfExtent_, 1.0f),
            tdseOverlayPotential_ ? 1.0f : 0.0f,
            0.0f);
        pipeline2d_->updateUniform(stateUniform2d_, reinterpret_cast<const float*>(&gpu2dState_));
        pipeline2d_->setVertexBuffer(vbo2d_.get());
        pipeline2d_->setIndexBuffer(ibo2d_.get());
        pipeline = pipeline2d_;
    }

    void process2dNavigation(int width, int height, float aspect, float dt) {
        ImGuiIO& io = ImGui::GetIO();
        const bool allowMouseCapture = !io.WantCaptureMouse;
        const bool allowKeyboardCapture = !io.WantCaptureKeyboard;

        if (twoDUseTdse_ && allowKeyboardCapture) {
            const Uint8* keys = SDL_GetKeyboardState(nullptr);
            glm::vec2 move(0.0f);
            if (keys[SDL_SCANCODE_W]) move.y += 1.0f;
            if (keys[SDL_SCANCODE_S]) move.y -= 1.0f;
            if (keys[SDL_SCANCODE_A]) move.x -= 1.0f;
            if (keys[SDL_SCANCODE_D]) move.x += 1.0f;
            if (glm::dot(move, move) > 0.0f) {
                const float len = std::sqrt(move.x * move.x + move.y * move.y);
                move /= len;
                tdsePacketCenter_ += move * tdsePacketMoveSpeed_ * std::max(dt, 0.0f);
                const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
                tdsePacketCenter_.x = std::clamp(tdsePacketCenter_.x, -0.95f * domain, 0.95f * domain);
                tdsePacketCenter_.y = std::clamp(tdsePacketCenter_.y, -0.95f * domain, 0.95f * domain);
                tdseWaveNeedsReset_ = true;
            }
        }

        float wheel = Context::Instance().consumeWheelDelta();
        if (wheel != 0.0f && allowMouseCapture) {
            // Exponential wheel zoom avoids sign issues and feels scale-invariant.
            twoDZoom_ *= std::exp(wheel * 0.12f);
            twoDZoom_ = std::clamp(twoDZoom_, 1e-6f, 1e6f);
        }

        float mx = 0.0f;
        float my = 0.0f;
        Uint32 mask = SDL_GetMouseState(&mx, &my);
        const bool draggingNow = allowMouseCapture && ((mask & SDL_BUTTON_LMASK) != 0 || (mask & SDL_BUTTON_MMASK) != 0);
        if (!draggingNow) {
            twoDDragging_ = false;
            return;
        }

        if (!twoDDragging_) {
            twoDDragging_ = true;
            twoDLastMouse_ = glm::vec2(mx, my);
            return;
        }

        const glm::vec2 curr(mx, my);
        const glm::vec2 delta = curr - twoDLastMouse_;
        twoDLastMouse_ = curr;

        const float safeWidth = std::max(1, width);
        const float safeHeight = std::max(1, height);
        const float ndcDx = (2.0f * delta.x) / static_cast<float>(safeWidth);
        const float ndcDy = (-2.0f * delta.y) / static_cast<float>(safeHeight);
        twoDPan_.x -= ndcDx / (std::max(twoDZoom_, 0.01f) * std::max(aspect, 0.001f));
        twoDPan_.y -= ndcDy / std::max(twoDZoom_, 0.01f);
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
