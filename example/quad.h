#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifdef ATOMS_ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#ifdef ATOMS_ENABLE_TORCHSCRIPT
#include <torch/script.h>
#endif

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

    // Reset to sensible defaults for a given domain half-extent
    void resetForDomain(float domainHalf) {
        target    = glm::vec3(0.0f);
        radius    = domainHalf * 3.0f;
        azimuth   = 0.8f;
        elevation = 1.1f;
        dragging_ = false;
    }

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

    // Called from main.cpp before the render pass for simulation updates.
    wgfx::ComputePass computePass3d;

    void dispatchComputeSimulations(float dt) {
        if (renderPath_ == RenderPath::Path3D) {
            computePass3d.prepare();
            dispatchCompute(computePass3d);
            computePass3d.end();
        } else if (renderPath_ == RenderPath::ViralComparison) {
            stepViralSimulation(dt);
            uploadViralGrid();
        }
    }

    void render(float dt) {
        if (renderPath_ == RenderPath::Path2D) {
            render2d(dt);
            return;
        }
        if (renderPath_ == RenderPath::Path3D) {
            render3d(dt);
            return;
        }
        if (renderPath_ == RenderPath::ViralComparison) {
            renderViral(dt);
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
                                               "2d\0"
                                               "3d TDSE\0"
                                               "viral dynamics\0")) {
            renderPath_ = static_cast<RenderPath>(std::clamp(pathIndex, 0, 3));
            if (renderPath_ == RenderPath::Path2D) {
                pipeline = pipeline2d_;
                pipeline->setVertexBuffer(vbo2d_.get());
                pipeline->setIndexBuffer(ibo2d_.get());
            } else if (renderPath_ == RenderPath::Path3D) {
                pipeline = pipeline3d_;
                pipeline->setVertexBuffer(vbo3d_.get());
                pipeline->setIndexBuffer(ibo3d_.get());
                // Snap the 3D camera to fit the current domain on first switch
                camera3d_.resetForDomain(tdse3dDomainHalf_);
            } else if (renderPath_ == RenderPath::ViralComparison) {
                pipeline = pipelineViral_;
                pipeline->setVertexBuffer(vbo2d_.get());
                pipeline->setIndexBuffer(ibo2d_.get());
            } else {
                pipeline = pipelineOrbital_;
                pipeline->setVertexBuffer(vbo_.get());
                pipeline->setIndexBuffer(ibo_.get());
            }
        }

        if (renderPath_ == RenderPath::ViralComparison) {
            ImGui::Separator();
            ImGui::Text("Viral Dynamics Comparison");
            ImGui::TextWrapped("Single GPU cellular automata simulation. Choose a model preset or tune the parameters directly.");

            int gridSize = viralGridSize_;
            int model = viralModel_;
            const int oldGrid = viralGridSize_;
            const int oldModel = viralModel_;
            const float oldDensity = viralPopulationDensityPercent_;
            const float oldInitialCases = viralInitialCasesPercent_;

            if (ImGui::Combo("model##viral", &model, "COVID\0Hantavirus\0Black Plague\0Custom\0")) {
                viralModel_ = std::clamp(model, 0, 3);
                if (viralModel_ == 0) {
                    viralSpreadPercent_ = 14.0f;
                    viralMortalityPercent_ = 3.0f;
                    viralDiseaseDurationTicks_ = 30;
                } else if (viralModel_ == 1) {
                    viralSpreadPercent_ = 5.0f;
                    viralMortalityPercent_ = 38.0f;
                    viralDiseaseDurationTicks_ = 8;
                } else if (viralModel_ == 2) {
                    viralSpreadPercent_ = 12.0f;
                    viralMortalityPercent_ = 70.0f;
                    viralDiseaseDurationTicks_ = 10;
                }
                viralNeedsReset_ = true;
            }

            ImGui::SliderInt("grid##viral", &gridSize, kMinViralGrid, 200);
            ImGui::SliderFloat("ticks/sec##viral", &viralTicksPerSecond_, 0.25f, 30.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::Checkbox("paused##viral", &viralPaused_);
            ImGui::SliderFloat("Spread Prob (%)##viral", &viralSpreadPercent_, 0.0f, 100.0f, "%.1f");
            ImGui::SliderFloat("Mortality Rate (%)##viral", &viralMortalityPercent_, 0.0f, 100.0f, "%.1f");
            ImGui::SliderFloat("Pop. Density (%)##viral", &viralPopulationDensityPercent_, 1.0f, 100.0f, "%.1f");
            ImGui::SliderFloat("Initial Cases (%)##viral", &viralInitialCasesPercent_, 0.01f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderInt("Disease Duration (ticks)##viral", &viralDiseaseDurationTicks_, 1, 60);
            ImGui::SliderFloat("zoom##viral", &viralZoom_, 0.25f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::TextWrapped("Presets: COVID ~3%% fatality. Hantavirus ~38%% HPS fatality. Black Plague uses untreated plague mortality (~66-93%%) with a shorter incubation/resolution window.");

            if (ImGui::Button("Reset viral sim")) {
                viralNeedsReset_ = true;
            }
            ImGui::SameLine();
            if (ImGui::Button(viralPaused_ ? "Resume##viral" : "Pause##viral")) {
                viralPaused_ = !viralPaused_;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset view##viral")) {
                viralZoom_ = 1.0f;
                viralPan_ = glm::vec2(0.0f);
            }

            viralGridSize_ = std::clamp(gridSize, kMinViralGrid, kMaxViralGrid);
            if (viralGridSize_ != oldGrid) {
                resizeViralBuffers();
            }
            if (viralModel_ != oldModel) {
                viralNeedsReset_ = true;
            }
            if (viralPopulationDensityPercent_ != oldDensity || viralInitialCasesPercent_ != oldInitialCases) {
                viralNeedsReset_ = true;
            }
            ImGui::Text("t = %.1f", viralTime_);
            ImGui::Text("S: %d  I: %d  R: %d  D: %d",
                viralSusceptibleCount_, viralInfectedCount_, viralRecoveredCount_, viralDeadCount_);
            const int resolvedPopulation = viralRecoveredCount_ + viralDeadCount_;
            const float fatalityRatio = (resolvedPopulation > 0)
                ? (100.0f * static_cast<float>(viralDeadCount_) / static_cast<float>(resolvedPopulation))
                : 0.0f;
            ImGui::Text("Fatality ratio (resolved): %.2f%%", fatalityRatio);

            ImGui::End();
            return;
        }

        if (renderPath_ == RenderPath::Path3D) {
            ImGui::Separator();
            ImGui::Text("3D TDSE FDTD");

            int gridSize     = tdse3dGridSize_;
            int substeps     = tdse3dSubsteps_;
            int integrator   = tdse3dIntegrator_;
            int potType      = static_cast<int>(tdse3dPotType_);
            int colorMode    = tdse3dColorMode_;
            int sliceAxis    = tdse3dSliceAxis_ + 1; // 0->volume, 1->x, 2->y, 3->z
            int marchSteps   = static_cast<int>(tdse3dMarchSteps_);

            const int oldGrid  = tdse3dGridSize_;
            const Potential3dType oldPot = tdse3dPotType_;
            const float oldStrength  = tdse3dPotStrength_;
            const float oldRadius    = tdse3dPotRadius_;
            const float oldDomain    = tdse3dDomainHalf_;
            const glm::vec3 oldPos   = tdse3dPacketPos_;
            const glm::vec3 oldMom   = tdse3dPacketMom_;
            const float oldSigma     = tdse3dSigma_;
            const bool oldAbsorb     = tdse3dUseAbsorbing_;
            const float oldAbsW      = tdse3dAbsorbWidth_;
            const float oldAbsS      = tdse3dAbsorbStrength_;

            ImGui::Combo("integrator##3d", &integrator, "Euler\0Crank-Nicolson\0");
            ImGui::SliderInt("grid N##3d", &gridSize, 8, kMaxTdse3dGrid);
            ImGui::SliderInt("substeps##3d", &substeps, 1, 16);
            ImGui::SliderFloat("dt##3d", &tdse3dDt_, 1e-4f, 0.5f, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("domain half##3d", &tdse3dDomainHalf_, 4.0f, 30.0f, "%.2f");
            ImGui::Separator();
            ImGui::Text("Initial wavepacket");
            ImGui::SliderFloat3("packet pos##3d",  glm::value_ptr(tdse3dPacketPos_), -10.0f, 10.0f, "%.2f");
            ImGui::SliderFloat3("packet mom##3d",  glm::value_ptr(tdse3dPacketMom_), -6.0f,  6.0f,  "%.2f");
            ImGui::SliderFloat("sigma##3d", &tdse3dSigma_, 0.3f, 6.0f, "%.2f");
            ImGui::Separator();
            ImGui::Text("Potential");
            ImGui::Combo("potential##3d", &potType, "Free\0Harmonic well\0Coulomb well\0Spherical barrier\0Double well\0");
            ImGui::SliderFloat("strength##3d", &tdse3dPotStrength_, 0.0f, 4.0f, "%.3f");
            ImGui::SliderFloat("radius##3d",   &tdse3dPotRadius_,   0.5f, 12.0f, "%.2f");
            ImGui::Checkbox("show potential##3d", &tdse3dShowPotential_);
            ImGui::Separator();
            ImGui::Text("Absorbing boundary");
            ImGui::Checkbox("absorbing##3d", &tdse3dUseAbsorbing_);
            if (tdse3dUseAbsorbing_) {
                ImGui::SliderFloat("absorb width##3d",    &tdse3dAbsorbWidth_,    0.5f, 10.0f, "%.2f");
                ImGui::SliderFloat("absorb strength##3d", &tdse3dAbsorbStrength_, 0.5f, 40.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            }
            ImGui::Separator();
            ImGui::Text("Visualization");
            ImGui::Combo("color##3d", &colorMode, "Inferno\0Magma\0Plasma\0Viridis\0\0\0Gray\0\0\0Phase\0");
            ImGui::Combo("view##3d", &sliceAxis, "Volume (ray-march)\0Slice X\0Slice Y\0Slice Z\0");
            if (sliceAxis > 0) {
                ImGui::SliderFloat("slice pos##3d", &tdse3dSlicePos_, -tdse3dDomainHalf_, tdse3dDomainHalf_, "%.2f");
            } else {
                ImGui::SliderInt("march steps##3d", &marchSteps, 16, 256);
                ImGui::SliderFloat("alpha scale##3d", &tdse3dAlphaScale_, 0.001f, 2.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            }
            ImGui::SliderFloat("intensity scale##3d", &tdse3dIntensityScale_, 0.01f, 20.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("intensity range##3d", &tdse3dIntensityRange_, 0.01f, 4.0f,  "%.4f", ImGuiSliderFlags_Logarithmic);

            ImGui::Separator();
            ImGui::Text("ONNX rollout");
#ifdef ATOMS_ENABLE_ONNX
            ImGui::InputText("model path##onnx3d", tdse3dOnnxModelPath_, sizeof(tdse3dOnnxModelPath_));
            if (ImGui::Button("Load ONNX model##3d")) {
                loadOnnxModel3d(std::string(tdse3dOnnxModelPath_));
            }
            ImGui::SameLine();
            if (ImGui::Button("Run ONNX rollout##3d")) {
                tdse3dOnnxRunRequested_ = true;
                tdse3dUseOnnx_ = true;
            }
            ImGui::Checkbox("use ONNX mode##3d", &tdse3dUseOnnx_);
            ImGui::TextWrapped("%s", tdse3dOnnxStatus_.c_str());
#else
            ImGui::TextWrapped("ONNX support is disabled. Reconfigure with -DATOMS_ENABLE_ONNX=ON and set ONNXRUNTIME_ROOT.");
#endif

            ImGui::Separator();
            ImGui::Text("TorchScript rollout");
#ifdef ATOMS_ENABLE_TORCHSCRIPT
            ImGui::InputText("model path##torchscript3d", tdse3dTorchscriptModelPath_, sizeof(tdse3dTorchscriptModelPath_));
            ImGui::Text("TorchScript expected grid: %d", tdse3dTorchscriptExpectedGrid_);
            if (ImGui::Button("Load TorchScript model##3d")) {
                loadTorchscriptModel3d(std::string(tdse3dTorchscriptModelPath_));
            }
            ImGui::SameLine();
            if (ImGui::Button("Run TorchScript rollout##3d")) {
                tdse3dTorchscriptRunRequested_ = true;
                tdse3dUseTorchscript_ = true;
            }
            ImGui::Checkbox("use TorchScript mode##3d", &tdse3dUseTorchscript_);
            ImGui::Checkbox("autoplay TorchScript##3d", &tdse3dTorchscriptAutoplay_);
            ImGui::SliderInt("TorchScript steps/frame##3d", &tdse3dTorchscriptStepsPerFrame_, 1, 8);
            ImGui::TextWrapped("%s", tdse3dTorchscriptStatus_.c_str());
#else
            ImGui::TextWrapped("TorchScript support is disabled. Reconfigure with -DATOMS_ENABLE_TORCHSCRIPT=ON and set TORCH_ROOT.");
#endif

            if (ImGui::Button("Reset 3D wavefunction")) {
                tdse3dNeedsReset_ = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Fit camera")) {
                camera3d_.resetForDomain(tdse3dDomainHalf_);
            }
            ImGui::SameLine();
            ImGui::Text("t = %.3f", tdse3dTime_);

            if (ImGui::CollapsingHeader("3D Camera")) {
                const glm::vec3 cp3 = camera3d_.position();
                ImGui::Text("pos: (%.2f, %.2f, %.2f)", cp3.x, cp3.y, cp3.z);
                ImGui::Text("radius: %.2f", camera3d_.radius);
                ImGui::SliderFloat("orbit speed##3d", &camera3d_.orbitSpeed, 0.001f, 0.05f, "%.4f");
            }

            // Commit
            tdse3dIntegrator_ = integrator;
            tdse3dSubsteps_   = substeps;
            tdse3dColorMode_  = colorMode;
            tdse3dMarchSteps_ = static_cast<float>(std::clamp(marchSteps, 16, 256));
            tdse3dSliceAxis_  = sliceAxis - 1;
            const Potential3dType newPotType = static_cast<Potential3dType>(std::clamp(potType, 0, 4));

            if (gridSize != oldGrid) {
                tdse3dGridSize_ = gridSize;
                resize3dBuffers();
            }
            if (newPotType != oldPot ||
                tdse3dPotStrength_ != oldStrength ||
                tdse3dPotRadius_   != oldRadius   ||
                tdse3dDomainHalf_  != oldDomain) {
                tdse3dPotType_   = newPotType;
                tdse3dPotDirty_  = true;
                tdse3dNeedsReset_ = true;
            } else {
                tdse3dPotType_ = newPotType;
            }
            if (tdse3dPacketPos_ != oldPos || tdse3dPacketMom_ != oldMom || tdse3dSigma_ != oldSigma) {
                tdse3dNeedsReset_ = true;
            }
            if (tdse3dUseAbsorbing_ != oldAbsorb ||
                tdse3dAbsorbWidth_  != oldAbsW   ||
                tdse3dAbsorbStrength_ != oldAbsS) {
                tdse3dNeedsReset_ = true;
            }

            ImGui::End();
            return;
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
                ImGui::SliderFloat("dt", &tdseDt_, 1e-6f, 1.0f, "%.6f", ImGuiSliderFlags_Logarithmic);
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
        Path2D = 1,
        Path3D = 2,
        ViralComparison = 3
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
    struct alignas(16) Gpu3dState {
        glm::mat4 invViewProj;
        glm::vec4 camPos;   // xyz=cam, w=time
        glm::vec4 params;   // x=gridSize, y=domainHalf, z=intensityScale, w=intensityRange
        glm::vec4 render;   // x=colorMode, y=aspect, z=showPotential, w=sliceAxis
        glm::vec4 march;    // x=stepCount, y=alphaScale, z=slicePos, w=reserved
    };
    struct alignas(16) GpuViralState {
        glm::vec4 params; // x=gridSize, y=aspect, z=time, w=reserved
        glm::vec4 view;   // x=zoom, y=panX, z=panY, w=populationDensity
        glm::vec4 covid;  // x=spread, y=mortality, z=activeScale, w=model
        glm::vec4 hanta;  // reserved
    };
    struct ViralCell {
        int state = 0; // 0 empty, 1 susceptible, 2 infected, 3 recovered, 4 dead
        int daysInfected = 0;
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
    wgfx::Pipeline* pipelineTdse2d_ = nullptr;
    wgfx::Pipeline* pipelineViral_ = nullptr;
    wgfx::Uniform* stateUniformViral_ = nullptr;
    wgfx::Uniform* viralCurrent_ = nullptr;

    std::unique_ptr<wgfx::VertexBuffer> vbo3d_;
    std::unique_ptr<wgfx::IndexBuffer> ibo3d_;
    wgfx::Uniform* stateUniform3d_ = nullptr;
    wgfx::Uniform* tdseStorage3d_ = nullptr;
    wgfx::Pipeline* pipeline3d_ = nullptr;

    RenderPath renderPath_ = RenderPath::Orbital;

    OrbitCamera camera_;   // Orbital / 2D camera
    OrbitCamera camera3d_; // 3D TDSE camera — tight defaults set in init3dCamera()
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
    float tdseDt_ = 0.1f;
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

    // ---- Viral dynamics comparison state ----
    static constexpr int kMinViralGrid = 20;
    static constexpr int kMaxViralGrid = 512;
    GpuViralState gpuViralState_{};
    int viralGridSize_ = 60;
    float viralTicksPerSecond_ = 5.0f;
    float viralTickAccumulator_ = 0.0f;
    bool viralPaused_ = false;
    bool viralNeedsReset_ = true;
    bool viralUploadDirty_ = true;
    uint32_t viralStepIndex_ = 1;
    float viralTime_ = 0.0f;
    float viralPopulationDensityPercent_ = 62.0f;
    int viralModel_ = 0; // 0=COVID, 1=Hantavirus, 2=Black Plague, 3=Custom
    float viralSpreadPercent_ = 14.0f;
    float viralMortalityPercent_ = 3.0f;
    float viralInitialCasesPercent_ = 1.0f;
    int viralDiseaseDurationTicks_ = 30;
    int viralSusceptibleCount_ = 0;
    int viralInfectedCount_ = 0;
    int viralRecoveredCount_ = 0;
    int viralDeadCount_ = 0;
    float viralZoom_ = 1.0f;
    glm::vec2 viralPan_ = glm::vec2(0.0f);
    bool viralDragging_ = false;
    glm::vec2 viralLastMouse_ = glm::vec2(0.0f);
    uint32_t viralSeed_ = 0x5eed1234u;
    std::mt19937 viralRng_{viralSeed_};
    std::vector<ViralCell> viralGrid_;
    std::vector<ViralCell> viralNextGrid_;
    std::vector<float> viralUpload_;

    bool prevW_ = false;
    bool prevS_ = false;
    bool prevE_ = false;
    bool prevD_ = false;
    bool prevR_ = false;
    bool prevF_ = false;
    bool prevT_ = false;
    bool prevG_ = false;

    // ---- 3D TDSE state ----
    static constexpr int kMaxTdse3dGrid = 128;
    Gpu3dState gpu3dState_{};
    int tdse3dGridSize_         = 80;
    float tdse3dDomainHalf_     = 12.0f;
    float tdse3dDt_             = 0.06f;
    int tdse3dSubsteps_         = 4;
    int tdse3dIntegrator_       = 1;  // 0=Euler, 1=CN
    float tdse3dIntensityScale_ = 1.0f;
    float tdse3dIntensityRange_ = 0.3f;
    int tdse3dColorMode_        = 9;  // phase by default
    int tdse3dSliceAxis_        = -1; // -1=volume, 0/1/2=x/y/z
    float tdse3dSlicePos_       = 0.0f;
    float tdse3dMarchSteps_     = 96.0f;
    float tdse3dAlphaScale_     = 0.08f;
    bool tdse3dShowPotential_   = false;
    bool tdse3dUseAbsorbing_    = true;
    float tdse3dAbsorbWidth_    = 3.0f;
    float tdse3dAbsorbStrength_ = 12.0f;
    float tdse3dTime_           = 0.0f;

    bool tdse3dUseOnnx_          = false;
    bool tdse3dOnnxLoaded_       = false;
    bool tdse3dOnnxRunRequested_ = false;
    char tdse3dOnnxModelPath_[512] = "./tdse_fno_runs_safe/tdse_fno3d.onnx";
    std::string tdse3dOnnxStatus_ = "ONNX model not loaded";
    std::vector<float> tdse3dOnnxInput_;

#ifdef ATOMS_ENABLE_ONNX
    std::unique_ptr<Ort::Env> onnxEnv_;
    std::unique_ptr<Ort::SessionOptions> onnxSessionOptions_;
    std::unique_ptr<Ort::Session> onnxSession_;
    std::vector<std::string> onnxInputNamesStorage_;
    std::vector<std::string> onnxOutputNamesStorage_;
    std::vector<const char*> onnxInputNames_;
    std::vector<const char*> onnxOutputNames_;
#endif

    bool tdse3dUseTorchscript_            = false;
    bool tdse3dTorchscriptLoaded_         = false;
    bool tdse3dTorchscriptRunRequested_   = false;
    bool tdse3dTorchscriptAutoplay_       = true;
    int tdse3dTorchscriptStepsPerFrame_   = 1;
    int tdse3dTorchscriptExpectedGrid_    = 128;
    char tdse3dTorchscriptModelPath_[512] = "/Users/ethan/code/atoms/tdse_fno_runs_safe/tdse_fno3d_torchscript.pt";
    std::string tdse3dTorchscriptStatus_  = "TorchScript model not loaded";

#ifdef ATOMS_ENABLE_TORCHSCRIPT
    std::unique_ptr<torch::jit::script::Module> torchscriptModule_;
#endif

    enum class Potential3dType : int {
        Free = 0,
        HarmonicWell = 1,
        CoulombWell = 2,
        SphericalBarrier = 3,
        DoubleWell = 4
    };
    Potential3dType tdse3dPotType_ = Potential3dType::Free;
    float tdse3dPotStrength_  = 0.5f;
    float tdse3dPotRadius_    = 4.0f;
    glm::vec3 tdse3dPacketPos_  = glm::vec3(-4.0f, 0.0f, 0.0f);
    glm::vec3 tdse3dPacketMom_  = glm::vec3(2.0f,  0.0f, 0.0f);
    float tdse3dSigma_          = 2.0f;

    bool tdse3dNeedsReset_    = true;
    bool tdse3dPotDirty_      = true;
    int  tdse3dNormCounter_   = 0;

    // CPU-side scratch for initial upload only
    std::vector<float> tdse3dReal_;
    std::vector<float> tdse3dImag_;
    std::vector<float> tdse3dPot_;

    // GPU compute pipeline objects
    struct alignas(16) GpuComputeParams {
        uint32_t gridN;
        uint32_t mode;         // 0=Euler, 1=CN
        float    dt;
        float    absorbWidth;
        float    absorbStr;
        float    dx;
        float    _pad0;
        float    _pad1;
    };
    GpuComputeParams gpuComputeParams_{};

    wgfx::Compute*  computeEuler_    = nullptr; // entry="euler"
    wgfx::Compute*  computeCnRhs_    = nullptr; // entry="cn_rhs"
    wgfx::Compute*  computeCnIter_   = nullptr; // entry="cn_iter"
    wgfx::Compute*  computeCnFinish_ = nullptr; // entry="cn_finish"

    // Shared storage buffers (allocated to kMaxTdse3dGrid³)
    wgfx::Uniform*  computeParamsUni_ = nullptr; // binding 0 — uniform
    wgfx::Uniform*  gpuWaveA_         = nullptr; // binding 1 — vec2f (re,im)
    wgfx::Uniform*  gpuWaveB_         = nullptr; // binding 2 — vec2f (re,im)  (also render binding 1)
    wgfx::Uniform*  gpuPot_           = nullptr; // binding 3 — f32   potential (also render binding 2)

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

        // Load TDSE 2D pipeline for time-dependent simulations
        pipelineTdse2d_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "tdse2d.wgsl").c_str()));
        pipelineTdse2d_->setUniform(stateUniform2d_);  // Share same uniform as pipeline2d_
        pipelineTdse2d_->uniforms.setStorage(tdseStorage_);
        pipelineTdse2d_->targets = 1;
        pipelineTdse2d_->useDepth = false;
        pipelineTdse2d_->init(vbo2d_.get());

        pipeline3d_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "tdse3d.wgsl").c_str()));
        stateUniform3d_ = wgfx::createUniform(0, sizeof(Gpu3dState), reinterpret_cast<const float*>(&gpu3dState_));
        // setUniform done below after storage buffers are ready

        // ---- Allocate shared GPU storage buffers (kMaxTdse3dGrid³) ----
        const size_t kMaxCells = static_cast<size_t>(kMaxTdse3dGrid)
                               * static_cast<size_t>(kMaxTdse3dGrid)
                               * static_cast<size_t>(kMaxTdse3dGrid);
        // waveA and waveB: u32 packed half2 each (read-write for compute)
        gpuWaveA_ = wgfx::createStorage(1, kMaxCells * sizeof(uint32_t), nullptr, false);
        gpuWaveB_ = wgfx::createStorage(2, kMaxCells * sizeof(uint32_t), nullptr, false);
        // pot: f32 each (read-only everywhere)
        gpuPot_   = wgfx::createStorage(3, kMaxCells     * sizeof(float), nullptr, true);

        // ---- Render pipeline: read-only, fragment-only storage bindings ----
        // Wrapping the same GPU buffers without ownership to avoid double-free.
        // Fragment-only visibility avoids VERTEX_WRITABLE_STORAGE requirement.
        auto makeRenderStorage = [](wgfx::Uniform* src, int binding, bool readOnly) -> wgfx::Uniform* {
            wgfx::Uniform* u = new wgfx::Uniform();
            u->ownsBuffer    = false;          // don't free — src owns it
            u->isReadOnly    = readOnly;
            u->binding       = binding;
            u->minBindingSize = src->minBindingSize;
            u->buffer        = src->buffer;    // shared handle
            u->entry.binding = static_cast<uint32_t>(binding);
            u->entry.buffer  = src->buffer;    // same as createStorage: entry.buffer = buffer
            u->entry.offset  = 0;
            u->entry.size    = static_cast<uint64_t>(src->minBindingSize);
            return u;
        };
        wgfx::Uniform* renderWaveB = makeRenderStorage(gpuWaveB_, 1, true);
        wgfx::Uniform* renderPot   = makeRenderStorage(gpuPot_,   2, true);

        pipeline3d_->setUniform(stateUniform3d_);
        // Use fragment-only visibility for storage so no VERTEX_WRITABLE_STORAGE needed
        pipeline3d_->uniforms.visibility = WGPUShaderStage_Fragment;
        pipeline3d_->uniforms.setStorage(renderWaveB);
        pipeline3d_->uniforms.setStorage(renderPot);
        pipeline3d_->uniforms.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment; // restore
        pipeline3d_->targets = 1;
        pipeline3d_->useDepth = false;
        init3dBuffers();
        pipeline3d_->init(vbo3d_.get());

        pipelineViral_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "viral_compare.wgsl").c_str()));
        stateUniformViral_ = wgfx::createUniform(0, sizeof(GpuViralState), reinterpret_cast<const float*>(&gpuViralState_));
        const size_t kMaxViralCells = static_cast<size_t>(kMaxViralGrid) * static_cast<size_t>(kMaxViralGrid);
        viralCurrent_ = wgfx::createStorage(1, kMaxViralCells * 4 * sizeof(float), nullptr, false);
        wgfx::Uniform* renderViralCurrent = makeRenderStorage(viralCurrent_, 1, true);
        pipelineViral_->setUniform(stateUniformViral_);
        pipelineViral_->uniforms.visibility = WGPUShaderStage_Fragment;
        pipelineViral_->uniforms.setStorage(renderViralCurrent);
        pipelineViral_->uniforms.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        pipelineViral_->targets = 1;
        pipelineViral_->useDepth = false;
        pipelineViral_->setVertexBuffer(vbo2d_.get());
        pipelineViral_->setIndexBuffer(ibo2d_.get());
        pipelineViral_->init(vbo2d_.get());

        // ---- Build compute pipelines (all share one WGSL, different entry points) ----
        const std::string computeSrc = wgfx::loadFromFile(
            (std::string(RESOURCE_DIR) + "/" + "tdse3d_compute.wgsl").c_str());

        computeParamsUni_ = wgfx::createUniform(0, sizeof(GpuComputeParams),
            reinterpret_cast<const float*>(&gpuComputeParams_));

        auto makeCompute = [&](wgfx::Compute*& c, const std::string& entry) {
            c = wgfx::loadCompute(computeSrc);
            c->entryPoint = entry;   // picked up by fixed init()
            c->setUniform(computeParamsUni_);
            c->setStorage(gpuWaveA_);
            c->setStorage(gpuWaveB_);
            c->setStorage(gpuPot_);
            c->init();
        };
        makeCompute(computeEuler_,    "euler");
        makeCompute(computeCnRhs_,    "cn_rhs");
        makeCompute(computeCnIter_,   "cn_iter");
        makeCompute(computeCnFinish_, "cn_finish");


        // ---- Initial sim state ----
        resize3dBuffers();
        resizeViralBuffers();

        // Initialize the dedicated 3D TDSE camera
        camera3d_.resetForDomain(tdse3dDomainHalf_);
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

    void init3dBuffers() {
        const std::vector<float> vertices = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        const std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };
        vbo3d_.reset(wgfx::createVertexBuffer(vertices));
        vbo3d_->setTopology(PrimitiveTopology::TriangleList);
        vbo3d_->setAttribute(0, wgfx::vec3f, 0);
        ibo3d_.reset(wgfx::createIndexBuffer(indices));
        pipeline3d_->setVertexBuffer(vbo3d_.get());
        pipeline3d_->setIndexBuffer(ibo3d_.get());
    }

    void resizeViralBuffers() {
        viralGridSize_ = std::clamp(viralGridSize_, kMinViralGrid, kMaxViralGrid);
        const size_t cellCount = static_cast<size_t>(viralGridSize_) * static_cast<size_t>(viralGridSize_);
        viralGrid_.assign(cellCount, ViralCell{});
        viralNextGrid_.assign(cellCount, ViralCell{});
        viralUpload_.assign(cellCount * 4, 0.0f);
        viralNeedsReset_ = true;
        viralUploadDirty_ = true;
    }

    void resetViralSimulation() {
        const int n = std::clamp(viralGridSize_, kMinViralGrid, kMaxViralGrid);
        viralGridSize_ = n;
        const size_t cellCount = static_cast<size_t>(n) * static_cast<size_t>(n);
        if (viralGrid_.size() != cellCount) {
            viralGrid_.assign(cellCount, ViralCell{});
            viralNextGrid_.assign(cellCount, ViralCell{});
            viralUpload_.assign(cellCount * 4, 0.0f);
        }

        viralSeed_ += 0x9e3779b9u;
        viralRng_.seed(viralSeed_);
        std::uniform_real_distribution<float> random01(0.0f, 1.0f);
        const float density = std::clamp(viralPopulationDensityPercent_ * 0.01f, 0.01f, 1.0f);
        const float initialCases = std::clamp(viralInitialCasesPercent_ * 0.01f, 0.0001f, 0.25f);

        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(n) + static_cast<size_t>(x);
                const bool populated = random01(viralRng_) <= density;
                const bool seedCase = populated && random01(viralRng_) <= initialCases;
                viralGrid_[idx].state = populated ? (seedCase ? 2 : 1) : 0;
                viralGrid_[idx].daysInfected = 0;
            }
        }

        viralStepIndex_ = 1;
        viralTime_ = 0.0f;
        viralTickAccumulator_ = 0.0f;
        viralNeedsReset_ = false;
        viralUploadDirty_ = true;
        updateViralStats();
    }

    int viralIndex(int x, int y) const {
        return y * viralGridSize_ + x;
    }

    int countInfectedNeighbors(int x, int y) const {
        int count = 0;
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                if (ox == 0 && oy == 0) {
                    continue;
                }
                const int nx = x + ox;
                const int ny = y + oy;
                if (nx < 0 || ny < 0 || nx >= viralGridSize_ || ny >= viralGridSize_) {
                    continue;
                }
                if (viralGrid_[static_cast<size_t>(viralIndex(nx, ny))].state == 2) {
                    ++count;
                }
            }
        }
        return count;
    }

    void updateViralStats() {
        viralSusceptibleCount_ = 0;
        viralInfectedCount_ = 0;
        viralRecoveredCount_ = 0;
        viralDeadCount_ = 0;
        for (const ViralCell& cell : viralGrid_) {
            if (cell.state == 1) ++viralSusceptibleCount_;
            else if (cell.state == 2) ++viralInfectedCount_;
            else if (cell.state == 3) ++viralRecoveredCount_;
            else if (cell.state == 4) ++viralDeadCount_;
        }
    }

    void stepViralSimulation(float dt) {
        if (viralNeedsReset_) {
            resetViralSimulation();
        }
        if (viralPaused_ || viralTicksPerSecond_ <= 0.0f || viralGrid_.empty()) {
            return;
        }

        const float ticksPerSecond = std::clamp(viralTicksPerSecond_, 0.25f, 30.0f);
        viralTickAccumulator_ += std::max(dt, 0.0f) * ticksPerSecond;
        const int steps = std::clamp(static_cast<int>(std::floor(viralTickAccumulator_)), 0, 64);
        if (steps <= 0) {
            return;
        }
        viralTickAccumulator_ -= static_cast<float>(steps);

        std::uniform_real_distribution<float> random01(0.0f, 1.0f);
        const float spread = std::clamp(viralSpreadPercent_ * 0.01f, 0.0f, 1.0f);
        const float mortality = std::clamp(viralMortalityPercent_ * 0.01f, 0.0f, 1.0f);
        const int duration = std::max(1, viralDiseaseDurationTicks_);
        const float deathHazardPerTick = 1.0f - std::pow(1.0f - mortality, 1.0f / static_cast<float>(duration));

        for (int step = 0; step < steps; ++step) {
            viralNextGrid_ = viralGrid_;
            for (int y = 0; y < viralGridSize_; ++y) {
                for (int x = 0; x < viralGridSize_; ++x) {
                    const size_t idx = static_cast<size_t>(viralIndex(x, y));
                    const ViralCell& cell = viralGrid_[idx];
                    ViralCell& next = viralNextGrid_[idx];

                    if (cell.state == 1) {
                        const int infectedNeighbors = countInfectedNeighbors(x, y);
                        if (infectedNeighbors > 0) {
                            const float pInfection = 1.0f - std::pow(1.0f - spread, static_cast<float>(infectedNeighbors));
                            if (random01(viralRng_) < pInfection) {
                                next.state = 2;
                                next.daysInfected = 0;
                            }
                        }
                    } else if (cell.state == 2) {
                        next.daysInfected = cell.daysInfected + 1;
                        if (random01(viralRng_) < deathHazardPerTick) {
                            next.state = 4;
                            next.daysInfected = 0;
                        } else if (next.daysInfected >= duration) {
                            next.state = 3;
                            next.daysInfected = 0;
                        }
                    }
                }
            }
            viralGrid_.swap(viralNextGrid_);
            viralTime_ += 1.0f;
            ++viralStepIndex_;
        }

        updateViralStats();
        viralUploadDirty_ = true;
    }

    void uploadViralGrid() {
        if (!viralUploadDirty_ || !viralCurrent_ || viralGrid_.empty()) {
            return;
        }

        const float invDuration = 1.0f / static_cast<float>(std::max(1, viralDiseaseDurationTicks_));
        if (viralUpload_.size() != viralGrid_.size() * 4) {
            viralUpload_.assign(viralGrid_.size() * 4, 0.0f);
        }
        for (size_t i = 0; i < viralGrid_.size(); ++i) {
            const size_t base = i * 4;
            viralUpload_[base + 0] = static_cast<float>(viralGrid_[i].state);
            viralUpload_[base + 1] = static_cast<float>(viralGrid_[i].daysInfected) * invDuration;
            viralUpload_[base + 2] = 0.0f;
            viralUpload_[base + 3] = 0.0f;
        }

        wgfx::queue.writeBuffer(viralCurrent_->buffer, 0,
            viralUpload_.data(), viralUpload_.size() * sizeof(float));
        viralUploadDirty_ = false;
    }

    void resize3dBuffers() {
        tdse3dGridSize_ = std::clamp(tdse3dGridSize_, 8, kMaxTdse3dGrid);
        const size_t N = static_cast<size_t>(tdse3dGridSize_);
        const size_t total = N * N * N;
        tdse3dReal_.assign(total, 0.0f);
        tdse3dImag_.assign(total, 0.0f);
        tdse3dPot_.assign(total, 0.0f);
        tdse3dPotDirty_ = true;
        tdse3dNeedsReset_ = true;
    }

    void rebuild3dPotential() {
        const int N = tdse3dGridSize_;
        const float h = std::max(tdse3dDomainHalf_, 1.0f);
        const float dx = (2.0f * h) / static_cast<float>(N - 1);
        const float strength = std::max(tdse3dPotStrength_, 0.0f);
        const float radius = std::max(tdse3dPotRadius_, 0.1f);

        for (int iz = 0; iz < N; ++iz) {
            const float z = -h + static_cast<float>(iz) * dx;
            for (int iy = 0; iy < N; ++iy) {
                const float y = -h + static_cast<float>(iy) * dx;
                for (int ix = 0; ix < N; ++ix) {
                    const float x = -h + static_cast<float>(ix) * dx;
                    const size_t idx = static_cast<size_t>(iz) * static_cast<size_t>(N) * static_cast<size_t>(N)
                                     + static_cast<size_t>(iy) * static_cast<size_t>(N)
                                     + static_cast<size_t>(ix);
                    const float r = std::sqrt(x*x + y*y + z*z);
                    float v = 0.0f;
                    switch (tdse3dPotType_) {
                    case Potential3dType::Free:
                        v = 0.0f;
                        break;
                    case Potential3dType::HarmonicWell:
                        v = 0.5f * strength * r * r;
                        break;
                    case Potential3dType::CoulombWell:
                        v = -strength / std::max(r, 0.3f);
                        break;
                    case Potential3dType::SphericalBarrier:
                        v = (r <= radius) ? 0.0f : strength;
                        break;
                    case Potential3dType::DoubleWell: {
                        const float r1 = std::sqrt((x - radius * 0.5f)*(x - radius * 0.5f) + y*y + z*z);
                        const float r2 = std::sqrt((x + radius * 0.5f)*(x + radius * 0.5f) + y*y + z*z);
                        v = -strength / std::max(r1, 0.3f) - strength / std::max(r2, 0.3f);
                        break;
                    }
                    }
                    tdse3dPot_[idx] = v;
                }
            }
        }
        tdse3dPotDirty_ = false;
        // Upload potential to GPU (read-only buffer)
        if (gpuPot_) {
            wgfx::queue.writeBuffer(gpuPot_->buffer, 0,
                tdse3dPot_.data(), tdse3dPot_.size() * sizeof(float));
        }
    }

    void reset3dWavefunction() {
        if (tdse3dPotDirty_) { rebuild3dPotential(); }
        const int N = tdse3dGridSize_;
        const float h = std::max(tdse3dDomainHalf_, 1.0f);
        const float dx = (2.0f * h) / static_cast<float>(N - 1);
        const float sig = std::max(tdse3dSigma_, 0.1f);
        const float inv2s2 = 1.0f / (2.0f * sig * sig);
        const glm::vec3 c = tdse3dPacketPos_;
        const glm::vec3 p = tdse3dPacketMom_;

        float norm = 0.0f;
        for (int iz = 0; iz < N; ++iz) {
            const float z = -h + static_cast<float>(iz) * dx;
            for (int iy = 0; iy < N; ++iy) {
                const float y = -h + static_cast<float>(iy) * dx;
                for (int ix = 0; ix < N; ++ix) {
                    const float x = -h + static_cast<float>(ix) * dx;
                    const size_t idx = static_cast<size_t>(iz)*static_cast<size_t>(N)*static_cast<size_t>(N)
                                     + static_cast<size_t>(iy)*static_cast<size_t>(N)
                                     + static_cast<size_t>(ix);
                    const float dx_ = x - c.x;
                    const float dy_ = y - c.y;
                    const float dz_ = z - c.z;
                    const float env = std::exp(-(dx_*dx_ + dy_*dy_ + dz_*dz_) * inv2s2);
                    const float phase = p.x * dx_ + p.y * dy_ + p.z * dz_;
                    tdse3dReal_[idx] = env * std::cos(phase);
                    tdse3dImag_[idx] = env * std::sin(phase);
                    norm += (tdse3dReal_[idx]*tdse3dReal_[idx] + tdse3dImag_[idx]*tdse3dImag_[idx]);
                }
            }
        }
        if (norm > 1e-12f) {
            const float inv = 1.0f / std::sqrt(norm);
            for (size_t i = 0; i < tdse3dReal_.size(); ++i) {
                tdse3dReal_[i] *= inv;
                tdse3dImag_[i] *= inv;
            }
        }
        tdse3dNeedsReset_ = false;
        tdse3dNormCounter_ = 0;
        tdse3dTime_ = 0.0f;
        // Pack (re, im) interleaved and upload to waveA on GPU
        if (gpuWaveA_) {
            std::vector<uint32_t> packed(tdse3dReal_.size());
            for (size_t i = 0; i < tdse3dReal_.size(); ++i) {
                packed[i] = glm::packHalf2x16(glm::vec2(tdse3dReal_[i], tdse3dImag_[i]));
            }
            wgfx::queue.writeBuffer(gpuWaveA_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
            wgfx::queue.writeBuffer(gpuWaveB_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
        }
    }

    void upload3dWaveToGpu() {
        if (!gpuWaveA_ || !gpuWaveB_ || tdse3dReal_.size() != tdse3dImag_.size()) {
            return;
        }
        std::vector<uint32_t> packed(tdse3dReal_.size());
        for (size_t i = 0; i < tdse3dReal_.size(); ++i) {
            packed[i] = glm::packHalf2x16(glm::vec2(tdse3dReal_[i], tdse3dImag_[i]));
        }
        wgfx::queue.writeBuffer(gpuWaveA_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
        wgfx::queue.writeBuffer(gpuWaveB_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
    }

#ifdef ATOMS_ENABLE_ONNX
    bool loadOnnxModel3d(const std::string& modelPath) {
        try {
            if (modelPath.empty()) {
                tdse3dOnnxStatus_ = "ONNX load failed: model path is empty";
                tdse3dOnnxLoaded_ = false;
                return false;
            }

            if (!std::filesystem::exists(modelPath)) {
                tdse3dOnnxStatus_ = "ONNX load failed: file not found";
                tdse3dOnnxLoaded_ = false;
                return false;
            }

            if (!onnxEnv_) {
                onnxEnv_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "atoms_tdse3d");
            }

            onnxSessionOptions_ = std::make_unique<Ort::SessionOptions>();
            onnxSessionOptions_->SetIntraOpNumThreads(1);
            onnxSessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            onnxSession_ = std::make_unique<Ort::Session>(*onnxEnv_, modelPath.c_str(), *onnxSessionOptions_);

            Ort::AllocatorWithDefaultOptions allocator;
            const size_t inputCount = onnxSession_->GetInputCount();
            const size_t outputCount = onnxSession_->GetOutputCount();

            onnxInputNamesStorage_.clear();
            onnxOutputNamesStorage_.clear();
            onnxInputNames_.clear();
            onnxOutputNames_.clear();

            for (size_t i = 0; i < inputCount; ++i) {
                auto name = onnxSession_->GetInputNameAllocated(i, allocator);
                onnxInputNamesStorage_.push_back(name.get() ? name.get() : "");
            }
            for (size_t i = 0; i < outputCount; ++i) {
                auto name = onnxSession_->GetOutputNameAllocated(i, allocator);
                onnxOutputNamesStorage_.push_back(name.get() ? name.get() : "");
            }
            for (const std::string& n : onnxInputNamesStorage_) {
                onnxInputNames_.push_back(n.c_str());
            }
            for (const std::string& n : onnxOutputNamesStorage_) {
                onnxOutputNames_.push_back(n.c_str());
            }

            if (onnxInputNames_.empty() || onnxOutputNames_.empty()) {
                tdse3dOnnxStatus_ = "ONNX load failed: model has no inputs or outputs";
                tdse3dOnnxLoaded_ = false;
                return false;
            }

            tdse3dOnnxLoaded_ = true;
            tdse3dOnnxStatus_ = "ONNX model loaded";
            return true;
        } catch (const std::exception& e) {
            tdse3dOnnxLoaded_ = false;
            tdse3dOnnxStatus_ = std::string("ONNX load failed: ") + e.what();
            return false;
        }
    }

    bool runOnnxRollout3d() {
        try {
            if (!tdse3dOnnxLoaded_ || !onnxSession_) {
                tdse3dOnnxStatus_ = "ONNX rollout skipped: no model loaded";
                return false;
            }

            if (tdse3dPotDirty_) {
                rebuild3dPotential();
            }
            if (tdse3dNeedsReset_) {
                reset3dWavefunction();
            }

            const int N = tdse3dGridSize_;
            const size_t cells = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(N);
            if (cells == 0 || tdse3dPot_.size() != cells || tdse3dReal_.size() != cells || tdse3dImag_.size() != cells) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: invalid 3D buffer sizes";
                return false;
            }

            tdse3dOnnxInput_.resize(cells * 3);
            std::memcpy(tdse3dOnnxInput_.data() + 0 * cells, tdse3dPot_.data(),  cells * sizeof(float));
            std::memcpy(tdse3dOnnxInput_.data() + 1 * cells, tdse3dReal_.data(), cells * sizeof(float));
            std::memcpy(tdse3dOnnxInput_.data() + 2 * cells, tdse3dImag_.data(), cells * sizeof(float));

            const std::array<int64_t, 5> inputShape = {1, 3, N, N, N};
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                tdse3dOnnxInput_.data(),
                tdse3dOnnxInput_.size(),
                inputShape.data(),
                inputShape.size());

            std::vector<Ort::Value> outputs = onnxSession_->Run(
                Ort::RunOptions{nullptr},
                onnxInputNames_.data(),
                &inputTensor,
                1,
                onnxOutputNames_.data(),
                onnxOutputNames_.size());

            if (outputs.empty() || !outputs[0].IsTensor()) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: no tensor output";
                return false;
            }

            Ort::TensorTypeAndShapeInfo outInfo = outputs[0].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> outShape = outInfo.GetShape();
            if (outShape.size() != 5 || outShape[1] != 2 || outShape[2] != N || outShape[3] != N || outShape[4] != N) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: output shape mismatch";
                return false;
            }

            const float* outData = outputs[0].GetTensorData<float>();
            if (!outData) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: null output data";
                return false;
            }

            for (size_t i = 0; i < cells; ++i) {
                tdse3dReal_[i] = outData[i];
                tdse3dImag_[i] = outData[cells + i];
            }
            upload3dWaveToGpu();
            tdse3dTime_ += std::max(tdse3dDt_, 0.0f);
            tdse3dOnnxStatus_ = "ONNX rollout complete";
            return true;
        } catch (const std::exception& e) {
            tdse3dOnnxStatus_ = std::string("ONNX rollout failed: ") + e.what();
            return false;
        }
    }
#else
    bool loadOnnxModel3d(const std::string&) {
        tdse3dOnnxLoaded_ = false;
        tdse3dOnnxStatus_ = "ONNX support disabled at compile time";
        return false;
    }

    bool runOnnxRollout3d() {
        tdse3dOnnxStatus_ = "ONNX support disabled at compile time";
        return false;
    }
#endif

#ifdef ATOMS_ENABLE_TORCHSCRIPT
    bool loadTorchscriptModel3d(const std::string& modelPath) {
        try {
            if (modelPath.empty()) {
                tdse3dTorchscriptStatus_ = "TorchScript load failed: model path is empty";
                tdse3dTorchscriptLoaded_ = false;
                return false;
            }
            if (!std::filesystem::exists(modelPath)) {
                tdse3dTorchscriptStatus_ = "TorchScript load failed: file not found";
                tdse3dTorchscriptLoaded_ = false;
                return false;
            }

            auto module = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath));
            module->eval();
            torchscriptModule_ = std::move(module);

            tdse3dTorchscriptLoaded_ = true;
            tdse3dTorchscriptStatus_ = "TorchScript model loaded";
            return true;
        } catch (const std::exception& e) {
            tdse3dTorchscriptLoaded_ = false;
            tdse3dTorchscriptStatus_ = std::string("TorchScript load failed: ") + e.what();
            return false;
        }
    }

    bool runTorchscriptRollout3d() {
        try {
            if (!tdse3dTorchscriptLoaded_ || !torchscriptModule_) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout skipped: no model loaded";
                return false;
            }

            if (tdse3dGridSize_ != tdse3dTorchscriptExpectedGrid_) {
                tdse3dTorchscriptStatus_ =
                    "TorchScript rollout blocked: grid mismatch. Set grid N to "
                    + std::to_string(tdse3dTorchscriptExpectedGrid_) +
                    " and reset 3D wavefunction.";
                return false;
            }

            if (tdse3dPotDirty_) {
                rebuild3dPotential();
            }
            if (tdse3dNeedsReset_) {
                reset3dWavefunction();
            }

            const int N = tdse3dGridSize_;
            const size_t cells = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(N);
            if (cells == 0 || tdse3dPot_.size() != cells || tdse3dReal_.size() != cells || tdse3dImag_.size() != cells) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout failed: invalid 3D buffer sizes";
                return false;
            }

            std::vector<float> inputData(cells * 3);
            std::memcpy(inputData.data() + 0 * cells, tdse3dPot_.data(), cells * sizeof(float));
            std::memcpy(inputData.data() + 1 * cells, tdse3dReal_.data(), cells * sizeof(float));
            std::memcpy(inputData.data() + 2 * cells, tdse3dImag_.data(), cells * sizeof(float));

            torch::NoGradGuard noGrad;
            torch::Tensor input = torch::from_blob(inputData.data(), {1, 3, N, N, N}, torch::kFloat32).clone();
            torch::Tensor output = torchscriptModule_->forward({input}).toTensor().to(torch::kCPU).contiguous();

            if (output.dim() != 5 || output.size(0) != 1 || output.size(1) != 2 || output.size(2) != N || output.size(3) != N || output.size(4) != N) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout failed: output shape mismatch";
                return false;
            }

            const float* outData = output.data_ptr<float>();
            if (!outData) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout failed: null output data";
                return false;
            }

            for (size_t i = 0; i < cells; ++i) {
                tdse3dReal_[i] = outData[i];
                tdse3dImag_[i] = outData[cells + i];
            }
            upload3dWaveToGpu();
            tdse3dTime_ += std::max(tdse3dDt_, 0.0f);
            tdse3dTorchscriptStatus_ = "TorchScript rollout complete";
            return true;
        } catch (const std::exception& e) {
            tdse3dTorchscriptStatus_ = std::string("TorchScript rollout failed: ") + e.what();
            return false;
        }
    }
#else
    bool loadTorchscriptModel3d(const std::string&) {
        tdse3dTorchscriptLoaded_ = false;
        tdse3dTorchscriptStatus_ = "TorchScript support disabled at compile time";
        return false;
    }

    bool runTorchscriptRollout3d() {
        tdse3dTorchscriptStatus_ = "TorchScript support disabled at compile time";
        return false;
    }
#endif

    // Dispatch GPU compute for one FDTD substep.
    // Called by render3d(); the ComputePass is owned externally in main.cpp.
    // Returns the number of workgroups per axis.
    uint32_t computeWorkgroups() const {
        return (static_cast<uint32_t>(tdse3dGridSize_) + 3u) / 4u;
    }

    void dispatchCompute(wgfx::ComputePass& cp) {
        if (tdse3dUseTorchscript_) {
            int steps = 0;
            if (tdse3dTorchscriptAutoplay_) {
                steps = std::clamp(tdse3dTorchscriptStepsPerFrame_, 1, 32);
            }
            if (tdse3dTorchscriptRunRequested_) {
                steps = std::max(steps, 1);
                tdse3dTorchscriptRunRequested_ = false;
            }

            for (int i = 0; i < steps; ++i) {
                if (!runTorchscriptRollout3d()) {
                    break;
                }
            }
            return;
        }

        if (tdse3dUseOnnx_) {
            if (tdse3dOnnxRunRequested_) {
                runOnnxRollout3d();
                tdse3dOnnxRunRequested_ = false;
            }
            return;
        }

        if (tdse3dPotDirty_)   { rebuild3dPotential(); }
        if (tdse3dNeedsReset_) { reset3dWavefunction(); }
        if (!computeEuler_)    { return; }

        const float h  = std::max(tdse3dDomainHalf_, 1.0f);
        const float dx = (2.0f * h) / static_cast<float>(tdse3dGridSize_ - 1);
        const float absW = tdse3dUseAbsorbing_
            ? std::clamp(tdse3dAbsorbWidth_, 0.1f, h * 0.9f) : 0.0f;

        gpuComputeParams_.gridN       = static_cast<uint32_t>(tdse3dGridSize_);
        gpuComputeParams_.mode        = static_cast<uint32_t>(tdse3dIntegrator_);
        gpuComputeParams_.dt          = std::clamp(tdse3dDt_, 1e-6f, 2.0f);
        gpuComputeParams_.absorbWidth = absW;
        gpuComputeParams_.absorbStr   = std::max(tdse3dAbsorbStrength_, 0.0f);
        gpuComputeParams_.dx          = dx;

        // Write params ONCE at offset 0 — bypasses the accumulating dynamic
        // offset counter that would overflow after ~106 frames.
        wgfx::queue.writeBuffer(computeParamsUni_->buffer, 0,
            &gpuComputeParams_, sizeof(GpuComputeParams));

        // Pin every compute pipeline's dynamic offset for binding 0 to 0.
        // This must match the write above (offset 0 in the uniform buffer).
        auto pinOffset = [](wgfx::Compute* c) {
            if (!c) return;
            c->uniforms.dynamicOffsets.resize(1, 0);
            c->uniforms.dynamicOffsets[0] = 0;
            // Also reset the quantity so clear() won't accidentally re-advance it.
            if (!c->uniforms.uniforms.empty())
                c->uniforms.uniforms[0]->quantity = 0;
        };
        pinOffset(computeEuler_);
        pinOffset(computeCnRhs_);
        pinOffset(computeCnIter_);
        pinOffset(computeCnFinish_);

        const uint32_t wg = computeWorkgroups();
        const int substeps = std::clamp(tdse3dSubsteps_, 1, 32);
        const int CN_ITERS = 4;

        for (int s = 0; s < substeps; ++s) {
            if (tdse3dIntegrator_ == 0) {
                cp.drawXYZ(computeEuler_, wg, wg, wg);
            } else {
                cp.drawXYZ(computeCnRhs_, wg, wg, wg);
                for (int iter = 0; iter < CN_ITERS; ++iter) {
                    cp.drawXYZ(computeCnIter_, wg, wg, wg);
                }
                cp.drawXYZ(computeCnFinish_, wg, wg, wg);
            }
            std::swap(gpuWaveA_, gpuWaveB_);
            tdse3dTime_ += gpuComputeParams_.dt;
        }
    }


    void step3dSimulation() { /* replaced by dispatchCompute */ }

    void upload3dField() { /* no-op: GPU writes directly to waveB render buffer */ }

    // Kept for reference — no longer called:


    void render3d(float dt) {
        // Simulation is now dispatched via dispatchCompute3d() BEFORE the render pass.
        // Here we only update camera + render uniforms.
        int width = 1280, height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);

        ImGuiIO& io = ImGui::GetIO();
        const float wheel = Context::Instance().consumeWheelDelta();
        camera3d_.zoomSpeed = std::max(tdse3dDomainHalf_ * 0.05f, 0.5f);
        camera3d_.process(dt, !io.WantCaptureMouse, wheel);

        const float nearPlane = std::max(tdse3dDomainHalf_ * 0.01f, 0.01f);
        const float farPlane  = tdse3dDomainHalf_ * 20.0f;
        const glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, nearPlane, farPlane);
        const glm::mat4 view = glm::lookAt(camera3d_.position(), camera3d_.target, glm::vec3(0, 1, 0));
        const glm::mat4 vp   = proj * view;
        const glm::mat4 invVP = glm::inverse(vp);

        gpu3dState_.invViewProj = invVP;
        const glm::vec3 cp = camera3d_.position();
        gpu3dState_.camPos  = glm::vec4(cp, tdse3dTime_);
        gpu3dState_.params  = glm::vec4(
            static_cast<float>(tdse3dGridSize_),
            std::max(tdse3dDomainHalf_, 0.5f),
            std::max(tdse3dIntensityScale_, 0.0001f),
            std::max(tdse3dIntensityRange_, 0.0001f));
        gpu3dState_.render  = glm::vec4(
            static_cast<float>(tdse3dColorMode_),
            aspect,
            tdse3dShowPotential_ ? 1.0f : 0.0f,
            static_cast<float>(tdse3dSliceAxis_));
        gpu3dState_.march   = glm::vec4(
            tdse3dMarchSteps_,
            tdse3dAlphaScale_,
            tdse3dSlicePos_,
            0.0f);

        pipeline3d_->updateUniform(stateUniform3d_, reinterpret_cast<const float*>(&gpu3dState_));
        pipeline3d_->setVertexBuffer(vbo3d_.get());
        pipeline3d_->setIndexBuffer(ibo3d_.get());
        pipeline = pipeline3d_;
    }

    void renderViral(float dt) {
        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        processViralNavigation(width, height, aspect, dt);

        gpuViralState_.params = glm::vec4(static_cast<float>(viralGridSize_), aspect, viralTime_, 0.0f);
        gpuViralState_.view = glm::vec4(
            std::max(viralZoom_, 0.001f),
            viralPan_.x,
            viralPan_.y,
            std::clamp(viralPopulationDensityPercent_ * 0.01f, 0.01f, 1.0f));
        gpuViralState_.covid = glm::vec4(
            std::clamp(viralSpreadPercent_ * 0.01f, 0.0f, 1.0f),
            std::clamp(viralMortalityPercent_ * 0.01f, 0.0f, 1.0f),
            1.0f,
            static_cast<float>(viralModel_));
        gpuViralState_.hanta = glm::vec4(0.0f);

        pipelineViral_->updateUniform(stateUniformViral_, reinterpret_cast<const float*>(&gpuViralState_));
        pipelineViral_->setVertexBuffer(vbo2d_.get());
        pipelineViral_->setIndexBuffer(ibo2d_.get());
        pipeline = pipelineViral_;
    }

    void processViralNavigation(int width, int height, float aspect, float dt) {
        (void)dt;
        ImGuiIO& io = ImGui::GetIO();
        const bool allowMouseCapture = !io.WantCaptureMouse;

        float wheel = Context::Instance().consumeWheelDelta();
        if (wheel != 0.0f && allowMouseCapture) {
            viralZoom_ *= std::exp(wheel * 0.12f);
            viralZoom_ = std::clamp(viralZoom_, 0.25f, 20.0f);
        }

        float mx = 0.0f;
        float my = 0.0f;
        Uint32 mask = SDL_GetMouseState(&mx, &my);
        const bool draggingNow = allowMouseCapture && ((mask & SDL_BUTTON_LMASK) != 0 || (mask & SDL_BUTTON_MMASK) != 0);
        if (!draggingNow) {
            viralDragging_ = false;
            return;
        }

        if (!viralDragging_) {
            viralDragging_ = true;
            viralLastMouse_ = glm::vec2(mx, my);
            return;
        }

        const glm::vec2 mouse(mx, my);
        const glm::vec2 delta = mouse - viralLastMouse_;
        viralLastMouse_ = mouse;

        const float h = static_cast<float>(std::max(height, 1));
        glm::vec2 panDelta(
            -2.0f * delta.x / h * std::max(aspect, 0.001f) / std::max(viralZoom_, 0.001f),
             2.0f * delta.y / h / std::max(viralZoom_, 0.001f));
        viralPan_ += panDelta;
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
                    const bool inBarrier = (std::abs(x) <= barrierHalfWidth);
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
        const float dt = std::clamp(tdseDt_, 1e-6f, 1.0f);
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
                const int CN_ITERS = 4;
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
        gpu2dState_.render = glm::vec4(twoDTime_, aspect, std::max(twoDPhaseSpeed_, 0.0f), 0.0f);
        gpu2dState_.pan = glm::vec4(twoDPan_.x, twoDPan_.y, twoDSliceZ_, twoDIntegrateDepth_ ? 1.0f : 0.0f);
        gpu2dState_.tdse = glm::vec4(
            static_cast<float>(tdseGridSize_),
            std::max(tdseDomainHalfExtent_, 1.0f),
            tdseOverlayPotential_ ? 1.0f : 0.0f,
            0.0f);

        // Choose pipeline based on mode
        wgfx::Pipeline* activePipeline = twoDUseTdse_ ? pipelineTdse2d_ : pipeline2d_;
        activePipeline->updateUniform(stateUniform2d_, reinterpret_cast<const float*>(&gpu2dState_));
        activePipeline->setVertexBuffer(vbo2d_.get());
        activePipeline->setIndexBuffer(ibo2d_.get());
        pipeline = activePipeline;
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
