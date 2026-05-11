#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "context.h"
#include "imgui.h"
#include "wgfx.h"

class Quad {
public:
    static Quad& Instance() {
        static Quad instance;
        return instance;
    }

    wgfx::Pipeline* pipeline = nullptr;

    void dispatchComputeSimulations(float dt) {
        stepViralSimulation(dt);
        uploadViralGrid();
    }

    void render(float dt) {
        renderViral(dt);
    }

    void drawImGuiPanel() {
        ImGui::Begin("Viral Dynamics");
        ImGui::TextWrapped("Cellular automata infection model with real-world inspired presets.");

        int gridSize = viralGridSize_;
        int model = viralModel_;
        const int oldGrid = viralGridSize_;
        const float oldDensity = viralPopulationDensityPercent_;
        const float oldInitialCases = viralInitialCasesPercent_;

        if (ImGui::Combo("model", &model, "COVID\0Hantavirus\0Black Plague\0Custom\0")) {
            viralModel_ = std::clamp(model, 0, 3);
            applyViralPreset();
            viralNeedsReset_ = true;
        }

        ImGui::SliderInt("grid", &gridSize, kMinViralGrid, 200);
        ImGui::SliderFloat("ticks/sec", &viralTicksPerSecond_, 0.25f, 30.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("paused", &viralPaused_);
        ImGui::SliderFloat("Spread Prob (%)", &viralSpreadPercent_, 0.0f, 100.0f, "%.1f");
        ImGui::SliderFloat("Mortality Rate (%)", &viralMortalityPercent_, 0.0f, 100.0f, "%.1f");
        ImGui::SliderFloat("Pop. Density (%)", &viralPopulationDensityPercent_, 1.0f, 100.0f, "%.1f");
        ImGui::SliderFloat("Initial Cases (%)", &viralInitialCasesPercent_, 0.01f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("Disease Duration (ticks)", &viralDiseaseDurationTicks_, 1, 60);
        ImGui::SliderFloat("zoom", &viralZoom_, 0.25f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);

        if (ImGui::Button("Reset sim")) {
            viralNeedsReset_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(viralPaused_ ? "Resume" : "Pause")) {
            viralPaused_ = !viralPaused_;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset view")) {
            viralZoom_ = 1.0f;
            viralPan_ = glm::vec2(0.0f);
        }

        viralGridSize_ = std::clamp(gridSize, kMinViralGrid, kMaxViralGrid);
        if (viralGridSize_ != oldGrid) {
            resizeViralBuffers();
        }
        if (viralPopulationDensityPercent_ != oldDensity || viralInitialCasesPercent_ != oldInitialCases) {
            viralNeedsReset_ = true;
        }

        ImGui::Separator();
        ImGui::Text("t = %.1f", viralTime_);
        ImGui::Text("S: %d  I: %d  R: %d  D: %d",
            viralSusceptibleCount_, viralInfectedCount_, viralRecoveredCount_, viralDeadCount_);
        const int resolvedPopulation = viralRecoveredCount_ + viralDeadCount_;
        const float fatalityRatio = (resolvedPopulation > 0)
            ? (100.0f * static_cast<float>(viralDeadCount_) / static_cast<float>(resolvedPopulation))
            : 0.0f;
        ImGui::Text("Fatality ratio (resolved): %.2f%%", fatalityRatio);

        ImGui::Separator();
        ImGui::TextWrapped("Spread: P(infection) = 1 - (1 - p)^n, where p is spread probability and n is infected neighbors.");
        ImGui::TextWrapped("Death hazard per tick: 1 - (1 - mortality)^(1 / duration). Recovered cells return to grey.");
        ImGui::TextWrapped("Mouse wheel zooms. Drag the viral field to pan.");
        ImGui::End();
    }

private:
    struct alignas(16) GpuViralState {
        glm::vec4 params; // x=gridSize, y=aspect, z=time, w=reserved
        glm::vec4 view;   // x=zoom, y=panX, z=panY, w=populationDensity
        glm::vec4 covid;  // x=spread, y=mortality, z=reserved, w=model
        glm::vec4 hanta;  // reserved
    };

    struct ViralCell {
        int state = 0; // 0 empty, 1 susceptible, 2 infected, 3 recovered, 4 dead
        int daysInfected = 0;
    };

    static constexpr int kMinViralGrid = 20;
    static constexpr int kMaxViralGrid = 512;

    std::unique_ptr<wgfx::VertexBuffer> vbo_;
    std::unique_ptr<wgfx::IndexBuffer> ibo_;
    wgfx::Uniform* stateUniform_ = nullptr;
    wgfx::Uniform* viralCurrent_ = nullptr;

    GpuViralState gpuViralState_{};
    int viralGridSize_ = 60;
    float viralTicksPerSecond_ = 5.0f;
    float viralTickAccumulator_ = 0.0f;
    bool viralPaused_ = false;
    bool viralNeedsReset_ = true;
    bool viralUploadDirty_ = true;
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

    Quad() {
        pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/viral_compare.wgsl").c_str()));

        stateUniform_ = wgfx::createUniform(0, sizeof(GpuViralState), reinterpret_cast<const float*>(&gpuViralState_));
        const size_t maxCells = static_cast<size_t>(kMaxViralGrid) * static_cast<size_t>(kMaxViralGrid);
        viralCurrent_ = wgfx::createStorage(1, maxCells * 4 * sizeof(float), nullptr, true);

        pipeline->setUniform(stateUniform_);
        pipeline->setStorage(viralCurrent_);
        pipeline->targets = 1;
        pipeline->useDepth = false;

        initFullscreenBuffers();
        pipeline->init(vbo_.get());

        resizeViralBuffers();
        resetViralSimulation();
        uploadViralGrid();
    }

    Quad(const Quad&) = delete;
    void operator=(const Quad&) = delete;

    void initFullscreenBuffers() {
        const std::vector<float> vertices = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        const std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };

        vbo_.reset(wgfx::createVertexBuffer(vertices));
        vbo_->setTopology(PrimitiveTopology::TriangleList);
        vbo_->setAttribute(0, wgfx::vec3f, 0);

        ibo_.reset(wgfx::createIndexBuffer(indices));
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    void applyViralPreset() {
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
        std::vector<size_t> susceptible;

        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(n) + static_cast<size_t>(x);
                const bool populated = random01(viralRng_) <= density;
                viralGrid_[idx].state = populated ? 1 : 0;
                viralGrid_[idx].daysInfected = 0;
                if (populated) {
                    susceptible.push_back(idx);
                }
            }
        }

        if (!susceptible.empty()) {
            std::shuffle(susceptible.begin(), susceptible.end(), viralRng_);
            const int seedCount = std::clamp(
                static_cast<int>(std::round(static_cast<float>(susceptible.size()) * initialCases)),
                1,
                static_cast<int>(susceptible.size()));
            for (int i = 0; i < seedCount; ++i) {
                viralGrid_[susceptible[static_cast<size_t>(i)]].state = 2;
            }
        }

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

    void renderViral(float dt) {
        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        processViralNavigation(width, height, aspect, dt);

        gpuViralState_.params = glm::vec4(static_cast<float>(viralGridSize_), aspect, viralTime_, 0.0f);
        gpuViralState_.view = glm::vec4(
            std::clamp(viralZoom_, 0.25f, 20.0f),
            viralPan_.x,
            viralPan_.y,
            viralPopulationDensityPercent_ * 0.01f);
        gpuViralState_.covid = glm::vec4(
            viralSpreadPercent_ * 0.01f,
            viralMortalityPercent_ * 0.01f,
            1.0f,
            static_cast<float>(viralModel_));
        gpuViralState_.hanta = glm::vec4(0.0f);

        pipeline->updateUniform(stateUniform_, reinterpret_cast<const float*>(&gpuViralState_));
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    void processViralNavigation(int width, int height, float aspect, float dt) {
        (void)dt;
        ImGuiIO& io = ImGui::GetIO();
        const float wheel = Context::Instance().consumeWheelDelta();
        if (!io.WantCaptureMouse && wheel != 0.0f) {
            const float zoomFactor = std::pow(1.12f, wheel);
            viralZoom_ = std::clamp(viralZoom_ * zoomFactor, 0.25f, 20.0f);
        }

        float mx = 0.0f;
        float my = 0.0f;
        const Uint32 mask = SDL_GetMouseState(&mx, &my);
        const bool draggingNow = !io.WantCaptureMouse && ((mask & SDL_BUTTON_LMASK) != 0);
        if (draggingNow) {
            const glm::vec2 mouse(mx, my);
            if (!viralDragging_) {
                viralDragging_ = true;
                viralLastMouse_ = mouse;
            } else {
                const glm::vec2 delta = mouse - viralLastMouse_;
                const float safeWidth = static_cast<float>(std::max(width, 1));
                const float safeHeight = static_cast<float>(std::max(height, 1));
                viralPan_.x += (delta.x / safeWidth) * 2.0f * aspect / std::max(viralZoom_, 0.001f);
                viralPan_.y -= (delta.y / safeHeight) * 2.0f / std::max(viralZoom_, 0.001f);
                viralLastMouse_ = mouse;
            }
        } else {
            viralDragging_ = false;
        }
    }
};
