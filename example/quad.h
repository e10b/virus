#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "context.h"
#include "imgui.h"
#include "wgfx.h"

namespace orbital {

struct QuantumSettings {
    int n = 2;
    int l = 1;
    int m = 0;
    int sampleCount = 100000;

    void clamp() {
        n = std::max(1, n);
        l = std::clamp(l, 0, n - 1);
        m = std::clamp(m, -l, l);
    }
};

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity = glm::vec3(0.0f);
    glm::vec4 color = glm::vec4(1.0f);
};

class OrbitalPhysics {
public:
    static constexpr double kPi = 3.14159265358979323846;

    glm::vec3 sphericalToCartesian(float r, float theta, float phi) const {
        return glm::vec3(
            r * std::sin(theta) * std::cos(phi),
            r * std::cos(theta),
            r * std::sin(theta) * std::sin(phi)
        );
    }

    double sampleRadius(const QuantumSettings& q, std::mt19937& rng) const {
        const int bins = 4096;
        const double rMax = 10.0 * q.n * q.n;
        std::vector<double> cdf(bins);
        double dr = rMax / (bins - 1);
        double sum = 0.0;

        for (int i = 0; i < bins; ++i) {
            double r = i * dr;
            double rho = 2.0 * r / q.n;
            int k = q.n - q.l - 1;
            int alpha = 2 * q.l + 1;

            double L = 1.0;
            double Lm1 = 1.0 + alpha - rho;
            if (k == 1) {
                L = Lm1;
            } else if (k > 1) {
                double Lm2 = 1.0;
                for (int j = 2; j <= k; ++j) {
                    L = ((2 * j - 1 + alpha - rho) * Lm1 - (j - 1 + alpha) * Lm2) / j;
                    Lm2 = Lm1;
                    Lm1 = L;
                }
            }

            double norm = std::pow(2.0 / q.n, 3.0) * std::tgamma(q.n - q.l) / (2.0 * q.n * std::tgamma(q.n + q.l + 1));
            double radial = std::sqrt(norm) * std::exp(-rho / 2.0) * std::pow(rho, q.l) * L;
            double pdf = r * r * radial * radial;
            sum += pdf;
            cdf[i] = sum;
        }

        for (double& v : cdf) {
            v /= sum;
        }

        std::uniform_real_distribution<double> dis(0.0, 1.0);
        double u = dis(rng);
        int idx = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        return idx * dr;
    }

    double samplePolar(const QuantumSettings& q, std::mt19937& rng) const {
        const int bins = 2048;
        std::vector<double> cdf(bins);
        double dtheta = kPi / (bins - 1);
        double sum = 0.0;

        int mAbs = std::abs(q.m);
        for (int i = 0; i < bins; ++i) {
            double theta = i * dtheta;
            double x = std::cos(theta);

            double pmm = 1.0;
            if (mAbs > 0) {
                double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
                double fact = 1.0;
                for (int j = 1; j <= mAbs; ++j) {
                    pmm *= -fact * somx2;
                    fact += 2.0;
                }
            }

            double plm;
            if (q.l == mAbs) {
                plm = pmm;
            } else {
                double pm1m = x * (2 * mAbs + 1) * pmm;
                if (q.l == mAbs + 1) {
                    plm = pm1m;
                } else {
                    for (int ll = mAbs + 2; ll <= q.l; ++ll) {
                        double pll = ((2 * ll - 1) * x * pm1m - (ll + mAbs - 1) * pmm) / (ll - mAbs);
                        pmm = pm1m;
                        pm1m = pll;
                    }
                    plm = pm1m;
                }
            }

            double pdf = std::sin(theta) * plm * plm;
            sum += pdf;
            cdf[i] = sum;
        }

        for (double& v : cdf) {
            v /= sum;
        }

        std::uniform_real_distribution<double> dis(0.0, 1.0);
        double u = dis(rng);
        int idx = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        return idx * dtheta;
    }

    float sampleAzimuth(std::mt19937& rng) const {
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        return static_cast<float>(2.0 * kPi * dis(rng));
    }

    glm::vec3 flowVelocity(const Particle& p, int m) const {
        double r = glm::length(p.position);
        if (r < 1e-6) {
            return glm::vec3(0.0f);
        }

        double theta = std::acos(p.position.y / r);
        double phi = std::atan2(p.position.z, p.position.x);
        double sinTheta = std::sin(theta);
        if (std::abs(sinTheta) < 1e-4) {
            sinTheta = 1e-4;
        }

        double vMag = m / (r * sinTheta);
        return glm::vec3(
            static_cast<float>(-vMag * std::sin(phi)),
            0.0f,
            static_cast<float>(vMag * std::cos(phi))
        );
    }

    float normalizedIntensity(const glm::vec3& pos, const QuantumSettings& q) const {
        double r = glm::length(pos);
        double theta = std::acos(pos.y / std::max(r, 1e-6));

        double rho = 2.0 * r / q.n;
        int k = q.n - q.l - 1;
        int alpha = 2 * q.l + 1;

        double L = 1.0;
        if (k == 1) {
            L = 1.0 + alpha - rho;
        } else if (k > 1) {
            double Lm2 = 1.0;
            double Lm1 = 1.0 + alpha - rho;
            for (int j = 2; j <= k; ++j) {
                L = ((2 * j - 1 + alpha - rho) * Lm1 - (j - 1 + alpha) * Lm2) / j;
                Lm2 = Lm1;
                Lm1 = L;
            }
        }

        double norm = std::pow(2.0 / q.n, 3.0) * std::tgamma(q.n - q.l) / (2.0 * q.n * std::tgamma(q.n + q.l + 1));
        double radial = std::sqrt(norm) * std::exp(-rho / 2.0) * std::pow(rho, q.l) * L;
        double radialP = radial * radial;

        double x = std::cos(theta);
        int mAbs = std::abs(q.m);
        double pmm = 1.0;
        if (mAbs > 0) {
            double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
            double fact = 1.0;
            for (int j = 1; j <= mAbs; ++j) {
                pmm *= -fact * somx2;
                fact += 2.0;
            }
        }

        double plm;
        if (q.l == mAbs) {
            plm = pmm;
        } else {
            double pm1m = x * (2 * mAbs + 1) * pmm;
            if (q.l == mAbs + 1) {
                plm = pm1m;
            } else {
                for (int ll = mAbs + 2; ll <= q.l; ++ll) {
                    double pll = ((2 * ll - 1) * x * pm1m - (ll + mAbs - 1) * pmm) / (ll - mAbs);
                    pmm = pm1m;
                    pm1m = pll;
                }
                plm = pm1m;
            }
        }

        double intensity = radialP * (plm * plm);
        return std::clamp(static_cast<float>(intensity * 1.5 * std::pow(5.0, q.n)), 0.0f, 1.0f);
    }
};

class ColorMap {
public:
    enum class Mode : int {
        Fire = 0,
        Gray = 1,
        CyanMagenta = 2
    };

    glm::vec4 map(float t) const {
        t = std::clamp(t, 0.0f, 1.0f);
        if (mode_ == Mode::Gray) {
            return glm::vec4(t, t, t, 1.0f);
        }
        if (mode_ == Mode::CyanMagenta) {
            return glm::vec4(0.2f + 0.8f * t, 1.0f - 0.6f * t, 1.0f, 1.0f);
        }

        const glm::vec4 c0(0.0f, 0.0f, 0.0f, 1.0f);
        const glm::vec4 c1(0.5f, 0.0f, 0.99f, 1.0f);
        const glm::vec4 c2(0.8f, 0.0f, 0.0f, 1.0f);
        const glm::vec4 c3(1.0f, 0.5f, 0.0f, 1.0f);
        const glm::vec4 c4(1.0f, 1.0f, 0.0f, 1.0f);
        const glm::vec4 c5(1.0f, 1.0f, 1.0f, 1.0f);

        float x = t * 5.0f;
        int seg = std::clamp(static_cast<int>(x), 0, 4);
        float local = x - static_cast<float>(seg);
        switch (seg) {
        case 0: return c0 + local * (c1 - c0);
        case 1: return c1 + local * (c2 - c1);
        case 2: return c2 + local * (c3 - c2);
        case 3: return c3 + local * (c4 - c3);
        default: return c4 + local * (c5 - c4);
        }
    }

    void setMode(Mode mode) { mode_ = mode; }
    Mode mode() const { return mode_; }

private:
    Mode mode_ = Mode::Fire;
};

class OctantClipper {
public:
    glm::vec3 origin = glm::vec3(0.0f);
    int removedOctant = 1;

    bool isVisible(const glm::vec3& p) const {
        glm::vec3 v = p - origin;
        bool inRemoved = false;
        switch (removedOctant) {
        case 1: inRemoved = v.x >= 0.0f && v.y >= 0.0f && v.z >= 0.0f; break;
        case 2: inRemoved = v.x < 0.0f && v.y >= 0.0f && v.z >= 0.0f; break;
        case 3: inRemoved = v.x < 0.0f && v.y < 0.0f && v.z >= 0.0f; break;
        case 4: inRemoved = v.x >= 0.0f && v.y < 0.0f && v.z >= 0.0f; break;
        case 5: inRemoved = v.x >= 0.0f && v.y >= 0.0f && v.z < 0.0f; break;
        case 6: inRemoved = v.x < 0.0f && v.y >= 0.0f && v.z < 0.0f; break;
        case 7: inRemoved = v.x < 0.0f && v.y < 0.0f && v.z < 0.0f; break;
        default: inRemoved = v.x >= 0.0f && v.y < 0.0f && v.z < 0.0f; break;
        }
        return !inRemoved;
    }
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

class AsyncParticleGenerator {
public:
    ~AsyncParticleGenerator() { stop(); }

    void start() {
        worker_ = std::thread([this]() { run(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(requestMutex_);
            stopping_ = true;
            hasRequest_ = true;
        }
        requestCv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void request(const QuantumSettings& settings, const OrbitalPhysics* physics, const ColorMap* colorMap) {
        {
            std::lock_guard<std::mutex> lock(requestMutex_);
            requestSettings_ = settings;
            requestPhysics_ = physics;
            requestColorMap_ = colorMap;
            hasRequest_ = true;
        }
        requestCv_.notify_one();
    }

    bool consume(std::vector<Particle>& out) {
        std::lock_guard<std::mutex> lock(resultMutex_);
        if (!hasResult_) {
            return false;
        }
        out = std::move(result_);
        result_.clear();
        hasResult_ = false;
        return true;
    }

    bool isBusy() const { return busy_; }

private:
    void run() {
        while (true) {
            QuantumSettings settings;
            const OrbitalPhysics* physics = nullptr;
            const ColorMap* colorMap = nullptr;

            {
                std::unique_lock<std::mutex> lock(requestMutex_);
                requestCv_.wait(lock, [this]() { return hasRequest_; });
                if (stopping_) {
                    return;
                }
                settings = requestSettings_;
                physics = requestPhysics_;
                colorMap = requestColorMap_;
                hasRequest_ = false;
                busy_ = true;
            }

            std::vector<Particle> generated;
            generated.reserve(static_cast<size_t>(settings.sampleCount));
            std::mt19937 rng(std::random_device{}());
            for (int i = 0; i < settings.sampleCount; ++i) {
                glm::vec3 pos = physics->sphericalToCartesian(
                    static_cast<float>(physics->sampleRadius(settings, rng)),
                    static_cast<float>(physics->samplePolar(settings, rng)),
                    physics->sampleAzimuth(rng)
                );
                float t = physics->normalizedIntensity(pos, settings);
                generated.push_back({ pos, glm::vec3(0.0f), colorMap->map(t) });
            }

            {
                std::lock_guard<std::mutex> lock(resultMutex_);
                result_ = std::move(generated);
                hasResult_ = true;
            }
            busy_ = false;
        }
    }

    std::thread worker_;
    std::mutex requestMutex_;
    std::condition_variable requestCv_;
    bool hasRequest_ = false;
    bool stopping_ = false;

    QuantumSettings requestSettings_;
    const OrbitalPhysics* requestPhysics_ = nullptr;
    const ColorMap* requestColorMap_ = nullptr;

    std::mutex resultMutex_;
    std::vector<Particle> result_;
    bool hasResult_ = false;

    std::atomic<bool> busy_ = false;
};

} // namespace orbital

class Quad {
public:
    static Quad& Instance() {
        static Quad instance;
        return instance;
    }

    wgfx::Pipeline* pipeline = nullptr;

    void render(float dt) {
        syncGenerated();
        processShortcuts();

        ImGuiIO& io = ImGui::GetIO();
        float wheel = Context::Instance().consumeWheelDelta();
        camera_.process(dt, !io.WantCaptureMouse, wheel);

        for (auto& p : particles_) {
            p.velocity = physics_.flowVelocity(p, quantum_.m);
            double rr = glm::length(p.position);
            if (rr > 1e-6) {
                double theta = std::acos(p.position.y / rr);
                glm::vec3 tempPos = p.position + p.velocity * flowDt_;
                double newPhi = std::atan2(tempPos.z, tempPos.x);
                p.position = physics_.sphericalToCartesian(static_cast<float>(rr), static_cast<float>(theta), static_cast<float>(newPhi));
            }
        }

        uploadVisibleParticles();

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 2000.0f);
        glm::mat4 view = glm::lookAt(camera_.position(), camera_.target, glm::vec3(0, 1, 0));
        glm::mat4 vp = projection * view;
        pipeline->updateUniform(cameraUniform_, glm::value_ptr(vp));

        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    void drawImGuiPanel() {
        ImGui::Begin("Orbital Controls");

        int n = quantum_.n;
        int l = quantum_.l;
        int m = quantum_.m;
        int count = quantum_.sampleCount;
        int colorMode = static_cast<int>(colorMap_.mode());
        int octant = clipper_.removedOctant;
        float origin[3] = { clipper_.origin.x, clipper_.origin.y, clipper_.origin.z };

        bool regen = false;
        if (ImGui::SliderInt("n", &n, 1, 7)) regen = true;
        l = std::clamp(l, 0, n - 1);
        if (ImGui::SliderInt("l", &l, 0, std::max(0, n - 1))) regen = true;
        m = std::clamp(m, -l, l);
        if (ImGui::SliderInt("m", &m, -std::max(1, l), std::max(1, l))) regen = true;
        if (ImGui::SliderInt("samples", &count, 1000, kMaxParticles)) regen = true;

        if (ImGui::Combo("colorspace", &colorMode, "Fire\0Gray\0Cyan-Magenta\0")) {
            colorMap_.setMode(static_cast<orbital::ColorMap::Mode>(colorMode));
            recolorParticles();
        }

        ImGui::Separator();
        ImGui::Text("Selected octant is removed");
        ImGui::SliderFloat3("origin", origin, -200.0f, 200.0f, "%.2f");
        ImGui::SliderInt("removed octant", &octant, 1, 8);
        clipper_.origin = glm::vec3(origin[0], origin[1], origin[2]);
        clipper_.removedOctant = std::clamp(octant, 1, 8);

        if (ImGui::Button("Apply quantum")) {
            regen = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Recolor")) {
            recolorParticles();
        }

        ImGui::Text("Default removes octant 1 at origin (0,0,0)");
        ImGui::Text("Generator: %s", generator_.isBusy() ? "busy" : "idle");

        if (regen) {
            quantum_.n = n;
            quantum_.l = l;
            quantum_.m = m;
            quantum_.sampleCount = count;
            quantum_.clamp();
            requestRegeneration();
        }

        ImGui::End();
    }

private:
    static constexpr int kMaxParticles = 250000;

    std::unique_ptr<wgfx::VertexBuffer> vbo_;
    std::unique_ptr<wgfx::IndexBuffer> ibo_;
    wgfx::Uniform* cameraUniform_ = nullptr;

    orbital::OrbitalPhysics physics_;
    orbital::ColorMap colorMap_;
    orbital::OctantClipper clipper_;
    orbital::OrbitCamera camera_;
    orbital::QuantumSettings quantum_;
    orbital::AsyncParticleGenerator generator_;

    std::vector<orbital::Particle> particles_;
    std::vector<float> packedVertices_;

    bool prevW_ = false;
    bool prevS_ = false;
    bool prevE_ = false;
    bool prevD_ = false;
    bool prevR_ = false;
    bool prevF_ = false;
    bool prevT_ = false;
    bool prevG_ = false;

    float flowDt_ = 0.5f;

    Quad() {
        pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "atoms_sphere.wgsl").c_str()));

        float identity[16] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        cameraUniform_ = wgfx::createUniform(0, sizeof(identity), identity);
        pipeline->setUniform(cameraUniform_);

        pipeline->targets = 1;
        pipeline->useDepth = false;

        initBuffers();
        particles_ = generateSync();
        pipeline->init(vbo_.get());
        generator_.start();
    }

    ~Quad() {
        generator_.stop();
    }

    Quad(const Quad&) = delete;
    void operator=(const Quad&) = delete;

    void initBuffers() {
        packedVertices_.assign(static_cast<size_t>(kMaxParticles) * 6, 0.0f);

        std::vector<uint32_t> indices(static_cast<size_t>(kMaxParticles));
        for (uint32_t i = 0; i < static_cast<uint32_t>(kMaxParticles); ++i) {
            indices[i] = i;
        }

        vbo_.reset(wgfx::createVertexBuffer(packedVertices_));
        vbo_->setTopology(PrimitiveTopology::PointList);
        vbo_->setAttribute(0, wgfx::vec3f, 0);
        vbo_->setAttribute(1, wgfx::vec3f, 3);

        ibo_.reset(wgfx::createIndexBuffer(indices));
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    std::vector<orbital::Particle> generateSync() {
        std::vector<orbital::Particle> generated;
        generated.reserve(static_cast<size_t>(quantum_.sampleCount));

        std::mt19937 rng(std::random_device{}());
        for (int i = 0; i < quantum_.sampleCount; ++i) {
            glm::vec3 p = physics_.sphericalToCartesian(
                static_cast<float>(physics_.sampleRadius(quantum_, rng)),
                static_cast<float>(physics_.samplePolar(quantum_, rng)),
                physics_.sampleAzimuth(rng)
            );
            float t = physics_.normalizedIntensity(p, quantum_);
            generated.push_back({ p, glm::vec3(0.0f), colorMap_.map(t) });
        }

        return generated;
    }

    void requestRegeneration() {
        generator_.request(quantum_, &physics_, &colorMap_);
    }

    void syncGenerated() {
        std::vector<orbital::Particle> generated;
        if (generator_.consume(generated)) {
            particles_ = std::move(generated);
        }
    }

    void recolorParticles() {
        for (auto& p : particles_) {
            float t = physics_.normalizedIntensity(p.position, quantum_);
            p.color = colorMap_.map(t);
        }
    }

    void uploadVisibleParticles() {
        size_t visible = 0;
        for (const auto& p : particles_) {
            if (!clipper_.isVisible(p.position)) {
                continue;
            }
            if (visible >= static_cast<size_t>(kMaxParticles)) {
                break;
            }

            size_t base = visible * 6;
            packedVertices_[base + 0] = p.position.x;
            packedVertices_[base + 1] = p.position.y;
            packedVertices_[base + 2] = p.position.z;
            packedVertices_[base + 3] = p.color.r;
            packedVertices_[base + 4] = p.color.g;
            packedVertices_[base + 5] = p.color.b;
            ++visible;
        }

        ibo_->indexCount = static_cast<uint32_t>(visible);
        if (visible > 0) {
            wgfx::queue.writeBuffer(vbo_->buffer, 0, packedVertices_.data(), visible * 6 * sizeof(float));
        }
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

        bool regen = false;
        if (w && !prevW_) { quantum_.n += 1; regen = true; }
        if (s && !prevS_) { quantum_.n -= 1; regen = true; }
        if (e && !prevE_) { quantum_.l += 1; regen = true; }
        if (d && !prevD_) { quantum_.l -= 1; regen = true; }
        if (r && !prevR_) { quantum_.m += 1; regen = true; }
        if (f && !prevF_) { quantum_.m -= 1; regen = true; }
        if (t && !prevT_) { quantum_.sampleCount += 10000; regen = true; }
        if (g && !prevG_) { quantum_.sampleCount -= 10000; regen = true; }

        quantum_.sampleCount = std::clamp(quantum_.sampleCount, 1000, kMaxParticles);
        quantum_.clamp();

        prevW_ = w;
        prevS_ = s;
        prevE_ = e;
        prevD_ = d;
        prevR_ = r;
        prevF_ = f;
        prevT_ = t;
        prevG_ = g;

        if (regen) {
            requestRegeneration();
        }
    }
};
