#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>

struct SpinOrbitalState {
    int n = 1;
    int l = 0;
    int m = 0;
    int spin = 1; // +1 up, -1 down
};

class SlaterVMC {
public:
    struct WalkerState {
        std::vector<glm::vec3> positions;
        double psi = 1.0;
    };

    struct Stats {
        int acceptedMoves = 0;
        int attemptedMoves = 0;
        double acceptance = 0.0;
        double avgEnergy = 0.0;
        int measurements = 0;
    };

    void setNucleusCharge(int z) {
        nucleusZ_ = std::max(1, z);
    }

    void setParameters(double stepSize, double zetaScale, double jastrowBeta) {
        stepSize_ = std::clamp(stepSize, 0.01, 3.0);
        zetaScale_ = std::clamp(zetaScale, 0.1, 5.0);
        jastrowBeta_ = std::clamp(jastrowBeta, 0.0, 2.0);
    }

    void setWalkerCount(int walkerCount) {
        const int clamped = std::clamp(walkerCount, 1, 1024);
        if (clamped == walkerCount_) {
            return;
        }
        walkerCount_ = clamped;
        if (!orbitals_.empty()) {
            resetWalkers(lastSeed_ ^ 0x9e3779b9u);
        }
    }

    int walkerCount() const {
        return walkerCount_;
    }

    bool configure(const std::vector<SpinOrbitalState>& orbitals, std::string& err) {
        err.clear();
        orbitals_ = orbitals;
        if (orbitals_.empty()) {
            err = "No occupied spin orbitals";
            return false;
        }

        upOrbitals_.clear();
        downOrbitals_.clear();
        for (const auto& o : orbitals_) {
            if (o.spin >= 0) {
                upOrbitals_.push_back(o);
            } else {
                downOrbitals_.push_back(o);
            }
        }

        if (upOrbitals_.empty() && downOrbitals_.empty()) {
            err = "No spin blocks in configuration";
            return false;
        }

        resetWalkers();
        return true;
    }

    void resetWalkers(uint32_t seed = 0x12345678u) {
        lastSeed_ = seed;
        rng_.seed(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        const float spread = static_cast<float>(std::max(0.3, 2.0 / std::sqrt(static_cast<double>(nucleusZ_))));

        walkers_.assign(static_cast<size_t>(walkerCount_), WalkerState{});
        for (auto& walker : walkers_) {
            walker.positions.assign(orbitals_.size(), glm::vec3(0.0f));
            for (auto& p : walker.positions) {
                p = glm::vec3(normal(rng_), normal(rng_), normal(rng_)) * spread;
            }
            walker.psi = wavefunction(walker.positions);
            if (!std::isfinite(walker.psi) || std::abs(walker.psi) < kTiny) {
                walker.psi = kTiny;
            }
        }

        acceptedMoves_ = 0;
        attemptedMoves_ = 0;
        measuredEnergySum_ = 0.0;
        measurements_ = 0;
        sampleCloud_.clear();
    }

    void step(int sweeps, int thermalizationSweeps, int measureEverySweeps) {
        if (orbitals_.empty() || walkers_.empty()) {
            return;
        }

        const int electrons = static_cast<int>(orbitals_.size());
        std::uniform_real_distribution<double> unit01(0.0, 1.0);
        std::uniform_real_distribution<float> disp(-static_cast<float>(stepSize_), static_cast<float>(stepSize_));

        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (auto& walker : walkers_) {
                for (int i = 0; i < electrons; ++i) {
                    attemptedMoves_++;
                    std::vector<glm::vec3> trial = walker.positions;
                    trial[static_cast<size_t>(i)] += glm::vec3(disp(rng_), disp(rng_), disp(rng_));

                    const double psiNew = wavefunction(trial);
                    const double oldP = walker.psi * walker.psi;
                    const double newP = psiNew * psiNew;
                    const double ratio = (oldP > kTiny) ? (newP / oldP) : 0.0;

                    if (ratio >= 1.0 || unit01(rng_) < ratio) {
                        walker.positions.swap(trial);
                        walker.psi = psiNew;
                        acceptedMoves_++;
                    }
                }
            }

            if (sweep >= thermalizationSweeps) {
                const int rel = sweep - thermalizationSweeps;
                if (measureEverySweeps > 0 && (rel % measureEverySweeps) == 0) {
                    for (const auto& walker : walkers_) {
                        measuredEnergySum_ += localEnergyFiniteDifference(walker.positions);
                        measurements_++;
                        for (const auto& p : walker.positions) {
                            sampleCloud_.push_back(p);
                        }
                    }
                    if (sampleCloud_.size() > maxCloudPoints_) {
                        const size_t extra = sampleCloud_.size() - maxCloudPoints_;
                        sampleCloud_.erase(sampleCloud_.begin(), sampleCloud_.begin() + static_cast<std::ptrdiff_t>(extra));
                    }
                }
            }
        }
    }

    Stats stats() const {
        Stats s;
        s.acceptedMoves = acceptedMoves_;
        s.attemptedMoves = attemptedMoves_;
        s.acceptance = (attemptedMoves_ > 0) ? (static_cast<double>(acceptedMoves_) / static_cast<double>(attemptedMoves_)) : 0.0;
        s.avgEnergy = (measurements_ > 0) ? (measuredEnergySum_ / static_cast<double>(measurements_)) : 0.0;
        s.measurements = measurements_;
        return s;
    }

    const std::vector<glm::vec3>& positions() const {
        static const std::vector<glm::vec3> empty;
        if (walkers_.empty()) {
            return empty;
        }
        return walkers_.front().positions;
    }

    const std::vector<glm::vec3>& pointCloud() const {
        return sampleCloud_;
    }

    void clearPointCloud() {
        sampleCloud_.clear();
    }

    void setMaxCloudPoints(size_t maxPoints) {
        maxCloudPoints_ = std::max<size_t>(64, maxPoints);
        if (sampleCloud_.size() > maxCloudPoints_) {
            const size_t extra = sampleCloud_.size() - maxCloudPoints_;
            sampleCloud_.erase(sampleCloud_.begin(), sampleCloud_.begin() + static_cast<std::ptrdiff_t>(extra));
        }
    }

private:
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kTiny = 1e-14;

    std::vector<SpinOrbitalState> orbitals_;
    std::vector<SpinOrbitalState> upOrbitals_;
    std::vector<SpinOrbitalState> downOrbitals_;
    std::vector<WalkerState> walkers_;
    std::vector<glm::vec3> sampleCloud_;
    size_t maxCloudPoints_ = 200000;
    int walkerCount_ = 64;

    int nucleusZ_ = 1;
    double stepSize_ = 0.35;
    double zetaScale_ = 1.0;
    double jastrowBeta_ = 0.2;

    std::mt19937 rng_{0x12345678u};
    uint32_t lastSeed_ = 0x12345678u;

    int acceptedMoves_ = 0;
    int attemptedMoves_ = 0;
    double measuredEnergySum_ = 0.0;
    int measurements_ = 0;

    static double factorialInt(int v) {
        if (v <= 1) {
            return 1.0;
        }
        double out = 1.0;
        for (int i = 2; i <= v; ++i) {
            out *= static_cast<double>(i);
        }
        return out;
    }

    static double associatedLaguerre(int k, int alpha, double x) {
        if (k <= 0) {
            return 1.0;
        }
        double lm2 = 1.0;
        double lm1 = 1.0 + static_cast<double>(alpha) - x;
        if (k == 1) {
            return lm1;
        }

        double l = lm1;
        for (int j = 2; j <= k; ++j) {
            l = (((2.0 * j - 1.0 + static_cast<double>(alpha)) - x) * lm1 - (j - 1.0 + static_cast<double>(alpha)) * lm2) / static_cast<double>(j);
            lm2 = lm1;
            lm1 = l;
        }
        return l;
    }

    static double associatedLegendre(int l, int mAbs, double x) {
        double pmm = 1.0;
        if (mAbs > 0) {
            const double somx2 = std::sqrt(std::max(0.0, (1.0 - x) * (1.0 + x)));
            double fact = 1.0;
            for (int j = 1; j <= mAbs; ++j) {
                pmm = pmm * (-fact) * somx2;
                fact += 2.0;
            }
        }

        if (l == mAbs) {
            return pmm;
        }

        double pm1m = x * static_cast<double>(2 * mAbs + 1) * pmm;
        if (l == mAbs + 1) {
            return pm1m;
        }

        double pll = pm1m;
        for (int ll = mAbs + 2; ll <= l; ++ll) {
            pll = ((2.0 * ll - 1.0) * x * pm1m - (ll + mAbs - 1.0) * pmm) / static_cast<double>(ll - mAbs);
            pmm = pm1m;
            pm1m = pll;
        }
        return pll;
    }

    double radialHydrogenic(double r, int n, int l) const {
        const double nF = std::max(1.0, static_cast<double>(n));
        const double zEff = zetaScale_ * static_cast<double>(nucleusZ_);
        const double rho = 2.0 * zEff * r / nF;
        const int k = std::max(0, n - l - 1);
        const int alpha = 2 * l + 1;

        const double num = factorialInt(std::max(0, n - l - 1));
        const double den = factorialInt(std::max(0, n + l));
        const double norm = std::pow(2.0 * zEff / nF, 1.5) * std::sqrt(num / (2.0 * nF * std::max(den, 1e-20)));
        const double lag = associatedLaguerre(k, alpha, rho);
        return norm * std::exp(-0.5 * rho) * std::pow(std::max(rho, 1e-12), static_cast<double>(l)) * lag;
    }

    static double realSphericalHarmonic(int l, int m, double theta, double phi) {
        const int mAbs = std::abs(m);
        const double x = std::cos(theta);
        const double p = associatedLegendre(l, mAbs, x);
        const double angNum = factorialInt(std::max(0, l - mAbs));
        const double angDen = std::max(factorialInt(std::max(0, l + mAbs)), 1e-20);
        const double nrm = std::sqrt((2.0 * l + 1.0) / (4.0 * kPi) * (angNum / angDen));

        if (m == 0) {
            return nrm * p;
        }
        if (m > 0) {
            return std::sqrt(2.0) * nrm * p * std::cos(static_cast<double>(m) * phi);
        }
        return std::sqrt(2.0) * nrm * p * std::sin(static_cast<double>(mAbs) * phi);
    }

    double orbitalPhi(const SpinOrbitalState& o, const glm::vec3& p) const {
        const double r = std::sqrt(static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y + static_cast<double>(p.z) * p.z);
        const double theta = (r > 1e-12) ? std::acos(std::clamp(static_cast<double>(p.y) / r, -1.0, 1.0)) : 0.0;
        const double phi = std::atan2(static_cast<double>(p.z), static_cast<double>(p.x));

        const double rad = radialHydrogenic(r, o.n, o.l);
        const double ylm = realSphericalHarmonic(o.l, o.m, theta, phi);
        return rad * ylm;
    }

    static double determinant(std::vector<std::vector<double>> m) {
        const int n = static_cast<int>(m.size());
        if (n == 0) {
            return 1.0;
        }

        double det = 1.0;
        int sign = 1;

        for (int col = 0; col < n; ++col) {
            int pivot = col;
            double best = std::abs(m[static_cast<size_t>(col)][static_cast<size_t>(col)]);
            for (int row = col + 1; row < n; ++row) {
                const double v = std::abs(m[static_cast<size_t>(row)][static_cast<size_t>(col)]);
                if (v > best) {
                    best = v;
                    pivot = row;
                }
            }

            if (best < 1e-20) {
                return 0.0;
            }

            if (pivot != col) {
                std::swap(m[static_cast<size_t>(pivot)], m[static_cast<size_t>(col)]);
                sign = -sign;
            }

            const double diag = m[static_cast<size_t>(col)][static_cast<size_t>(col)];
            det *= diag;

            for (int row = col + 1; row < n; ++row) {
                const double factor = m[static_cast<size_t>(row)][static_cast<size_t>(col)] / diag;
                for (int k = col + 1; k < n; ++k) {
                    m[static_cast<size_t>(row)][static_cast<size_t>(k)] -= factor * m[static_cast<size_t>(col)][static_cast<size_t>(k)];
                }
            }
        }

        return static_cast<double>(sign) * det;
    }

    double blockDeterminant(const std::vector<glm::vec3>& pos, int spinSign, const std::vector<SpinOrbitalState>& orbitals) const {
        std::vector<size_t> indices;
        for (size_t i = 0; i < this->orbitals_.size(); ++i) {
            if ((this->orbitals_[i].spin >= 0 ? 1 : -1) == spinSign) {
                indices.push_back(i);
            }
        }

        const int n = static_cast<int>(orbitals.size());
        if (n == 0) {
            return 1.0;
        }
        if (static_cast<int>(indices.size()) != n) {
            return 0.0;
        }

        std::vector<std::vector<double>> mat(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                mat[static_cast<size_t>(i)][static_cast<size_t>(j)] = orbitalPhi(orbitals[static_cast<size_t>(j)], pos[indices[static_cast<size_t>(i)]]);
            }
        }

        return determinant(std::move(mat));
    }

    double jastrowFactor(const std::vector<glm::vec3>& pos) const {
        if (jastrowBeta_ <= 0.0) {
            return 1.0;
        }

        double sum = 0.0;
        const int n = static_cast<int>(pos.size());
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const glm::vec3 d = pos[static_cast<size_t>(i)] - pos[static_cast<size_t>(j)];
                const double rij = std::sqrt(static_cast<double>(d.x) * d.x + static_cast<double>(d.y) * d.y + static_cast<double>(d.z) * d.z);
                sum += rij / (1.0 + jastrowBeta_ * rij);
            }
        }
        return std::exp(0.5 * sum);
    }

    double wavefunction(const std::vector<glm::vec3>& pos) const {
        const double detUp = blockDeterminant(pos, +1, upOrbitals_);
        const double detDn = blockDeterminant(pos, -1, downOrbitals_);
        const double jastrow = jastrowFactor(pos);
        const double psi = detUp * detDn * jastrow;
        if (!std::isfinite(psi)) {
            return 0.0;
        }
        return psi;
    }

    double potentialEnergy(const std::vector<glm::vec3>& pos) const {
        const int n = static_cast<int>(pos.size());
        double v = 0.0;

        for (int i = 0; i < n; ++i) {
            const glm::vec3 p = pos[static_cast<size_t>(i)];
            const double r = std::sqrt(static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y + static_cast<double>(p.z) * p.z);
            v += -static_cast<double>(nucleusZ_) / std::max(r, 1e-8);
        }

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const glm::vec3 d = pos[static_cast<size_t>(i)] - pos[static_cast<size_t>(j)];
                const double rij = std::sqrt(static_cast<double>(d.x) * d.x + static_cast<double>(d.y) * d.y + static_cast<double>(d.z) * d.z);
                v += 1.0 / std::max(rij, 1e-8);
            }
        }

        return v;
    }

    double localEnergyFiniteDifference(const std::vector<glm::vec3>& pos) const {
        const double psi0 = wavefunction(pos);
        if (std::abs(psi0) < kTiny) {
            return 1e6;
        }

        const double h = 1e-2;
        const double invH2 = 1.0 / (h * h);

        double laplaceOverPsi = 0.0;
        for (size_t i = 0; i < pos.size(); ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                std::vector<glm::vec3> plus = pos;
                std::vector<glm::vec3> minus = pos;
                plus[i][axis] += static_cast<float>(h);
                minus[i][axis] -= static_cast<float>(h);

                const double psiPlus = wavefunction(plus);
                const double psiMinus = wavefunction(minus);
                laplaceOverPsi += (psiPlus - 2.0 * psi0 + psiMinus) * invH2 / psi0;
            }
        }

        const double kinetic = -0.5 * laplaceOverPsi;
        const double potential = potentialEnergy(pos);
        return kinetic + potential;
    }
};
