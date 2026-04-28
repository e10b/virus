import os
import re

with open('example/quad.h', 'r') as f:
    content = f.read()

# Add vectors
if "std::vector<float> tdseRhsReal_;" not in content:
    content = content.replace("std::vector<float> tdseNextImag_;", "std::vector<float> tdseNextImag_;\n    std::vector<float> tdseRhsReal_;\n    std::vector<float> tdseRhsImag_;")

# Add integrator toggle
if "int tdseIntegrator_ = 1;" not in content:
    content = content.replace("int tdseSubstepsPerFrame_ = 10;", "int tdseSubstepsPerFrame_ = 10;\n    int tdseIntegrator_ = 1;")

# Add UI for integrator
if "ImGui::Combo(\"integrator\"," not in content:
    content = content.replace("ImGui::SliderInt(\"substeps/frame\", &tdseSubstepsPerFrame_, 1, 32);", "ImGui::Combo(\"integrator\", &tdseIntegrator_, \"Euler\\0Crank-Nicolson\\0\");\n                ImGui::SliderInt(\"substeps/frame\", &tdseSubstepsPerFrame_, 1, 32);")

# Update resizeTdseBuffers
if "tdseRhsReal_.assign" not in content:
    content = content.replace("tdseNextImag_.assign(cellCount, 0.0f);", "tdseNextImag_.assign(cellCount, 0.0f);\n        tdseRhsReal_.assign(cellCount, 0.0f);\n        tdseRhsImag_.assign(cellCount, 0.0f);")

# Replace stepTdseSimulation
new_step = """
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
        const float invDx2 = 1.0f / std::max(dx * dx, 1e-8f);
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
                norm += (tdseReal_[i] * tdseReal_[i] + tdseImag_[i] * tdseImag_[i]) * dx * dx;
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
"""

start_idx = content.find("    void stepTdseSimulation() {")
end_idx = content.find("    void uploadTdseField() {", start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_step.strip() + "\n\n" + content[end_idx:]
    with open('example/quad.h', 'w') as f:
        f.write(content)
