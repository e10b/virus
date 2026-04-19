#include "audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

bool AudioManager::initialize()
{
    if (stream_) {
        return true;
    }

    if (!(SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO)) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            std::printf("Audio subsystem init failed: %s\n", SDL_GetError());
            return false;
        }
    }

    spec_.format = SDL_AUDIO_F32;
    spec_.channels = 1;
    spec_.freq = 48000;

    stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec_, nullptr, nullptr);
    if (!stream_) {
        std::printf("Audio init failed: %s\n", SDL_GetError());
        return false;
    }

    if (!SDL_ResumeAudioStreamDevice(stream_)) {
        std::printf("Audio resume failed: %s\n", SDL_GetError());
        SDL_DestroyAudioStream(stream_);
        stream_ = nullptr;
        return false;
    }

    buildClickBuffer();
    return true;
}

void AudioManager::playClick()
{
    playClickWithVolume(1.0f);
}

void AudioManager::playClickWithVolume(float volume)
{
    if (!initialize() || clickBuffer_.empty()) {
        return;
    }

    SDL_ClearAudioStream(stream_);

    // Apply volume scaling to buffer
    std::vector<float> scaledBuffer = clickBuffer_;
    for (float& sample : scaledBuffer) {
        sample *= volume;
    }

    const int byteCount = static_cast<int>(scaledBuffer.size() * sizeof(float));
    if (!SDL_PutAudioStreamData(stream_, scaledBuffer.data(), byteCount)) {
        std::printf("Audio queue failed: %s\n", SDL_GetError());
        return;
    }

    SDL_FlushAudioStream(stream_);
}

void AudioManager::playRaytracedClick(float directGain, const std::vector<AudioReflectionTap>& reflections)
{
    if (!initialize() || clickBuffer_.empty()) {
        return;
    }

    const float clampedDirect = std::clamp(directGain, 0.0f, 1.0f);
    size_t maxDelaySamples = 0;
    for (const AudioReflectionTap& tap : reflections) {
        if (tap.gain <= 0.0f) {
            continue;
        }
        const size_t delaySamples = static_cast<size_t>(std::max(0.0f, tap.delaySeconds) * static_cast<float>(spec_.freq));
        if (delaySamples > maxDelaySamples) {
            maxDelaySamples = delaySamples;
        }
    }

    std::vector<float> mixed(clickBuffer_.size() + maxDelaySamples + 1, 0.0f);

    // Direct path.
    if (clampedDirect > 0.0f) {
        for (size_t i = 0; i < clickBuffer_.size(); ++i) {
            mixed[i] += clickBuffer_[i] * clampedDirect;
        }
    }

    // First-order reflected paths.
    for (const AudioReflectionTap& tap : reflections) {
        const float gain = std::clamp(tap.gain, 0.0f, 1.0f);
        if (gain <= 0.0f) {
            continue;
        }
        const size_t delaySamples = static_cast<size_t>(std::max(0.0f, tap.delaySeconds) * static_cast<float>(spec_.freq));
        for (size_t i = 0; i < clickBuffer_.size(); ++i) {
            const size_t outIndex = i + delaySamples;
            if (outIndex < mixed.size()) {
                mixed[outIndex] += clickBuffer_[i] * gain;
            }
        }
    }

    for (float& sample : mixed) {
        sample = std::clamp(sample, -1.0f, 1.0f);
    }

    SDL_ClearAudioStream(stream_);

    const int byteCount = static_cast<int>(mixed.size() * sizeof(float));
    if (!SDL_PutAudioStreamData(stream_, mixed.data(), byteCount)) {
        std::printf("Audio queue failed: %s\n", SDL_GetError());
        return;
    }

    SDL_FlushAudioStream(stream_);
}

void AudioManager::buildClickBuffer()
{
    const float durationSeconds = 0.16f;
    const int sampleCount = static_cast<int>(spec_.freq * durationSeconds);
    clickBuffer_.resize(sampleCount);

    for (int i = 0; i < sampleCount; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(spec_.freq);
        const float envelope = std::exp(-28.0f * t);
        const float toneA = std::sinf(2.0f * kPi * 660.0f * t);
        const float toneB = 0.35f * std::sinf(2.0f * kPi * 1320.0f * t);
        clickBuffer_[i] = 0.28f * envelope * (toneA + toneB);
    }
}

AudioManager::~AudioManager()
{
    if (stream_) {
        SDL_DestroyAudioStream(stream_);
        stream_ = nullptr;
    }
}
