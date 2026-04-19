#pragma once

#include <SDL3/SDL.h>

#include <vector>

struct AudioReflectionTap
{
    float delaySeconds = 0.0f;
    float gain = 0.0f;
};

class AudioManager
{
public:
    static AudioManager& Instance()
    {
        static AudioManager instance;
        return instance;
    }

    bool initialize();
    void playClick();
    void playClickWithVolume(float volume);
    void playRaytracedClick(float directGain, const std::vector<AudioReflectionTap>& reflections);

    ~AudioManager();

private:
    AudioManager() = default;
    AudioManager(const AudioManager&) = delete;
    AudioManager& operator=(const AudioManager&) = delete;

    void buildClickBuffer();

    SDL_AudioStream* stream_ = nullptr;
    SDL_AudioSpec spec_{};
    std::vector<float> clickBuffer_;
};
