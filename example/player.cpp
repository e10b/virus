#include "player.h"

#include <iostream>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#include "input.h"
#include "audio.h"

#include "constants.h"

#include "context.h"

#include "compute/voxel_volume.h"

namespace {
constexpr float kPlayerRadius = 4.0f;
constexpr float kPlayerHeight = 24.0f;
constexpr float kEyeHeight = 21.0f;
constexpr float kWalkSpeed = 48.0f;
constexpr float kSprintMultiplier = 1.75f;
constexpr float kFlySpeed = 36.0f;
constexpr float kFastFlySpeed = 400.0f;
// Target jump apex: ~1 meter = 12 voxels.
// v = sqrt(2gh) with g = 117.6 voxels/s^2 and h = 12 voxels.
constexpr float kJumpSpeed = 53.1263f;
constexpr float kGroundProbeDepth = 0.25f;
constexpr float kMotionStep = 1.0f;
constexpr float kSoundMaxDistance = 192.0f;
const float kSoundSpeedVoxelsPerSecond = 343.0f * World::voxelsPerMeter;

struct AcousticRayHit {
    bool hit = false;
    float distance = 0.0f;
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);
    uint16_t material = 0;
};

float materialAbsorption(uint16_t material)
{
    switch (material) {
    case 1: return 0.52f; // sand
    case 2: return 0.34f; // grass
    case 3: return 0.48f; // dirt
    case 4: return 0.18f; // stone
    case 5: return 0.50f; // snow
    case 6: return 0.24f; // dark rock
    case 7: return 0.38f; // mossy surface
    case VoxelCompute::PLAYER_PLACED_MATERIAL: return 0.32f;
    case VoxelCompute::SOUND_VOXEL_MATERIAL: return 0.22f;
    default: return 0.35f;
    }
}

float distanceAttenuation(float distance)
{
    // Stable attenuation with distance in voxel units.
    return 1.0f / (1.0f + 0.03f * distance + 0.0009f * distance * distance);
}

AcousticRayHit traceRayFirstHit(
    const VoxelCompute& voxelCompute,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float maxDistance)
{
    AcousticRayHit result;

    const glm::vec3 dir = glm::normalize(direction);
    const float step = 0.75f;
    glm::ivec3 previousVoxel = glm::ivec3(glm::floor(origin));

    for (float t = step; t <= maxDistance; t += step) {
        const glm::vec3 p = origin + dir * t;
        const glm::ivec3 voxel = glm::ivec3(glm::floor(p));

        if (voxel == previousVoxel) {
            continue;
        }

        if (voxelCompute.isVoxelOccupied(voxel.x, voxel.y, voxel.z)) {
            result.hit = true;
            result.distance = t;
            result.position = p;
            result.material = voxelCompute.getVoxel(voxel.x, voxel.y, voxel.z);

            const glm::ivec3 delta = voxel - previousVoxel;
            if (delta.x != 0) {
                result.normal = glm::vec3(static_cast<float>(-delta.x), 0.0f, 0.0f);
            } else if (delta.y != 0) {
                result.normal = glm::vec3(0.0f, static_cast<float>(-delta.y), 0.0f);
            } else {
                result.normal = glm::vec3(0.0f, 0.0f, static_cast<float>(-delta.z));
            }
            result.normal = glm::normalize(result.normal);
            return result;
        }

        previousVoxel = voxel;
    }

    return result;
}

bool traceSegmentOccluded(
    const VoxelCompute& voxelCompute,
    const glm::vec3& a,
    const glm::vec3& b,
    uint16_t* blockingMaterial = nullptr)
{
    const glm::vec3 delta = b - a;
    const float distance = glm::length(delta);
    if (distance <= 0.001f) {
        return false;
    }

    const glm::vec3 dir = delta / distance;
    const float step = 0.75f;
    glm::ivec3 previousVoxel = glm::ivec3(glm::floor(a));

    for (float t = step; t < distance; t += step) {
        const glm::vec3 p = a + dir * t;
        const glm::ivec3 voxel = glm::ivec3(glm::floor(p));
        if (voxel == previousVoxel) {
            continue;
        }

        if (voxelCompute.isVoxelOccupied(voxel.x, voxel.y, voxel.z)) {
            if (blockingMaterial) {
                *blockingMaterial = voxelCompute.getVoxel(voxel.x, voxel.y, voxel.z);
            }
            return true;
        }
        previousVoxel = voxel;
    }

    return false;
}

void computeRaytracedAcoustics(
    const VoxelCompute& voxelCompute,
    const glm::vec3& sourcePos,
    const glm::vec3& listenerPos,
    float& directGain,
    std::vector<AudioReflectionTap>& reflections)
{
    reflections.clear();
    directGain = 0.0f;

    const glm::vec3 directVec = listenerPos - sourcePos;
    const float directDistance = glm::length(directVec);
    if (directDistance < 0.001f || directDistance > kSoundMaxDistance) {
        return;
    }

    const float directFalloff = distanceAttenuation(directDistance);
    uint16_t blockingMaterial = 0;
    const bool occluded = traceSegmentOccluded(voxelCompute, sourcePos, listenerPos, &blockingMaterial);

    if (occluded) {
        const float transmission = 1.0f - materialAbsorption(blockingMaterial);
        directGain = directFalloff * transmission * 0.22f;
    } else {
        directGain = directFalloff;
    }

    static const std::array<glm::vec3, 12> kBounceDirs = {
        glm::normalize(glm::vec3(1.0f, 0.2f, 0.0f)),
        glm::normalize(glm::vec3(-1.0f, 0.2f, 0.0f)),
        glm::normalize(glm::vec3(0.0f, 0.2f, 1.0f)),
        glm::normalize(glm::vec3(0.0f, 0.2f, -1.0f)),
        glm::normalize(glm::vec3(1.0f, 0.4f, 1.0f)),
        glm::normalize(glm::vec3(-1.0f, 0.4f, 1.0f)),
        glm::normalize(glm::vec3(1.0f, 0.4f, -1.0f)),
        glm::normalize(glm::vec3(-1.0f, 0.4f, -1.0f)),
        glm::normalize(glm::vec3(1.0f, -0.1f, 0.0f)),
        glm::normalize(glm::vec3(-1.0f, -0.1f, 0.0f)),
        glm::normalize(glm::vec3(0.0f, -0.1f, 1.0f)),
        glm::normalize(glm::vec3(0.0f, -0.1f, -1.0f))
    };

    for (const glm::vec3& dir : kBounceDirs) {
        AcousticRayHit hit = traceRayFirstHit(voxelCompute, sourcePos, dir, kSoundMaxDistance);
        if (!hit.hit) {
            continue;
        }

        const glm::vec3 bouncePoint = hit.position + hit.normal * 0.15f;
        const glm::vec3 toListener = listenerPos - bouncePoint;
        const float segmentDistance = glm::length(toListener);
        if (segmentDistance < 0.001f) {
            continue;
        }

        const glm::vec3 reflectedDir = glm::reflect(dir, hit.normal);
        const float reflectionAlignment = glm::dot(glm::normalize(toListener), glm::normalize(reflectedDir));
        if (reflectionAlignment < 0.72f) {
            continue;
        }

        if (traceSegmentOccluded(voxelCompute, bouncePoint, listenerPos, nullptr)) {
            continue;
        }

        const float totalDistance = hit.distance + segmentDistance;
        if (totalDistance <= directDistance + 1.0f || totalDistance > kSoundMaxDistance * 1.5f) {
            continue;
        }

        const float reflectivity = 1.0f - materialAbsorption(hit.material);
        const float gain = distanceAttenuation(totalDistance) * reflectivity * 0.85f;
        if (gain < 0.015f) {
            continue;
        }

        AudioReflectionTap tap;
        tap.delaySeconds = (totalDistance - directDistance) / kSoundSpeedVoxelsPerSecond;
        tap.gain = gain;
        reflections.push_back(tap);
    }

    std::sort(reflections.begin(), reflections.end(), [](const AudioReflectionTap& a, const AudioReflectionTap& b) {
        return a.gain > b.gain;
    });
    if (reflections.size() > 4) {
        reflections.resize(4);
    }
}

float clampFloat(float value, float minValue, float maxValue)
{
    return glm::clamp(value, minValue, maxValue);
}

bool sphereOverlapsSolidVoxel(const VoxelCompute& voxelCompute, const glm::vec3& center, float radius)
{
    int minX = static_cast<int>(std::floor(center.x - radius));
    int maxX = static_cast<int>(std::floor(center.x + radius));
    int minY = static_cast<int>(std::floor(center.y - radius));
    int maxY = static_cast<int>(std::floor(center.y + radius));
    int minZ = static_cast<int>(std::floor(center.z - radius));
    int maxZ = static_cast<int>(std::floor(center.z + radius));

    float radiusSq = radius * radius;
    for (int z = minZ; z <= maxZ; ++z) {
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                if (!voxelCompute.isVoxelOccupied(x, y, z)) {
                    continue;
                }

                float closestX = clampFloat(center.x, static_cast<float>(x), static_cast<float>(x + 1));
                float closestY = clampFloat(center.y, static_cast<float>(y), static_cast<float>(y + 1));
                float closestZ = clampFloat(center.z, static_cast<float>(z), static_cast<float>(z + 1));
                glm::vec3 delta = center - glm::vec3(closestX, closestY, closestZ);
                if (glm::dot(delta, delta) < radiusSq) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool capsuleOverlapsWorld(const VoxelCompute& voxelCompute, const glm::vec3& feetPosition, const glm::vec3& size)
{
    if (feetPosition.y < 0.0f) {
        return true;
    }

    float radius = size.x * 0.5f;
    float cylinderHeight = glm::max(0.0f, size.y - radius * 2.0f);
    int sampleCount = glm::max(2, static_cast<int>(std::ceil(cylinderHeight / glm::max(radius, 1.0f))) + 1);

    for (int sample = 0; sample < sampleCount; ++sample) {
        float t = sampleCount == 1 ? 0.0f : static_cast<float>(sample) / static_cast<float>(sampleCount - 1);
        glm::vec3 sphereCenter = feetPosition + glm::vec3(0.0f, radius + cylinderHeight * t, 0.0f);
        if (sphereOverlapsSolidVoxel(voxelCompute, sphereCenter, radius)) {
            return true;
        }
    }

    return false;
}

bool moveCapsuleAxis(const VoxelCompute& voxelCompute, glm::vec3& position, const glm::vec3& size, int axis, float delta)
{
    if (delta == 0.0f) {
        return false;
    }

    int steps = glm::max(1, static_cast<int>(std::ceil(std::abs(delta) / kMotionStep)));
    float stepDelta = delta / static_cast<float>(steps);

    for (int step = 0; step < steps; ++step) {
        glm::vec3 candidate = position;
        candidate[axis] += stepDelta;
        if (capsuleOverlapsWorld(voxelCompute, candidate, size)) {
            return true;
        }
        position = candidate;
    }

    return false;
}

bool isGrounded(const VoxelCompute& voxelCompute, const glm::vec3& position, const glm::vec3& size)
{
    return capsuleOverlapsWorld(voxelCompute, position - glm::vec3(0.0f, kGroundProbeDepth, 0.0f), size);
}

glm::vec3 resolveCapsuleSpawn(const VoxelCompute& voxelCompute, glm::vec3 position, const glm::vec3& size)
{
    if (!capsuleOverlapsWorld(voxelCompute, position, size)) {
        return position;
    }

    for (int offset = 1; offset <= 128; ++offset) {
        glm::vec3 candidate = position + glm::vec3(0.0f, static_cast<float>(offset), 0.0f);
        if (!capsuleOverlapsWorld(voxelCompute, candidate, size)) {
            return candidate;
        }
    }

    return position;
}
} // namespace

Player::Player() : Entity(), camera_(getPosition()), canJump_(false), noclip_(true)
{
    setSize(glm::vec3(kPlayerRadius * 2.0f, kPlayerHeight, kPlayerRadius * 2.0f));
    camera_.setFarPlane(World::renderDistance * 1.25f * 3);

    //teleport(glm::vec3(520.5f, 102.0f, -320.5f));
    teleport(glm::vec3(0.0f, 0.0f, 0.0f));
}

void Player::update(float dt)
{
    Input& input = Input::Instance();
    const Uint8* keyboardState = SDL_GetKeyboardState(nullptr);
    float x, y;
    Uint32 mouseButtons = SDL_GetMouseState(&x, &y);

    // close
            //if (input.getKey(Key::Menu)) { Context::Instance().close(); }

    // build
    bool placing = (mouseButtons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    bool destroying = (mouseButtons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
    
    static int toolMode = 1;
    if (keyboardState[SDL_SCANCODE_1]) toolMode = 1;
    if (keyboardState[SDL_SCANCODE_2]) toolMode = 2;
    if (keyboardState[SDL_SCANCODE_3]) toolMode = 3;
    if (keyboardState[SDL_SCANCODE_5]) toolMode = 5;
    if (keyboardState[SDL_SCANCODE_6]) toolMode = 6;
    if (keyboardState[SDL_SCANCODE_7]) toolMode = 7;
    if (keyboardState[SDL_SCANCODE_8]) toolMode = 8;
    if (keyboardState[SDL_SCANCODE_9]) toolMode = 9;

    //Manager::RaycastResult raycast = chunks.raycast(camera_.getPosition(), camera_.getForward(), INFINITY);
    VoxelCompute& voxelCompute = VoxelCompute::Instance();
    
    // Prevent both placing and destroying at the same time
    if (placing && destroying) {
        // Do nothing if both buttons are pressed
    } else if (placing) {
        voxelCompute.raycast(
            camera_.getPosition(),
            camera_.getForward(),
            INFINITY,
            true,
            toolMode,
            true,
            getPosition(),
            getSize());
    } else if (destroying) {
        voxelCompute.raycast(camera_.getPosition(), camera_.getForward(), INFINITY, false, toolMode);
    }

    if (keyboardState[SDL_SCANCODE_G]) {pressed = true;} else {pressed = false;}

    // Test circular buffer with H key
    static bool hKeyPressed = false;
    if (keyboardState[SDL_SCANCODE_H] && !hKeyPressed) {
        voxelCompute.testCircularBuffer();
        hKeyPressed = true;
    } else if (!keyboardState[SDL_SCANCODE_H]) {
        hKeyPressed = false;
    }

    // Toggle walk/fly movement with F
    static bool fKeyPressed = false;
    if (keyboardState[SDL_SCANCODE_F] && !fKeyPressed) {
        noclip_ = !noclip_;
        if (noclip_) {
            setVelocity(glm::vec3(0.0f));
            canJump_ = false;
            printf("Movement mode: fly\n");
        } else {
            glm::vec3 resolvedPosition = resolveCapsuleSpawn(voxelCompute, getPosition(), getSize());
            teleport(resolvedPosition);
            setVelocity(glm::vec3(0.0f));
            canJump_ = isGrounded(voxelCompute, resolvedPosition, getSize());
            printf("Movement mode: walk\n");
        }
        fKeyPressed = true;
    } else if (!keyboardState[SDL_SCANCODE_F]) {
        fKeyPressed = false;
    }

    // Toggle hierarchical visualization with F5 key
    static bool f5KeyPressed = false;
    if (keyboardState[SDL_SCANCODE_F5] && !f5KeyPressed) {
        voxelCompute.showHierarchy = !voxelCompute.showHierarchy;
        f5KeyPressed = true;
    } else if (!keyboardState[SDL_SCANCODE_F5]) {
        f5KeyPressed = false;
    }

    // Toggle sky fog with F3 key
    static bool f3KeyPressed = false;
    if (keyboardState[SDL_SCANCODE_F3] && !f3KeyPressed) {
        voxelCompute.enableSkyFog = !voxelCompute.enableSkyFog;
        printf("Sky fog: %s\n", voxelCompute.enableSkyFog ? "ON" : "OFF");
        f3KeyPressed = true;
    } else if (!keyboardState[SDL_SCANCODE_F3]) {
        f3KeyPressed = false;
    }

    // Play a short UI click when J is pressed.
    static bool jKeyPressed = false;
    if (keyboardState[SDL_SCANCODE_J] && !jKeyPressed) {
        AudioManager::Instance().playClick();
        jKeyPressed = true;
    } else if (!keyboardState[SDL_SCANCODE_J]) {
        jKeyPressed = false;
    }

    // Raytraced acoustics for sound voxels: occlusion + first-bounce reflections.
    static float soundVoxelAudioTimer = 0.0f;
    soundVoxelAudioTimer -= dt;
    if (soundVoxelAudioTimer <= 0.0f && !voxelCompute.soundVoxelPositions.empty()) {
        soundVoxelAudioTimer = 0.16f;

        const glm::vec3 listenerPos = camera_.getPosition();
        const float maxDistSq = kSoundMaxDistance * kSoundMaxDistance;

        float nearestDistSq = std::numeric_limits<float>::max();
        glm::vec3 nearestSource = listenerPos;
        for (const glm::ivec3& voxelPosI : voxelCompute.soundVoxelPositions) {
            const glm::vec3 voxelPos = glm::vec3(voxelPosI) + glm::vec3(0.5f);
            const float distSq = glm::dot(voxelPos - listenerPos, voxelPos - listenerPos);
            if (distSq < nearestDistSq) {
                nearestDistSq = distSq;
                nearestSource = voxelPos;
            }
        }

        const bool hasSourceInRange = nearestDistSq <= maxDistSq;
        if (hasSourceInRange) {
            voxelCompute.setAudioRaytraceRequest(nearestSource, listenerPos, kSoundMaxDistance);
        }

        const VoxelCompute::AudioRaytraceOutput audioOut = voxelCompute.getLatestAudioRaytrace();
        std::vector<AudioReflectionTap> reflections;
        reflections.reserve(static_cast<size_t>(audioOut.reflectionCount));
        for (int i = 0; i < audioOut.reflectionCount; ++i) {
            AudioReflectionTap tap;
            tap.delaySeconds = audioOut.reflectionTaps[static_cast<size_t>(i)].x;
            tap.gain = audioOut.reflectionTaps[static_cast<size_t>(i)].y;
            reflections.push_back(tap);
        }

        // Avoid replaying stale GPU taps when no emitter is currently in range.
        if (hasSourceInRange && (audioOut.directGain > 0.02f || !reflections.empty())) {
            AudioManager::Instance().playRaytracedClick(audioOut.directGain, reflections);
        }
    }

    // Test voxel placement with T key
    static bool tKeyPressed = false;
    if (keyboardState[SDL_SCANCODE_T] && !tKeyPressed) {
        voxelCompute.testVoxelPlacement();
        tKeyPressed = true;
    } else if (!keyboardState[SDL_SCANCODE_T]) {
        tKeyPressed = false;
    }

    // look
    glm::vec2 deltaMouse = input.getDeltaMouse() * 0.003f;
    if (keyboardState[SDL_SCANCODE_V] || (SDL_BUTTON(SDL_BUTTON_LEFT) && mouseButtons))
    {
        camera_.setYaw(camera_.getYaw() - deltaMouse.x);
        camera_.setPitch(camera_.getPitch() - deltaMouse.y);
    }

    if (!noclip_)
    {
        glm::vec3 dir = glm::vec3(0.0f);

        if (keyboardState[SDL_SCANCODE_W])
            dir += camera_.getForwardAligned();
        if (keyboardState[SDL_SCANCODE_S])
            dir -= camera_.getForwardAligned();
        if (keyboardState[SDL_SCANCODE_A])
            dir -= camera_.getRight();
        if (keyboardState[SDL_SCANCODE_D])
            dir += camera_.getRight();

        glm::vec3 velocity = getVelocity();
        if (canJump_ && keyboardState[SDL_SCANCODE_SPACE])
        {
            velocity.y = kJumpSpeed;
            canJump_ = false;
        }

        velocity.y -= World::gravity * dt;

        glm::vec3 position = getPosition();
        float moveSpeed = kWalkSpeed * (keyboardState[SDL_SCANCODE_LSHIFT] ? kSprintMultiplier : 1.0f);
        glm::vec3 horizontalVelocity(0.0f);
        if (dir != glm::vec3(0.0f)) {
            horizontalVelocity = glm::normalize(dir) * moveSpeed;
        }

        moveCapsuleAxis(voxelCompute, position, getSize(), 0, horizontalVelocity.x * dt);
        moveCapsuleAxis(voxelCompute, position, getSize(), 2, horizontalVelocity.z * dt);

        bool hitVerticalSurface = moveCapsuleAxis(voxelCompute, position, getSize(), 1, velocity.y * dt);
        if (hitVerticalSurface) {
            if (velocity.y <= 0.0f) {
                canJump_ = true;
            }
            velocity.y = 0.0f;
        } else {
            canJump_ = false;
        }

        if (velocity.y <= 0.0f && isGrounded(voxelCompute, position, getSize())) {
            canJump_ = true;
            velocity.y = 0.0f;
        }

        teleport(position);
        setVelocity(glm::vec3(0.0f, velocity.y, 0.0f));
    }
    else // noclip
    {
        setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));

        glm::vec3 dir = glm::vec3(0.0f);

        // move
        if (keyboardState[SDL_SCANCODE_W])
            dir += camera_.getForwardAligned();
        if (keyboardState[SDL_SCANCODE_S])
            dir -= camera_.getForwardAligned();
        if (keyboardState[SDL_SCANCODE_A])
            dir -= camera_.getRight();
        if (keyboardState[SDL_SCANCODE_D])
            dir += camera_.getRight();
        if (keyboardState[SDL_SCANCODE_SPACE])
            dir += camera_.getUp();
        if (keyboardState[SDL_SCANCODE_LSHIFT])
            dir -= camera_.getUp();

        if (dir != glm::vec3(0.0f))
            teleport(getPosition() + glm::normalize(dir) * (keyboardState[SDL_SCANCODE_LCTRL] ? kFastFlySpeed : kFlySpeed) * dt);
        //teleport(getPosition() + glm::normalize(dir) * (input.getKey(Key::Run) ? 100.f : 10.f) * dt);
    }

    // update camera
    camera_.setPosition(getPosition() + glm::vec3(0.0f, kEyeHeight, 0.0f));
    camera_.setAspect(16.0f / 9.0f);

    // Debug: Print position every few seconds
    static float posTimer = 0.0f;
    posTimer += dt;
    if (posTimer > 2.0f) {
        glm::vec3 pos = getPosition();
        float distance = glm::length(pos);
        printf("Position: (%.1f, %.1f, %.1f) - Distance from origin: %.1f voxels\n", 
               pos.x, pos.y, pos.z, distance);
        posTimer = 0.0f;
    }
}

const Camera& Player::getCamera() const
{
    return camera_;
}
