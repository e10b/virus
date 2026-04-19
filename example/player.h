#pragma once

//#include <SDL.h>

#include "entity.h"
#include "camera.h"

#include "context.h"


class Player : public Entity
{
public:
	Player();

	void update(float dt);

	const Camera& getCamera() const;


	bool pressed = false;

	glm::ivec3 selectionPos;
private:
	Camera camera_;
	bool canJump_;
	bool noclip_;

	//void keyboardInput();
	//void mouseInput();
	//glm::vec3 velocity_ = glm::vec3(0.0f, 0.0f, 0.0f);

	//glm::vec2 camVel_ = glm::vec2(0.0f);

};