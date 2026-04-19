#include "entity.h"
#include "constants.h"

Entity::Entity() : position_(glm::vec3(0.0f, 0.0f, 0.0f)), velocity_(glm::vec3(0.0f, 0.0f, 0.0f)), size_(glm::vec3(0.8f, 1.8f, 0.8f))
{
}

void Entity::update(float dt)
{
	velocity_ -= glm::vec3(0.0f, World::gravity, 0.0f) * dt;

}

glm::vec3 Entity::getPosition() const
{
	return position_;
}

glm::vec3 Entity::getVelocity() const
{
	return velocity_;
}

glm::vec3 Entity::getSize() const
{
	return size_;
}


void Entity::teleport(glm::vec3 destination)
{
	position_ = destination;
}

void Entity::setVelocity(glm::vec3 vel)
{
	velocity_ = vel;
}

void Entity::setSize(glm::vec3 size)
{
	size_ = size;
}

