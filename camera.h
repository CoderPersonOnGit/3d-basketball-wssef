#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    glm::vec3 target  = {3.f, 1.5f, 0.f};
    float     radius  = 12.f;
    float     yaw     = -30.f;   // degrees
    float     pitch   = 20.f;    // degrees

    glm::vec3 position() const {
        float y   = glm::radians(yaw);
        float p   = glm::radians(pitch);
        return target + radius * glm::vec3(
            std::cos(p) * std::sin(y),
            std::sin(p),
            std::cos(p) * std::cos(y)
        );
    }

    glm::mat4 view() const {
        return glm::lookAt(position(), target, {0,1,0});
    }

    glm::mat4 projection(float aspect) const {
        return glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
    }

    void orbit(float dx, float dy) {
        yaw   += dx * 0.4f;
        pitch += dy * 0.4f;
        pitch  = glm::clamp(pitch, -89.f, 89.f);
    }

    void zoom(float delta) {
        radius -= delta * 0.5f;
        radius  = glm::clamp(radius, 2.f, 40.f);
    }
};