#pragma once
#include <GL/glew.h>
#include <vector>
#include <glm/glm.hpp>
#include "shader.h"
#include "physics.h"

struct TrailParticle {
    glm::vec3 pos;
    float     alpha;
    float     quality;
};

class Renderer {
public:
    Renderer();
    ~Renderer();

    // Draw the animated ball
    void drawBall(const Shader& sh, const glm::mat4& view, const glm::mat4& proj,
                  const glm::vec3& pos, const glm::vec3& lightPos, const glm::vec3& viewPos) const;

    // Update + draw trail
    void updateTrail(const std::vector<TrajectoryPoint>& traj, float quality);
    void drawTrail(const Shader& sh, const glm::mat4& view, const glm::mat4& proj) const;

    // Draw optimal arc as line strip (ghost arc)
    void setOptimalArc(const std::vector<TrajectoryPoint>& traj);
    void drawOptimalArc(const Shader& sh, const glm::mat4& view, const glm::mat4& proj) const;

private:
    // Sphere mesh
    GLuint sphereVAO_, sphereVBO_, sphereEBO_;
    int    sphereIdxCount_;
    void   buildSphere(int stacks = 32, int slices = 32);

    // Trail
    GLuint trailVAO_, trailVBO_;
    int    trailCount_;

    // Optimal arc
    GLuint arcVAO_, arcVBO_;
    int    arcCount_;
};