#pragma once
#include <GL/glew.h>
#include <vector>
#include <glm/glm.hpp>
#include "shader.h"


class Court {
public:
    Court();
    ~Court();
    void draw(const Shader& shader, const glm::mat4& view, const glm::mat4& proj) const;

  
    static constexpr float RIM_HEIGHT   = 3.05f;
    static constexpr float RIM_RADIUS   = 0.2286f;
    static constexpr float DIST_TO_RIM  = 6.4f;   
    static constexpr float SHOOTER_X    = 0.f;
    static constexpr float RIM_X        = SHOOTER_X + DIST_TO_RIM;

private:
    struct Mesh { GLuint vao, vbo; int count; GLenum mode; glm::vec4 color; };

    std::vector<Mesh> meshes_;

    Mesh buildLines(const std::vector<glm::vec3>& pts, GLenum mode, glm::vec4 col);
    void buildCourt();
    void addFloor();
    void addRim();
    void addBackboard();
    void addPole();
    void add3ptArc();
};