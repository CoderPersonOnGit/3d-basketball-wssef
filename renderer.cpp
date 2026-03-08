#include <renderer.h>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

static const float PI = 3.14159265f;

//-------Sphere-------
void Renderer::buildSphere(int stacks, int slices) {
     struct Vertex { glm::vec3 pos, normal; };
    std::vector<Vertex>       verts;
    std::vector<unsigned int> indices;

    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / stacks;
        float phi = v * PI;

        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / slices;
            float theta = u * 2.f * PI;

            glm::vec3 pos{
                std::sin(phi) * std::cos(theta),
                std::cos(phi),
                std::sin(phi) * std::sin(theta)
            };
            verts.push_back({ pos, pos });
        }
    }

    for (int i = 0; i < stacks; ++i)
        for (int j = 0; j < slices; ++j) {
            int r = i * (slices+1) + j;
            indices.insert(indices.end(),
                { (unsigned)r, (unsigned)(r+slices+1), (unsigned)(r+1),
                  (unsigned)(r+1), (unsigned)(r+slices+1), (unsigned)(r+slices+2) });
        }
    sphereIdxCount_ = (unsigned)indices.size();
    glGenVertexArrays(1, &sphereVAO_);
    glGenBuffers(1, &sphereVBO_);
    glGenBuffers(1, &sphereEBO_);
    glBindVertexArray(sphereVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO_);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

Renderer::Renderer() : trailCount_(0), arcCount_(0) {
    buildSphere();
    
    glGenVertexArrays(1, &trailVAO_);
    glGenBuffers(1, &trailVBO_);
    glBindVertexArray(trailVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, trailVBO_);
    glBufferData(GL_ARRAY_BUFFER, 4000 * sizeof(TrailParticle), nullptr, GL_DYNAMIC_DRAW);
    // pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TrailParticle), (void*)0);
    glEnableVertexAttribArray(0);
    // alpha
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(TrailParticle),
                          (void*)offsetof(TrailParticle, alpha));
    glEnableVertexAttribArray(1);
    // quality
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(TrailParticle),
                          (void*)offsetof(TrailParticle, quality));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    glGenVertexArrays(1, &arcVAO_);
    glGenBuffers(1, &arcVBO_);
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &sphereVAO_);
    glDeleteBuffers(1, &sphereVBO_);
    glDeleteBuffers(1, &sphereEBO_);
    glDeleteVertexArrays(1, &trailVAO_);
    glDeleteBuffers(1, &trailVBO_);
    glDeleteVertexArrays(1, &arcVAO_);
    glDeleteBuffers(1, &arcVBO_);
}

void Renderer::drawBall(const Shader& sh, const glm::mat4& view, const glm::mat4& proj,
const glm::vec3& pos, const glm::vec3& lightPos,
const glm::vec3& viewPos) const

{
    sh.use();
    glm::mat4 model = glm::translate(glm::mat4(1.f), pos);
    sh.setMat4("model", model);
    sh.setMat4("view", view);
    sh.setMat4("projection", proj);
    sh.setVec3("lightPos", lightPos);
    sh.setVec3("viewPos", viewPos);
    glBindVertexArray(sphereVAO_);
    glDrawElements(GL_TRIANGLES, sphereIdxCount_, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Renderer::updateTrail(const std::vector<TrajectoryPoint>& traj, float quality) {
    std::vector<TrailParticle> pts;
    int N = (int)traj.size();
    for (int i = 0; i < N; ++i) {
        float a = (float)(i+1) / N;
        pts.push_back({ traj[i].pos, a, quality });
    }
    trailCount_ = (int)pts.size();
    glBindVertexArray(trailVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, trailVBO_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, pts.size() * sizeof(TrailParticle), pts.data());
    glBindVertexArray(0);
}

void Renderer::drawTrail(const Shader& sh, const glm::mat4& view, const glm::mat4& proj) const {
    if (trailCount_ == 0) return;
    sh.use();
    sh.setMat4("view",       view);
    sh.setMat4("projection", proj);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(trailVAO_);
    glDrawArrays(GL_POINTS, 0, trailCount_);
    glBindVertexArray(0);
}

void Renderer::setOptimalArc(const std::vector<TrajectoryPoint>& traj) {
    std::vector<glm::vec3> pts;
    for (auto& p : traj) pts.push_back(p.pos);
    arcCount_ = (int)pts.size();

    glBindVertexArray(arcVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, arcVBO_);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(glm::vec3), pts.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void Renderer::drawOptimalArc(const Shader& sh, const glm::mat4& view, const glm::mat4& proj) const {
    if (arcCount_ < 2) return;
    sh.use();
    sh.setMat4("view",       view);
    sh.setMat4("projection", proj);
    sh.setVec4("lineColor",  {0.2f, 0.8f, 1.0f, 0.7f}); // cyan ghost arc
    glLineWidth(2.f);
    glBindVertexArray(arcVAO_);
    glDrawArrays(GL_LINE_STRIP, 0, arcCount_);
    glBindVertexArray(0);
}