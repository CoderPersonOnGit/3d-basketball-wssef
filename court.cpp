#include <court.h>
#include <cmath>

static const float PI = 3.14159265f;

Court::Mesh Court::buildLines(const std::vector<glm::vec3>& pts, GLenum mode, glm::vec4 col) {
    Mesh m;
    m.mode  = mode;
    m.color = col;
    m.count = (int)pts.size();
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(glm::vec3), pts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    return m;
}

void Court::addFloor() {
    std::vector<glm::vec3> pts;
    float half = 7.f;
    for (int i = -7; i <= 7; ++i) {
        pts.push_back({(float)i, 0, -half});
        pts.push_back({(float)i, 0,  half});
        pts.push_back({-half, 0, (float)i});
        pts.push_back({ half, 0, (float)i});
    }                                                        
    meshes_.push_back(buildLines(pts, GL_LINES, {0.25f, 0.18f, 0.10f, 1.f}));

    std::vector<glm::vec3> outline = {
        {-1.f,0,-5.f},{-1.f,0,5.f},{10.f,0,5.f},{10.f,0,-5.f},{-1.f,0,-5.f}
    };
    meshes_.push_back(buildLines(outline, GL_LINE_STRIP, {0.9f, 0.7f, 0.3f, 1.f}));
}                                                              

void Court::addRim() {

    std::vector<glm::vec3> rim;
    int N = 64;
    for (int i = 0; i < N; ++i) {
        float a = 2*PI * i / N;
        rim.push_back({ RIM_X + RIM_RADIUS * std::cos(a),
        RIM_HEIGHT,
        RIM_RADIUS * std::sin(a) });  
    }
    meshes_.push_back(buildLines(rim, GL_LINE_STRIP, {0.9f, 0.3f, 0.05f, 1.f}));

    for (int i = 0; i < 12; ++i) {
        float a = 2*PI * i / 12;
        float cx = RIM_X + RIM_RADIUS * std::cos(a);
        float cz = RIM_RADIUS * std::sin(a);
        // converge to narrower point below
        float bx = RIM_X + RIM_RADIUS * 0.4f * std::cos(a);
        float bz = RIM_RADIUS * 0.4f * std::sin(a);
        std::vector<glm::vec3> strand = {
            {cx, RIM_HEIGHT, cz},
            {(cx+bx)*0.5f, RIM_HEIGHT - 0.2f, (cz+bz)*0.5f},
            {bx, RIM_HEIGHT - 0.45f, bz}
        };
        meshes_.push_back(buildLines(strand, GL_LINE_STRIP, {0.9f, 0.9f, 0.9f, 0.6f}));
    }
}

void Court::addBackboard() {
    float bx = RIM_X + 0.15f;
    float by = RIM_HEIGHT;
    
    std::vector<glm::vec3> bb = {
        {bx, by - 0.3f, -0.9f}, {bx, by + 0.6f, -0.9f},
        {bx, by + 0.6f,  0.9f}, {bx, by - 0.3f,  0.9f},
        {bx, by - 0.3f, -0.9f}
    };
    meshes_.push_back(buildLines(bb, GL_LINE_STRIP, {0.8f, 0.8f, 0.9f, 0.9f}));

    std::vector<glm::vec3> inner = {
        {bx, by,        -0.3f}, {bx, by + 0.45f, -0.3f},
        {bx, by + 0.45f, 0.3f}, {bx, by,          0.3f},
        {bx, by,        -0.3f}
    };
    meshes_.push_back(buildLines(inner, GL_LINE_STRIP, {0.8f, 0.8f, 0.9f, 0.7f}));
}

void Court::addPole() {
    std::vector<glm::vec3> pole = {
        {RIM_X + 0.4f, 0.f, 0.f},
        {RIM_X + 0.4f, RIM_HEIGHT, 0.f},
        {RIM_X + 0.15f, RIM_HEIGHT, 0.f}   
    };
    meshes_.push_back(buildLines(pole, GL_LINE_STRIP, {0.7f, 0.7f, 0.7f, 1.f}));
}

void Court::add3ptArc() {
    std::vector<glm::vec3> arc;
    float r3 = 6.75f; 
    for (int i = 0; i <= 80; ++i) {
        float a = (-40.f + i) * PI / 180.f;
        arc.push_back({ RIM_X - r3 * std::cos(a), 0.f, r3 * std::sin(a) });
    }
    meshes_.push_back(buildLines(arc, GL_LINE_STRIP, {0.5f, 0.5f, 0.9f, 0.8f}));
}

Court::Court() { buildCourt(); }

void Court::buildCourt() {
    addFloor();
    addRim();
    addBackboard();
    addPole();
    add3ptArc();
}

Court::~Court() {
for (auto& m : meshes_) {
    glDeleteVertexArrays(1, &m.vao);
    glDeleteBuffers(1, &m.vbo);
    }
}


void Court::draw(const Shader& shader, const glm::mat4& view, const glm::mat4& proj) const {
    shader.use();
    shader.setMat4("view", view);
    shader.setMat4("projection", proj);
    for (auto& m : meshes_) {
        shader.setVec4("lineColor", m.color);
        glBindVertexArray(m.vao);
        glDrawArrays(m.mode, 0, m.count);
    }
    glBindVertexArray(0);
}

