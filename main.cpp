#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

#include "shader.h"
#include "physics.h"
#include "renderer.h"
#include "camera.h"
#include "court.h"

static const int WINDOW_WIDTH = 1280;
static const int WINDOW_HEIGHT = 720;

static Camera cam;
static bool mouseDown = false;
static double lastX = 0.0, lastY = 0.0;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouseDown = true;
            glfwGetCursorPos(window, &lastX, &lastY);
        } else if (action == GLFW_RELEASE) {
            mouseDown = false;
        }
    }
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (mouseDown) {
        float xoffset = (float)(xpos - lastX);
        float yoffset = (float)(lastY - ypos);
        lastX = xpos;
        lastY = ypos;
        cam.orbit(xoffset, yoffset);
    }
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    cam.zoom((float)yoffset);
}

static GLFWwindow* initGLFW() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        "Basketball 3D Trajectory Visualizer", nullptr, nullptr
    );

    if (!window) {
        std::cerr << "GLFW window creation failed\n";
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    return window;
}

struct shotState {
    float v0    = 8.5f;
    float theta = 52.f;
    float phi   = 0.f;
    float h0    = 1.8f;
};

static shotState shot;
static bool needRecompute = true;

static std::vector<TrajectoryPoint> currentTraj;
static std::vector<TrajectoryPoint> optimalTraj;
static float shotQualityVal = 0.f;
static float optAngle = 0.f;

static float animT = 0.f;
static bool animating = false;
static float animSpeed = 0.5f;

static PhysicsConfig phys;

void Recompute() {
    currentTraj = simulateDrag(shot.v0, shot.theta, shot.phi, shot.h0, phys);
    optAngle = findOptimalAngle(shot.v0, shot.h0, Court::DIST_TO_RIM, Court::RIM_HEIGHT, phys);
    optimalTraj = simulateDrag(shot.v0, optAngle, 0.f, shot.h0, phys);
    shotQualityVal = shotQuality(currentTraj, Court::DIST_TO_RIM, Court::RIM_HEIGHT);
    animT = 0.f;
    animating = true;
    needRecompute = false;

    std::cout << "\n=== Shot ===\n"
              << "  Launch speed  : " << shot.v0                    << " m/s\n"
              << "  Launch angle  : " << shot.theta                  << " deg\n"
              << "  Lateral angle : " << shot.phi                    << " deg\n"
              << "  Release height: " << shot.h0                    << " m\n"
              << "  Optimal angle : " << optAngle                    << " deg\n"
              << "  Shot quality  : " << (int)(shotQualityVal * 100) << " %\n";
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, true);
                break;
            case GLFW_KEY_SPACE:
                shot.v0    = 6.f + (float)rand() / RAND_MAX * 5.f;
                shot.theta = 35.f + (float)rand() / RAND_MAX * 30.f;
                shot.phi   = -10.f + (float)rand() / RAND_MAX * 20.f;
                needRecompute = true;
                break;
            case GLFW_KEY_R:
                shot = shotState{};
                needRecompute = true;
                break;
            case GLFW_KEY_UP:
                shot.theta = std::min(75.f, shot.theta + 1.f);
                needRecompute = true;
                break;
            case GLFW_KEY_DOWN:
                shot.theta = std::max(20.f, shot.theta - 1.f);
                needRecompute = true;
                break;
            case GLFW_KEY_RIGHT:
                shot.v0 = std::min(14.f, shot.v0 + 0.2f);
                needRecompute = true;
                break;
            case GLFW_KEY_LEFT:
                shot.v0 = std::max(4.f, shot.v0 - 0.2f);
                needRecompute = true;
                break;
            case GLFW_KEY_W:
                shot.phi = std::min(45.f, shot.phi + 1.f);
                needRecompute = true;
                break;
            case GLFW_KEY_S:
                shot.phi = std::max(-45.f, shot.phi - 1.f);
                needRecompute = true;
                break;
            case GLFW_KEY_E:
                shot.h0 = std::min(2.5f, shot.h0 + 0.1f);
                needRecompute = true;
                break;
            case GLFW_KEY_D:
                shot.h0 = std::max(1.2f, shot.h0 - 0.1f);
                needRecompute = true;
                break;
        }
    }
}

int main() {
    GLFWwindow* window = initGLFW();
    if (!window) return -1;

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    Shader ballShader  ("shaders/ball.vert",  "shaders/ball.frag");
    Shader trailShader ("shaders/trail.vert", "shaders/trail.frag");
    Shader courtShader ("shaders/court.vert", "shaders/court.frag");

    Renderer renderer;
    Court    court;

    glfwSetKeyCallback(window, key_callback);

    glm::vec3 lightPos = { 3.f, 8.f, 3.f };

    Recompute();
    renderer.setOptimalArc(optimalTraj);
    renderer.updateTrail(currentTraj, shotQualityVal);

    float prevTime = (float)glfwGetTime();

    std::cout << "\nControls:\n"
              << "  Mouse drag     - orbit camera\n"
              << "  Scroll         - zoom\n"
              << "  SPACE          - random shot\n"
              << "  R              - reset\n"
              << "  UP / DOWN      - angle +/-\n"
              << "  LEFT / RIGHT   - speed +/-\n"
              << "  W / S          - lateral angle +/-\n"
              << "  E / D          - release height +/-\n"
              << "  ESC            - quit\n\n";

    while (!glfwWindowShouldClose(window)) {
        float now = (float)glfwGetTime();
        float dt  = now - prevTime;
        prevTime  = now;

        glfwPollEvents();

        if (needRecompute) {
            Recompute();
            renderer.setOptimalArc(optimalTraj);
            renderer.updateTrail(currentTraj, shotQualityVal);
        }

        if (animating && !currentTraj.empty()) {
            animT += dt * animSpeed;
            float maxT = currentTraj.back().t;
            if (animT > maxT) {
                animT     = maxT;
                animating = false;
            }
        }

        glm::vec3 ballPos = currentTraj.empty() ? glm::vec3(0) : currentTraj[0].pos;
        if (!currentTraj.empty()) {
            for (size_t i = 1; i < currentTraj.size(); ++i) {
                if (currentTraj[i].t >= animT) {
                    float tt = (animT - currentTraj[i-1].t) /
                               (currentTraj[i].t - currentTraj[i-1].t + 1e-9f);
                    ballPos = glm::mix(currentTraj[i-1].pos, currentTraj[i].pos, tt);
                    break;
                }
            }
        }

        glClearColor(0.04f, 0.04f, 0.08f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        float aspect = (float)fbW / (float)(fbH ? fbH : 1);

        glm::mat4 view = cam.view();
        glm::mat4 proj = cam.projection(aspect);

        court.draw(courtShader, view, proj);
        renderer.drawOptimalArc(courtShader, view, proj);
        renderer.drawTrail(trailShader, view, proj);
        renderer.drawBall(ballShader, view, proj, ballPos, lightPos, cam.position());

        std::ostringstream title;
        title << std::fixed << std::setprecision(1)
              << "Basketball 3D  |  v0=" << shot.v0    << " m/s"
              << "  theta="              << shot.theta  << " deg"
              << "  phi="               << shot.phi    << " deg"
              << "  h0="                << shot.h0     << " m"
              << "  optimal="           << optAngle    << " deg"
              << "  quality="           << (int)(shotQualityVal * 100) << "%"
              << "  [SPACE=random  UP/DN=angle  LT/RT=speed  W/S=lateral  E/D=height  R=reset]";
        glfwSetWindowTitle(window, title.str().c_str());

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}