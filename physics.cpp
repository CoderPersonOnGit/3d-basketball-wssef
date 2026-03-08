#include "physics.h"
#include <cmath>
#include <algorithm>
#include <limits>

using glm::vec3;

struct State {float x, y, z, vx, vy, vz;};

static State deriv(const State& s, const PhysicsConfig& phys) {
    float k = phys.dragK();
    float speed = std::sqrt(s.vx * s.vx + s.vy * s.vy + s.vz * s.vz);
    return {
        s.vx, s.vy, s.vz,
        -k * speed *s.vx,
        -phys.g - k * speed * s.vy,
        -k * speed * s.vz 
    };
}

static State rk4Step(const State& s, float dt, const PhysicsConfig& phys) {
    auto add = [](const State& a, float h, const State& b) -> State {
        return { a.x + h*b.x, a.y + h*b.y, a.z + h*b.z,
                 a.vx + h*b.vx, a.vy + h*b.vy, a.vz + h*b.vz };
    };
    State k1 = deriv(s, phys);
    State k2 = deriv(add(s, dt*0.5f, k1), phys);
    State k3 = deriv(add(s, dt*0.5f, k2), phys);
    State k4 = deriv(add(s, dt, k3), phys);
    return {
        s.x  + dt/6*(k1.x  + 2*k2.x  + 2*k3.x  + k4.x),
        s.y  + dt/6*(k1.y  + 2*k2.y  + 2*k3.y  + k4.y),
        s.z  + dt/6*(k1.z  + 2*k2.z  + 2*k3.z  + k4.z),
        s.vx + dt/6*(k1.vx + 2*k2.vx + 2*k3.vx + k4.vx),
        s.vy + dt/6*(k1.vy + 2*k2.vy + 2*k3.vy + k4.vy),
        s.vz + dt/6*(k1.vz + 2*k2.vz + 2*k3.vz + k4.vz),
    };
}

std::vector<TrajectoryPoint> simulateDrag(
    float v0, float thetaDeg, float phiDeg,
    float h0,
    const PhysicsConfig& phys,
    float dt, float tMax)
{
    float theta = thetaDeg * 3.14159f / 180.f;
    float phi   = phiDeg   * 3.14159f / 180.f;

    State s { 0, h0, 0,
              v0 * std::cos(theta) * std::cos(phi),
              v0 * std::sin(theta),
              v0 * std::cos(theta) * std::sin(phi) };

    std::vector<TrajectoryPoint> pts;
    float t = 0;
    while (t < tMax && s.y >= 0) {
        float spd = std::sqrt(s.vx*s.vx + s.vy*s.vy + s.vz*s.vz);
        pts.push_back({ vec3(s.x, s.y, s.z), t, spd });
        s = rk4Step(s, dt, phys);
        t += dt;
    }
    return pts;
}

// -----------------------------------------------------------------------
// Optimal angle finder (for given speed, height, distance)
// -----------------------------------------------------------------------

float findOptimalAngle(float v0, float h0, float distForward,
                       float hRim, const PhysicsConfig& phys)
{
    auto miss = [&](float theta) -> float {
        auto traj = simulateDrag(v0, theta, 0.f, h0, phys, 0.005f, 3.0f);
        for (size_t i = 1; i < traj.size(); ++i) {
            if (traj[i].pos.x >= distForward) {
                float t_ = (distForward - traj[i-1].pos.x) /
                           (traj[i].pos.x - traj[i-1].pos.x + 1e-9f);
                float yAt = traj[i-1].pos.y + t_ * (traj[i].pos.y - traj[i-1].pos.y);
                return std::abs(yAt - hRim);
            }
        }
        return 1e6f;
    };

    const float gr = 0.6180339887f;
    float a = 30.f, b = 70.f;
    float c = b - gr * (b - a);
    float d = a + gr * (b - a);

    while (std::abs(b - a) > 0.02f) {
        if (miss(c) < miss(d)) b = d;
        else                   a = c;
        c = b - gr * (b - a);
        d = a + gr * (b - a);
    }

    return (a + b) * 0.5f;
}

float shotQuality(const std::vector<TrajectoryPoint>& traj, float distForward, float hRim) {
    for (size_t i = 1; i < traj.size(); ++i) {
        if (traj[i].pos.x >= distForward) {
            float t_ = (distForward - traj[i-1].pos.x) /
                       (traj[i].pos.x - traj[i-1].pos.x + 1e-9f);
            float yAt = traj[i-1].pos.y + t_ * (traj[i].pos.y - traj[i-1].pos.y);
            float missDist = std::abs(yAt - hRim);
            return std::max(0.f, 1.f - missDist / 0.2286f); // 0.2286m = half ball diameter
        }
    }
    return 0.f;
}

