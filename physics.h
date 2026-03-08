#pragma once
#include <vector>
#include <glm/glm.hpp>

struct PhysicsConfig {
    float g         = 9.81f;
    float rho       = 1.225f;    
    float Cd        = 0.47f;     
    float ballR     = 0.119f;    
    float ballMass  = 0.623f;    

    float dragK() const {
        return 0.5f * rho * Cd * 3.14159f * ballR * ballR / ballMass;
    }
};

struct TrajectoryPoint {
    glm::vec3 pos;
    float     t;     
    float     speed;
};


std::vector<TrajectoryPoint> simulateDrag(
    float v0, float thetaDeg, float phiDeg,   
    float h0,
    const PhysicsConfig& phys,
    float dt = 0.005f,
    float tMax = 3.0f
);

float findOptimalAngle(
    float v0, float h0,
    float distForward, 
    float hRim,          
    const PhysicsConfig& phys
);

float shotQuality(
    const std::vector<TrajectoryPoint>& traj,
    float distForward,
    float hRim
);