#pragma once

#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <optional>
#include <deque>
#include <chrono>

// NBA court constants (metres)
static constexpr float COURT_RIM_HEIGHT_M  = 3.05f;
static constexpr float COURT_RIM_RADIUS_M  = 0.2286f;
static constexpr float AVG_PERSON_HEIGHT_M = 1.80f;

// -----------------------------------------------------------------------
// Shot outcome
// -----------------------------------------------------------------------
enum class ShotResult {
    None,       // no shot in progress
    InFlight,   // ball is in the air
    Made,       // ball passed through rim
    Missed      // ball hit rim or missed entirely
};

// -----------------------------------------------------------------------
// Pose landmarks (pixel coords)
// -----------------------------------------------------------------------
struct PoseData {
    glm::vec2 wristRight    = {0, 0};
    glm::vec2 wristLeft     = {0, 0};
    glm::vec2 ankleRight    = {0, 0};
    glm::vec2 ankleLeft     = {0, 0};
    glm::vec2 shoulderRight = {0, 0};
    glm::vec2 shoulderLeft  = {0, 0};
    bool      valid         = false;
};

// -----------------------------------------------------------------------
// Ball trajectory sample (pixel space + timestamp)
// -----------------------------------------------------------------------
struct BallSample {
    glm::vec2 posPx;
    float     timeSec;
};

// -----------------------------------------------------------------------
// Full scene state for one frame
// -----------------------------------------------------------------------
struct SceneData {
    std::optional<glm::vec2> ballCenter;
    float                    ballRadius      = 0.f;
    PoseData                 pose;

    float       releaseHeightM  = 1.8f;
    float       distToRimM      = 6.0f;
    float       launchSpeedMs   = 8.0f;     // NEW: estimated launch speed
    float       launchAngleDeg  = 52.0f;    // NEW: estimated launch angle
    bool        hasValidShot    = false;
    bool        shootingRight   = true;     // NEW: direction of shot

    ShotResult  shotResult      = ShotResult::None;
    int         madeCount       = 0;
    int         attemptCount    = 0;
};

// -----------------------------------------------------------------------
// Camera calibration
// -----------------------------------------------------------------------
struct CameraCalib {
    float rimXpx      = -1.f;
    float rimYpx      = -1.f;
    float floorYpx    = -1.f;
    float metersPerPx = -1.f;
    int   frameW      = 1280;
    int   frameH      = 720;

    bool isCalibrated() const { return metersPerPx > 0.f && rimXpx >= 0.f; }

    glm::vec2 pixelToMetres(float px, float py) const {
        return { (px - rimXpx) * metersPerPx,
                 (floorYpx - py) * metersPerPx };
    }
};

// -----------------------------------------------------------------------
// Main vision system
// -----------------------------------------------------------------------
class VisionSystem {
public:
    VisionSystem();
    ~VisionSystem();

    bool openCamera(int deviceIndex = 0);
    bool openCamera(const std::string& url);
    void close();

    bool processFrame();

    const cv::Mat&   getFrame()     const { return displayFrame_; }
    const SceneData& getSceneData() const { return scene_; }
    CameraCalib&     getCalib()           { return calib_; }

    void setRimPixel(float x, float y);

    // HSV tuning
    cv::Scalar ballLowerHSV { 5,  100, 100 };
    cv::Scalar ballUpperHSV { 25, 255, 255 };
    float      minBallRadius = 8.f;
    float      maxBallRadius = 80.f;

private:
    cv::VideoCapture cap_;
    cv::Mat          rawFrame_;
    cv::Mat          displayFrame_;
    SceneData        scene_;
    CameraCalib      calib_;

    // MediaPipe
    struct MPImpl;
    MPImpl* mp_ = nullptr;

    // Ball trajectory buffer (recent samples for speed estimation + comparison)
    std::deque<BallSample>     ballHistory_;
    static constexpr int       MAX_HISTORY    = 90;   // ~3s at 30fps
    static constexpr float     HISTORY_WINDOW = 2.0f; // seconds kept

    // Optimal arc (pixel space) — recomputed when scene changes
    std::vector<glm::vec2>     optimalArcPx_;
    float                      lastOptAngle_  = 0.f;
    float                      lastV0_        = 0.f;

    // Shot state machine
    bool  ballWasNearRim_   = false;
    bool  shotInFlight_     = false;
    float shotStartTime_    = 0.f;
    float appStartTime_     = 0.f;
    float resultDisplayTimer_ = 0.f;

    // Frame timer
    float prevFrameTime_    = 0.f;
    float fps_              = 30.f;

    // Private methods
    std::optional<glm::vec2> detectBall(const cv::Mat& frame);
    PoseData                 detectPose(const cv::Mat& frame);
    void                     computeWorldCoords();
    void                     estimateLaunchParams();       // NEW
    void                     detectShotDirection();        // NEW
    void                     updateShotStateMachine();     // NEW
    void                     drawOverlay();
    void                     drawTrajectoryComparison();   // NEW
    void                     drawShotResult();             // NEW
    void                     drawStats();                  // NEW
    void                     projectAndDrawArc();

    std::vector<glm::vec2>   projectArcToPixels(
        float v0, float thetaDeg, float h0M, float distM, bool shootRight);

    float                    nowSec() const;
    bool                     ballNearRim() const;
    bool                     ballPassedThroughRim() const;
};

