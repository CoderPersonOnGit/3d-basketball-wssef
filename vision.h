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

// Common shot distances (metres)
static constexpr float DIST_LAYUP_M      = 1.5f;
static constexpr float DIST_FREE_THROW_M = 4.6f;
static constexpr float DIST_THREE_PT_M   = 6.75f;

// -----------------------------------------------------------------------
// Shot outcome
// -----------------------------------------------------------------------
enum class ShotResult {
    None,
    InFlight,
    Made,
    Missed
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
    float       distToRimM      = 0.0f;
    float       launchSpeedMs   = 8.0f;
    float       launchAngleDeg  = 52.0f;
    bool        hasValidShot    = false;
    bool        shootingRight   = true;

    ShotResult  shotResult      = ShotResult::None;
    int         madeCount       = 0;
    int         attemptCount    = 0;

    float       shotQuality     = 0.f;
    float       manualDistM     = 0.f;
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
    void setManualDistance(float distM) { scene_.manualDistM = distM; }

    // HSV tuning
    cv::Scalar ballLowerHSV { 5,  100, 100 };
    cv::Scalar ballUpperHSV { 25, 255, 255 };
    float      minBallRadius = 8.f;
    float      maxBallRadius = 35.f;

private:
    cv::VideoCapture cap_;
    cv::Mat          rawFrame_;
    cv::Mat          displayFrame_;
    SceneData        scene_;
    CameraCalib      calib_;

    struct MPImpl;
    MPImpl* mp_ = nullptr;

    // Ball trajectory buffer
    std::deque<BallSample>     ballHistory_;
    static constexpr int       MAX_HISTORY    = 90;
    static constexpr float     HISTORY_WINDOW = 2.0f;

    // Optimal arc cache
    std::vector<glm::vec2>     optimalArcPx_;
    float                      lastOptAngle_  = 0.f;
    float                      lastV0_        = 0.f;
    float                      lastDistM_     = 0.f;

    // Shot state machine
    bool  ballWasNearRim_     = false;
    bool  shotInFlight_       = false;
    float shotStartTime_      = 0.f;
    float appStartTime_       = 0.f;
    float resultDisplayTimer_ = 0.f;
    float lastShotClearTime_  = -999.f;

    // Frame timer
    float prevFrameTime_    = 0.f;
    float fps_              = 30.f;

    // Private methods
    std::optional<glm::vec2> detectBall(const cv::Mat& frame);
    PoseData                 detectPose(const cv::Mat& frame);
    void                     computeWorldCoords();
    void                     estimateLaunchParams();
    void                     detectShotDirection();
    void                     updateShotStateMachine();
    void                     computeShotQuality();
    void                     drawOverlay();
    void                     drawTrajectoryComparison();
    void                     drawShotResult();
    void                     drawStats();
    void                     drawAngleGauge();
    void                     drawShotQuality();
    void                     projectAndDrawArc();

    std::vector<glm::vec2>   projectArcToPixels(
        float v0, float thetaDeg, float h0M, float distM, bool shootRight);

    float                    nowSec() const;
    bool                     ballNearRim() const;
    bool                     ballPassedThroughRim() const;
};
