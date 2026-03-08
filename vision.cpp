#include "vision.h"
#include "physics.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

#ifndef NO_MEDIAPIPE
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#endif

static const float PI = 3.14159265f;

static constexpr int MP_RIGHT_SHOULDER = 12;
static constexpr int MP_LEFT_SHOULDER  = 11;
static constexpr int MP_RIGHT_WRIST    = 16;
static constexpr int MP_LEFT_WRIST     = 15;
static constexpr int MP_RIGHT_ANKLE    = 28;
static constexpr int MP_LEFT_ANKLE     = 27;

#ifndef NO_MEDIAPIPE
static const char* POSE_GRAPH = R"(
input_stream: "input_video"
output_stream: "pose_landmarks"
node {
  calculator: "PoseLandmarkCpu"
  input_stream: "IMAGE:input_video"
  output_stream: "LANDMARKS:pose_landmarks"
  node_options: {
    [type.googleapis.com/mediapipe.PoseLandmarkCpuOptions] {
      model_complexity: 0
    }
  }
}
)";
struct VisionSystem::MPImpl {
    mediapipe::CalculatorGraph graph;
    mediapipe::OutputStreamPoller* poller = nullptr;
    bool running = false;
};
#else
struct VisionSystem::MPImpl {};
#endif

// -----------------------------------------------------------------------
// Construction / destruction
// -----------------------------------------------------------------------

VisionSystem::VisionSystem() {
    mp_ = new MPImpl();
    appStartTime_ = (float)(cv::getTickCount() / cv::getTickFrequency());
    prevFrameTime_ = appStartTime_;

#ifndef NO_MEDIAPIPE
    auto config = mediapipe::ParseTextProtoOrDie<
        mediapipe::CalculatorGraphConfig>(POSE_GRAPH);
    mp_->graph.Initialize(config);
    auto sp = mp_->graph.AddOutputStreamPoller("pose_landmarks");
    if (sp.ok())
        mp_->poller = new mediapipe::OutputStreamPoller(std::move(sp.value()));
    mp_->graph.StartRun({});
    mp_->running = true;
    std::cout << "[Vision] MediaPipe enabled.\n";
#else
    std::cout << "[Vision] Running without MediaPipe (ball tracking only).\n";
#endif
}

VisionSystem::~VisionSystem() {
    close();
#ifndef NO_MEDIAPIPE
    if (mp_ && mp_->running) {
        mp_->graph.CloseInputStream("input_video");
        mp_->graph.WaitUntilDone();
        delete mp_->poller;
    }
#endif
    delete mp_;
}

// -----------------------------------------------------------------------
// Camera open / close
// -----------------------------------------------------------------------

bool VisionSystem::openCamera(int idx) {
    cap_.open(idx);
    if (!cap_.isOpened()) { std::cerr << "[Vision] Cannot open camera " << idx << "\n"; return false; }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    calib_.frameW = 1280; calib_.frameH = 720;
    std::cout << "[Vision] Camera opened (index=" << idx << ")\n";
    return true;
}

bool VisionSystem::openCamera(const std::string& url) {
    cap_.open(url);
    if (!cap_.isOpened()) { std::cerr << "[Vision] Cannot open: " << url << "\n"; return false; }
    calib_.frameW = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    calib_.frameH = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "[Vision] Stream opened: " << url << "\n";
    return true;
}

void VisionSystem::close() { if (cap_.isOpened()) cap_.release(); }

// -----------------------------------------------------------------------
// Main frame processing
// -----------------------------------------------------------------------

bool VisionSystem::processFrame() {
    if (!cap_.isOpened()) return false;
    cap_ >> rawFrame_;
    if (rawFrame_.empty()) return false;

    float now = nowSec();
    fps_ = 1.f / std::max(now - prevFrameTime_, 0.001f);
    prevFrameTime_ = now;

    displayFrame_     = rawFrame_.clone();
    scene_.ballCenter = detectBall(rawFrame_);
    scene_.pose       = detectPose(rawFrame_);

    if (scene_.pose.valid)  computeWorldCoords();
    detectShotDirection();

    // Update ball history
    if (scene_.ballCenter) {
        ballHistory_.push_back({ *scene_.ballCenter, now });
        while (!ballHistory_.empty() &&
               now - ballHistory_.front().timeSec > HISTORY_WINDOW)
            ballHistory_.pop_front();
        if ((int)ballHistory_.size() > MAX_HISTORY)
            ballHistory_.pop_front();
    }

    estimateLaunchParams();
    updateShotStateMachine();

    if (calib_.isCalibrated()) {
        // Recompute optimal arc if params changed meaningfully
        if (std::abs(scene_.launchSpeedMs - lastV0_) > 0.2f ||
            std::abs(scene_.distToRimM * 10 - lastOptAngle_ * 10) > 1.f) {
            PhysicsConfig phys;
            float opt = findOptimalAngle(scene_.launchSpeedMs,
                                         scene_.releaseHeightM,
                                         scene_.distToRimM,
                                         COURT_RIM_HEIGHT_M, phys);
            optimalArcPx_ = projectArcToPixels(
                scene_.launchSpeedMs, opt,
                scene_.releaseHeightM, scene_.distToRimM,
                scene_.shootingRight);
            lastOptAngle_ = opt;
            lastV0_       = scene_.launchSpeedMs;
        }
    }

    drawOverlay();
    if (calib_.isCalibrated()) projectAndDrawArc();
    drawTrajectoryComparison();
    drawShotResult();
    drawStats();

    return true;
}

// -----------------------------------------------------------------------
// Ball detection
// -----------------------------------------------------------------------

std::optional<glm::vec2> VisionSystem::detectBall(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, ballLowerHSV, ballUpperHSV, mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5,5});
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return std::nullopt;

    float bestScore = -1.f;
    glm::vec2 bestCenter;

    for (auto& c : contours) {
        float area = (float)cv::contourArea(c);
        if (area < PI * minBallRadius * minBallRadius) continue;
        float perim = (float)cv::arcLength(c, true);
        if (perim < 1.f) continue;
        float circ = 4.f * PI * area / (perim * perim);
        if (circ < 0.55f) continue;
        cv::Point2f ctr; float r;
        cv::minEnclosingCircle(c, ctr, r);
        if (r < minBallRadius || r > maxBallRadius) continue;
        float score = circ * area;
        if (score > bestScore) {
            bestScore = score;
            bestCenter = {ctr.x, ctr.y};
            scene_.ballRadius = r;
        }
    }

    return bestScore >= 0 ? std::make_optional(bestCenter) : std::nullopt;
}

// -----------------------------------------------------------------------
// Pose detection
// -----------------------------------------------------------------------

PoseData VisionSystem::detectPose(const cv::Mat& frame) {
    PoseData pose;
#ifndef NO_MEDIAPIPE
    if (!mp_->running || !mp_->poller) return pose;

    auto imgFrame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat mat = mediapipe::formats::MatView(imgFrame.get());
    cv::cvtColor(frame, mat, cv::COLOR_BGR2RGB);

    size_t tsUs = (size_t)(nowSec() * 1e6);
    mp_->graph.AddPacketToInputStream("input_video",
        mediapipe::Adopt(imgFrame.release()).At(mediapipe::Timestamp(tsUs)));

    mediapipe::Packet pkt;
    if (!mp_->poller->Next(&pkt)) return pose;

    auto& lms = pkt.Get<mediapipe::NormalizedLandmarkList>();
    if (lms.landmark_size() < 29) return pose;

    auto lm = [&](int i) -> glm::vec2 {
        return { lms.landmark(i).x() * frame.cols,
                 lms.landmark(i).y() * frame.rows };
    };

    pose.wristRight    = lm(MP_RIGHT_WRIST);
    pose.wristLeft     = lm(MP_LEFT_WRIST);
    pose.shoulderRight = lm(MP_RIGHT_SHOULDER);
    pose.shoulderLeft  = lm(MP_LEFT_SHOULDER);
    pose.ankleRight    = lm(MP_RIGHT_ANKLE);
    pose.ankleLeft     = lm(MP_LEFT_ANKLE);
    pose.valid         = true;
#endif
    return pose;
}

// -----------------------------------------------------------------------
// World coordinate computation
// -----------------------------------------------------------------------

void VisionSystem::computeWorldCoords() {
    if (!scene_.pose.valid) return;

    float floorY = std::max(scene_.pose.ankleRight.y, scene_.pose.ankleLeft.y);
    calib_.floorYpx = floorY;

    float shoulderY  = (scene_.pose.shoulderRight.y + scene_.pose.shoulderLeft.y) * 0.5f;
    float headY      = shoulderY - 0.15f * std::abs(floorY - shoulderY);
    float heightPx   = floorY - headY;

    if (heightPx > 20.f)
        calib_.metersPerPx = AVG_PERSON_HEIGHT_M / heightPx;

    if (!calib_.isCalibrated()) return;

    float wristY = std::min(scene_.pose.wristRight.y, scene_.pose.wristLeft.y);
    scene_.releaseHeightM = std::clamp(
        (calib_.floorYpx - wristY) * calib_.metersPerPx, 1.0f, 2.8f);

    if (calib_.rimXpx >= 0) {
        float personX     = (scene_.pose.ankleRight.x + scene_.pose.ankleLeft.x) * 0.5f;
        float distPx      = std::abs(calib_.rimXpx - personX);
        scene_.distToRimM = std::clamp(distPx * calib_.metersPerPx, 1.0f, 15.0f);
        scene_.hasValidShot = true;
    }
}

// -----------------------------------------------------------------------
// Detect which direction the person is shooting
// -----------------------------------------------------------------------

void VisionSystem::detectShotDirection() {
    if (!calib_.isCalibrated() || !scene_.pose.valid) return;

    float personX = (scene_.pose.ankleRight.x + scene_.pose.ankleLeft.x) * 0.5f;
    scene_.shootingRight = (personX < calib_.rimXpx);
}

// -----------------------------------------------------------------------
// Estimate launch speed and angle from ball history
// -----------------------------------------------------------------------

void VisionSystem::estimateLaunchParams() {
    if (!calib_.isCalibrated()) return;
    if (ballHistory_.size() < 6) return;

    // Use the most recent segment of ball positions
    int n = std::min((int)ballHistory_.size(), 12);
    auto it = ballHistory_.end() - n;

    std::vector<float> xs, ys, ts;
    for (auto i = it; i != ballHistory_.end(); ++i) {
        auto m = calib_.pixelToMetres(i->posPx.x, i->posPx.y);
        if (!scene_.shootingRight) m.x = -m.x;
        xs.push_back(m.x);
        ys.push_back(m.y);
        ts.push_back(i->timeSec);
    }

    // Normalise time
    float t0 = ts.front();
    for (auto& t : ts) t -= t0;

    // Fit linear vx from x positions
    float sumT = 0, sumX = 0, sumTX = 0, sumT2 = 0;
    for (int i = 0; i < n; ++i) {
        sumT  += ts[i]; sumX  += xs[i];
        sumTX += ts[i] * xs[i]; sumT2 += ts[i] * ts[i];
    }
    float denom = n * sumT2 - sumT * sumT;
    if (std::abs(denom) < 1e-9f) return;
    float vx = (n * sumTX - sumT * sumX) / denom;

    // Fit quadratic for vy: y = vy0*t - 0.5*g*t^2
    // Use first and last points
    float dt  = ts.back() - ts.front();
    float dy  = ys.back() - ys.front();
    if (dt < 0.02f) return;
    float vy = (dy + 0.5f * 9.81f * dt * dt) / dt;

    float v0    = std::sqrt(vx * vx + vy * vy);
    float theta = std::atan2(vy, vx) * 180.f / PI;

    scene_.launchSpeedMs  = std::clamp(v0,    4.0f, 14.0f);
    scene_.launchAngleDeg = std::clamp(theta, 20.0f, 75.0f);
}

// -----------------------------------------------------------------------
// Shot state machine — detects in-flight, made, missed
// -----------------------------------------------------------------------

void VisionSystem::updateShotStateMachine() {
    if (!calib_.isCalibrated() || !scene_.ballCenter) return;

    bool nearRim = ballNearRim();

    if (!shotInFlight_) {
        if (scene_.pose.valid) {
            float wristY = std::min(scene_.pose.wristRight.y,
                                    scene_.pose.wristLeft.y);
            float shoulderY = (scene_.pose.shoulderRight.y +
                               scene_.pose.shoulderLeft.y) * 0.5f;
            if (wristY < shoulderY && scene_.ballCenter->y < shoulderY) {
                shotInFlight_  = true;
                shotStartTime_ = nowSec();
                scene_.shotResult = ShotResult::InFlight;
                scene_.attemptCount++;
            }
        }
    } else {
        float elapsed = nowSec() - shotStartTime_;

        if (elapsed > 4.0f) {
            scene_.shotResult = ShotResult::Missed;
            shotInFlight_     = false;
            resultDisplayTimer_ = nowSec();
        } else if (nearRim && !ballWasNearRim_) {
            if (ballPassedThroughRim()) {
                scene_.shotResult = ShotResult::Made;
                scene_.madeCount++;
            } else {
                scene_.shotResult = ShotResult::Missed;
            }
            shotInFlight_       = false;
            resultDisplayTimer_ = nowSec();
        }
    }

    ballWasNearRim_ = nearRim;

    // Clear result after 2.5 seconds
    if (!shotInFlight_ &&
        scene_.shotResult != ShotResult::None &&
        nowSec() - resultDisplayTimer_ > 2.5f) {
        scene_.shotResult = ShotResult::None;
    }
}

bool VisionSystem::ballNearRim() const {
    if (!scene_.ballCenter || calib_.rimXpx < 0) return false;
    float dx = scene_.ballCenter->x - calib_.rimXpx;
    float dy = scene_.ballCenter->y - calib_.rimYpx;
    float rimRadiusPx = COURT_RIM_RADIUS_M / calib_.metersPerPx;
    return std::sqrt(dx*dx + dy*dy) < rimRadiusPx * 2.5f;
}

bool VisionSystem::ballPassedThroughRim() const {
    if (!scene_.ballCenter || calib_.rimXpx < 0) return false;
    float dx = scene_.ballCenter->x - calib_.rimXpx;
    float dy = scene_.ballCenter->y - calib_.rimYpx;
    float rimRadiusPx = COURT_RIM_RADIUS_M / calib_.metersPerPx;
    return std::sqrt(dx*dx + dy*dy) < rimRadiusPx * 1.2f &&
           scene_.ballCenter->y > calib_.rimYpx;
}

// -----------------------------------------------------------------------
// Calibration
// -----------------------------------------------------------------------

void VisionSystem::setRimPixel(float x, float y) {
    calib_.rimXpx = x;
    calib_.rimYpx = y;
    if (calib_.floorYpx < 0)
        calib_.floorYpx = (float)calib_.frameH - 20.f;
    if (calib_.metersPerPx < 0) {
        float rimHeightPx = calib_.floorYpx - y;
        if (rimHeightPx > 10.f)
            calib_.metersPerPx = COURT_RIM_HEIGHT_M / rimHeightPx;
    }
    scene_.hasValidShot = calib_.isCalibrated();
    std::cout << "[Vision] Rim set (" << x << ", " << y
              << ") scale=" << calib_.metersPerPx << " m/px\n";
}

// -----------------------------------------------------------------------
// Draw helpers
// -----------------------------------------------------------------------

void VisionSystem::drawOverlay() {
    if (scene_.ballCenter) {
        cv::circle(displayFrame_,
            {(int)scene_.ballCenter->x, (int)scene_.ballCenter->y},
            (int)scene_.ballRadius, {0, 140, 255}, 2);
        cv::circle(displayFrame_,
            {(int)scene_.ballCenter->x, (int)scene_.ballCenter->y},
            3, {0, 255, 255}, -1);
    }

    if (scene_.pose.valid) {
        auto joint = [&](glm::vec2 p, cv::Scalar c) {
            cv::circle(displayFrame_, {(int)p.x, (int)p.y}, 5, c, -1);
        };
        auto bone = [&](glm::vec2 a, glm::vec2 b) {
            cv::line(displayFrame_, {(int)a.x,(int)a.y},
                     {(int)b.x,(int)b.y}, {0,200,0}, 2);
        };
        joint(scene_.pose.wristRight,    {0,120,255});
        joint(scene_.pose.wristLeft,     {0,120,255});
        joint(scene_.pose.shoulderRight, {0,255,0});
        joint(scene_.pose.shoulderLeft,  {0,255,0});
        joint(scene_.pose.ankleRight,    {255,200,0});
        joint(scene_.pose.ankleLeft,     {255,200,0});
        bone(scene_.pose.shoulderRight, scene_.pose.wristRight);
        bone(scene_.pose.shoulderLeft,  scene_.pose.wristLeft);
        bone(scene_.pose.shoulderRight, scene_.pose.shoulderLeft);
        bone(scene_.pose.ankleRight,    scene_.pose.shoulderRight);
        bone(scene_.pose.ankleLeft,     scene_.pose.shoulderLeft);
    }

    if (calib_.rimXpx >= 0) {
        cv::circle(displayFrame_,
            {(int)calib_.rimXpx, (int)calib_.rimYpx}, 10, {255,80,0}, 2);
        cv::line(displayFrame_,
            {(int)calib_.rimXpx-14, (int)calib_.rimYpx},
            {(int)calib_.rimXpx+14, (int)calib_.rimYpx}, {255,80,0}, 2);
    }

    if (!calib_.isCalibrated()) {
        cv::putText(displayFrame_, "LEFT CLICK the RIM to calibrate",
            {20, calib_.frameH - 30},
            cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,100,255}, 2, cv::LINE_AA);
    }
}

void VisionSystem::projectAndDrawArc() {
    if (optimalArcPx_.size() < 2) return;

    for (size_t i = 1; i < optimalArcPx_.size(); ++i) {
        float t = (float)i / optimalArcPx_.size();
        cv::Scalar col(
            (uchar)(255*(1.f-t)),
            (uchar)(200+55*t),
            (uchar)(80*t)
        );
        cv::line(displayFrame_,
            {(int)optimalArcPx_[i-1].x, (int)optimalArcPx_[i-1].y},
            {(int)optimalArcPx_[i].x,   (int)optimalArcPx_[i].y},
            col, 3, cv::LINE_AA);
    }

    if (!optimalArcPx_.empty()) {
        std::string lbl = "Optimal: " + std::to_string((int)lastOptAngle_) + " deg  "
                        + std::to_string((int)(scene_.launchSpeedMs*10)/10.f) + " m/s";
        cv::putText(displayFrame_, lbl,
            {(int)optimalArcPx_[0].x + 10, (int)optimalArcPx_[0].y - 10},
            cv::FONT_HERSHEY_SIMPLEX, 0.65, {100,255,100}, 2, cv::LINE_AA);
    }
}

// -----------------------------------------------------------------------
// Draw actual ball trajectory vs optimal arc
// -----------------------------------------------------------------------

void VisionSystem::drawTrajectoryComparison() {
    if (ballHistory_.size() < 3 || !calib_.isCalibrated()) return;

    // Draw actual trail
    for (size_t i = 1; i < ballHistory_.size(); ++i) {
        float alpha = (float)i / ballHistory_.size();
        cv::Scalar col(
            (uchar)(255 * (1.f - alpha)),
            (uchar)(165 * alpha),
            (uchar)(255 * alpha)
        );
        cv::line(displayFrame_,
            {(int)ballHistory_[i-1].posPx.x, (int)ballHistory_[i-1].posPx.y},
            {(int)ballHistory_[i].posPx.x,   (int)ballHistory_[i].posPx.y},
            col, 2, cv::LINE_AA);
    }

    // Deviation from optimal arc at matching x positions
    if (optimalArcPx_.empty() || !scene_.ballCenter) return;

    float ballXpx = scene_.ballCenter->x;
    float minDist = 1e9f;
    glm::vec2 closestArcPt;

    for (auto& p : optimalArcPx_) {
        float d = std::abs(p.x - ballXpx);
        if (d < minDist) { minDist = d; closestArcPt = p; }
    }

    if (minDist < 40.f) {
        cv::line(displayFrame_,
            {(int)scene_.ballCenter->x, (int)scene_.ballCenter->y},
            {(int)closestArcPt.x,       (int)closestArcPt.y},
            {0, 80, 255}, 1, cv::LINE_AA);

        float deviationM = minDist * calib_.metersPerPx;
        std::string devStr = "Off by " +
            std::to_string((int)(deviationM * 100)) + " cm";
        cv::putText(displayFrame_, devStr,
            {(int)scene_.ballCenter->x + 12, (int)scene_.ballCenter->y - 12},
            cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,180,255}, 2, cv::LINE_AA);
    }
}

// -----------------------------------------------------------------------
// Draw MADE / MISSED banner
// -----------------------------------------------------------------------

void VisionSystem::drawShotResult() {
    if (scene_.shotResult == ShotResult::None ||
        scene_.shotResult == ShotResult::InFlight) return;

    bool made = (scene_.shotResult == ShotResult::Made);

    cv::Scalar bgCol  = made ? cv::Scalar(0,180,0)   : cv::Scalar(0,0,200);
    cv::Scalar txtCol = made ? cv::Scalar(0,255,100)  : cv::Scalar(80,80,255);
    std::string label = made ? "MADE!" : "MISSED";

    int cx = calib_.frameW / 2;
    int cy = calib_.frameH / 2 - 40;

    // Semi-transparent background
    cv::Mat overlay = displayFrame_.clone();
    cv::rectangle(overlay, {cx-140, cy-50}, {cx+140, cy+20}, bgCol, -1);
    cv::addWeighted(overlay, 0.55, displayFrame_, 0.45, 0, displayFrame_);

    cv::putText(displayFrame_, label, {cx-110, cy+10},
        cv::FONT_HERSHEY_DUPLEX, 1.8, txtCol, 3, cv::LINE_AA);
}

// -----------------------------------------------------------------------
// Stats panel (FPS, make%, launch speed, angle, direction)
// -----------------------------------------------------------------------

void VisionSystem::drawStats() {
    auto put = [&](const std::string& s, int row, cv::Scalar col = {220,220,220}) {
        cv::putText(displayFrame_, s, {20, row},
                    cv::FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv::LINE_AA);
    };

    int y = 30;
    put("FPS: " + std::to_string((int)fps_), y); y += 28;

    if (scene_.hasValidShot) {
        put("Release ht : " + std::to_string((int)(scene_.releaseHeightM*100)) + " cm",
            y, {180,255,120}); y += 28;
        put("Dist to rim: " + std::to_string((int)(scene_.distToRimM*10)/10.f) + " m",
            y, {180,255,120}); y += 28;
        put("Est. speed : " + std::to_string((int)(scene_.launchSpeedMs*10)/10.f) + " m/s",
            y, {255,220,100}); y += 28;
        put("Est. angle : " + std::to_string((int)scene_.launchAngleDeg) + " deg",
            y, {255,220,100}); y += 28;
        put("Direction  : " + std::string(scene_.shootingRight ? "Right ->" : "<- Left"),
            y, {200,200,255}); y += 28;
    }

    if (scene_.attemptCount > 0) {
        float pct = 100.f * scene_.madeCount / scene_.attemptCount;
        put("Makes: " + std::to_string(scene_.madeCount) +
            "/" + std::to_string(scene_.attemptCount) +
            "  (" + std::to_string((int)pct) + "%)",
            y, {100,255,255});
    }

    if (scene_.shotResult == ShotResult::InFlight) {
        put("IN FLIGHT...", calib_.frameH - 30, {0,255,200});
    }
}

// -----------------------------------------------------------------------
// Arc projection (direction-aware)
// -----------------------------------------------------------------------

std::vector<glm::vec2> VisionSystem::projectArcToPixels(
    float v0, float thetaDeg, float h0M, float distM, bool shootRight)
{
    PhysicsConfig phys;
    auto traj = simulateDrag(v0, thetaDeg, 0.f, h0M, phys, 0.01f, 3.0f);

    float shooterXpx = calib_.rimXpx;
    if (scene_.pose.valid) {
        shooterXpx = (scene_.pose.ankleRight.x + scene_.pose.ankleLeft.x) * 0.5f;
    } else {
        shooterXpx = shootRight
            ? calib_.rimXpx - distM / calib_.metersPerPx
            : calib_.rimXpx + distM / calib_.metersPerPx;
    }

    float dirSign = shootRight ? 1.f : -1.f;

    std::vector<glm::vec2> pts;
    for (auto& tp : traj) {
        if (tp.pos.x > distM + 0.1f) break;
        float xPx = shooterXpx + dirSign * tp.pos.x / calib_.metersPerPx;
        float yPx = calib_.floorYpx - tp.pos.y / calib_.metersPerPx;
        if (xPx < 0 || xPx > (float)calib_.frameW) break;
        if (yPx < 0 || yPx > (float)calib_.frameH) continue;
        pts.push_back({xPx, yPx});
    }
    return pts;
}

float VisionSystem::nowSec() const {
    return (float)(cv::getTickCount() / cv::getTickFrequency());
}