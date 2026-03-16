#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "vision.h"

static VisionSystem* g_vision = nullptr;

static void onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN && g_vision) {
        g_vision->setRimPixel((float)x, (float)y);
    }
}

void printUsage() {
    std::cout << "\n=== Basketball Shot Analyzer ===\n"
              << "Usage:\n"
              << "  camera_app             - use default webcam (index 0)\n"
              << "  camera_app <index>     - use camera index\n"
              << "  camera_app <url>       - use DroidCam stream\n"
              << "    e.g. camera_app http://192.168.1.100:4747/video\n\n"
              << "Controls:\n"
              << "  Left click   - set rim position (required for arc)\n"
              << "  Q / ESC      - quit\n"
              << "  S            - save current frame\n"
              << "  T            - toggle HSV tuning window\n"
              << "  M            - toggle HSV mask preview\n"
              << "  1            - set distance: Layup (1.5m)\n"
              << "  2            - set distance: Free Throw (4.6m)\n"
              << "  3            - set distance: Three-Point (6.75m)\n"
              << "  0            - set distance: AUTO\n\n";
}

int main(int argc, char** argv) {
    printUsage();

    VisionSystem vision;
    g_vision = &vision;

    bool opened = false;

    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg.find("http") != std::string::npos ||
            arg.find("rtsp") != std::string::npos) {
            opened = vision.openCamera(arg);
        } else {
            try {
                int idx = std::stoi(arg);
                opened = vision.openCamera(idx);
            } catch (...) {
                opened = vision.openCamera(arg);
            }
        }
    } else {
        opened = vision.openCamera(0);
        if (!opened) {
            std::cout << "[App] Default camera failed. Trying DroidCam on localhost...\n";
            opened = vision.openCamera("http://localhost:4747/video");
        }
    }

    if (!opened) {
        std::cerr << "[App] Could not open any camera. Exiting.\n";
        return -1;
    }

    const std::string mainWin = "Basketball Shot Analyzer";
    const std::string tuneWin = "HSV Tuner  |  T to toggle";
    const std::string maskWin = "Ball Mask  |  M to toggle";

    cv::namedWindow(mainWin, cv::WINDOW_NORMAL);
    cv::resizeWindow(mainWin, 1280, 720);
    cv::setMouseCallback(mainWin, onMouse);

    int hLow  = (int)vision.ballLowerHSV[0];
    int sLow  = (int)vision.ballLowerHSV[1];
    int vLow  = (int)vision.ballLowerHSV[2];
    int hHigh = (int)vision.ballUpperHSV[0];
    int sHigh = (int)vision.ballUpperHSV[1];
    int vHigh = (int)vision.ballUpperHSV[2];
    int rMin  = (int)vision.minBallRadius;
    int rMax  = (int)vision.maxBallRadius;

    bool showTuner = false;
    bool showMask  = false;

    std::cout << "[App] Running. Left-click the RIM to enable arc overlay.\n";
    std::cout << "[App] Press T to open HSV tuner, M to see ball mask.\n";
    std::cout << "[App] Press 1/2/3 to set shot distance, 0 for auto.\n";

    int frameCount = 0;

    while (true) {
        if (!vision.processFrame()) {
            std::cerr << "[App] Lost camera feed.\n";
            break;
        }

        // Skip frames for faster playback (change 2 to 3 for 3x speed)
        static int frameSkip = 0;
        frameSkip++;
        if (frameSkip % 5 != 0) {
            cv::waitKey(1);
            continue;
        }

        if (showTuner) {
            vision.ballLowerHSV  = cv::Scalar(hLow,  sLow,  vLow);
            vision.ballUpperHSV  = cv::Scalar(hHigh, sHigh, vHigh);
            vision.minBallRadius = (float)rMin;
            vision.maxBallRadius = (float)rMax;
        }

        cv::imshow(mainWin, vision.getFrame());

        if (showMask) {
            cv::Mat hsv, mask;
            cv::cvtColor(vision.getFrame(), hsv, cv::COLOR_BGR2HSV);
            cv::inRange(hsv,
                cv::Scalar(hLow, sLow, vLow),
                cv::Scalar(hHigh, sHigh, vHigh),
                mask);
            cv::Mat green(mask.size(), CV_8UC3, cv::Scalar(0, 255, 80));
            cv::Mat result;
            cv::addWeighted(vision.getFrame(), 0.4, green, 0.0, 0, result);
            green.copyTo(result, mask);
            cv::addWeighted(vision.getFrame(), 0.4, result, 0.6, 0, result);
            cv::putText(result, "GREEN = detected as ball candidate",
                {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,255,80}, 2);
            cv::namedWindow(maskWin, cv::WINDOW_NORMAL);
            cv::resizeWindow(maskWin, 1280, 720);
            cv::imshow(maskWin, result);
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;

        if (key == 's') {
            std::string fname = "shot_" + std::to_string(frameCount++) + ".png";
            cv::imwrite(fname, vision.getFrame());
            std::cout << "[App] Saved: " << fname << "\n";
        }

        if (key == '1') { vision.setManualDistance(DIST_LAYUP_M);      std::cout << "[App] Distance: Layup (1.5m)\n"; }
        if (key == '2') { vision.setManualDistance(DIST_FREE_THROW_M); std::cout << "[App] Distance: Free Throw (4.6m)\n"; }
        if (key == '3') { vision.setManualDistance(DIST_THREE_PT_M);   std::cout << "[App] Distance: Three-Point (6.75m)\n"; }
        if (key == '0') { vision.setManualDistance(0.f);               std::cout << "[App] Distance: AUTO\n"; }

        if (key == 't' || key == 'T') {
            showTuner = !showTuner;
            if (showTuner) {
                cv::namedWindow(tuneWin, cv::WINDOW_NORMAL);
                cv::resizeWindow(tuneWin, 500, 320);
                cv::createTrackbar("H Low",  tuneWin, &hLow,  179);
                cv::createTrackbar("H High", tuneWin, &hHigh, 179);
                cv::createTrackbar("S Low",  tuneWin, &sLow,  255);
                cv::createTrackbar("S High", tuneWin, &sHigh, 255);
                cv::createTrackbar("V Low",  tuneWin, &vLow,  255);
                cv::createTrackbar("V High", tuneWin, &vHigh, 255);
                cv::createTrackbar("R Min",  tuneWin, &rMin,  80);
                cv::createTrackbar("R Max",  tuneWin, &rMax,  150);
                cv::Mat swatch(60, 500, CV_8UC3, cv::Scalar(30, 30, 30));
                cv::imshow(tuneWin, swatch);
                std::cout << "[App] HSV Tuner opened.\n";
            } else {
                cv::destroyWindow(tuneWin);
                std::cout << "[App] HSV locked: H=" << hLow << "-" << hHigh
                          << " S=" << sLow << "-" << sHigh
                          << " V=" << vLow << "-" << vHigh
                          << " R=" << rMin << "-" << rMax << "\n";
            }
        }

        if (key == 'm' || key == 'M') {
            showMask = !showMask;
            if (!showMask) cv::destroyWindow(maskWin);
        }
    }

    vision.close();
    cv::destroyAllWindows();
    return 0;
}