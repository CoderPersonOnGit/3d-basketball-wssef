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
              << "  S            - save current frame\n\n"
              << "Setup:\n"
              << "  1. Install DroidCam on phone + PC\n"
              << "  2. Connect phone and PC to same WiFi\n"
              << "  3. Run with the IP shown in DroidCam app\n\n";
}

int main(int argc, char** argv) {
    printUsage();

    VisionSystem vision;
    g_vision = &vision;

    bool opened = false;

    if (argc >= 2) {
        std::string arg = argv[1];
        // Check if it's a URL or a number
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
        // Try default camera
        opened = vision.openCamera(0);
        if (!opened) {
            // Try DroidCam default port
            std::cout << "[App] Default camera failed. Trying DroidCam on localhost...\n";
            opened = vision.openCamera("http://localhost:4747/video");
        }
    }

    if (!opened) {
        std::cerr << "[App] Could not open any camera. Exiting.\n";
        return -1;
    }

    const std::string windowName = "Basketball Shot Analyzer";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1280, 720);
    cv::setMouseCallback(windowName, onMouse);

    std::cout << "[App] Running. Left-click the RIM to enable arc overlay.\n";

    int frameCount = 0;

    while (true) {
        if (!vision.processFrame()) {
            std::cerr << "[App] Lost camera feed.\n";
            break;
        }

        cv::imshow(windowName, vision.getFrame());

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;

        if (key == 's') {
            std::string fname = "shot_" + std::to_string(frameCount++) + ".png";
            cv::imwrite(fname, vision.getFrame());
            std::cout << "[App] Saved: " << fname << "\n";
        }
    }

    vision.close();
    cv::destroyAllWindows();
    return 0;
}