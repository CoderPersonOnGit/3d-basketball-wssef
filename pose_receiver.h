#pragma once
#include <glm/glm.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif

struct PoseData {
    glm::vec2 wristRight    = {0, 0};
    glm::vec2 wristLeft     = {0, 0};
    glm::vec2 ankleRight    = {0, 0};
    glm::vec2 ankleLeft     = {0, 0};
    glm::vec2 shoulderRight = {0, 0};
    glm::vec2 shoulderLeft  = {0, 0};
    bool      valid         = false;

    // Ball detection from MediaPipe object detector
    glm::vec2 ballCenter    = {0, 0};
    float     ballRadius    = 0.f;
    bool      ballValid     = false;
};

class PoseReceiver {
public:
    PoseReceiver(int port = 5005) : port_(port), sock_(INVALID_SOCKET) {}

    void start() {
#ifdef _WIN32
        WSADATA w;
        WSAStartup(MAKEWORD(2,2), &w);
#endif
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(sock_, (sockaddr*)&addr, sizeof(addr));

#ifdef _WIN32
        u_long mode = 1;
        ioctlsocket(sock_, FIONBIO, &mode);
#endif
        running_ = true;
        thread_  = std::thread(&PoseReceiver::loop, this);
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
#ifdef _WIN32
        closesocket(sock_);
        WSACleanup();
#endif
    }

    PoseData get() {
        std::lock_guard<std::mutex> lk(mtx_);
        return latest_;
    }

private:
    int                port_;
    SOCKET             sock_;
    std::thread        thread_;
    std::mutex         mtx_;
    std::atomic<bool>  running_{false};
    PoseData           latest_;

    void loop() {
        char buf[4096];
        while (running_) {
            int n = recv(sock_, buf, sizeof(buf)-1, 0);
            if (n <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            buf[n] = '\0';
            PoseData pd = parse(buf);
            std::lock_guard<std::mutex> lk(mtx_);
            latest_ = pd;
        }
    }

    static float extractFloat(const std::string& s, size_t& pos) {
        while (pos < s.size() &&
               (s[pos]==' '||s[pos]=='['||s[pos]==',')) pos++;
        size_t end = pos;
        while (end < s.size() && s[end]!=']' && s[end]!=',') end++;
        float v = std::stof(s.substr(pos, end - pos));
        pos = end;
        return v;
    }

    static glm::vec2 extractVec2(const std::string& s, const std::string& key) {
        size_t p = s.find("\"" + key + "\"");
        if (p == std::string::npos) return {0, 0};
        p = s.find('[', p);
        if (p == std::string::npos) return {0, 0};
        float x = extractFloat(s, p);
        float y = extractFloat(s, p);
        return {x, y};
    }

    // Extracts [x, y, r] array for ball
    static bool extractVec3(const std::string& s, const std::string& key,
                             float& x, float& y, float& r) {
        size_t p = s.find("\"" + key + "\"");
        if (p == std::string::npos) return false;
        // check for null
        size_t colon = s.find(':', p);
        if (colon == std::string::npos) return false;
        size_t val = colon + 1;
        while (val < s.size() && s[val] == ' ') val++;
        if (s[val] == 'n') return false;  // null
        p = s.find('[', colon);
        if (p == std::string::npos) return false;
        x = extractFloat(s, p);
        y = extractFloat(s, p);
        r = extractFloat(s, p);
        return true;
    }

    static PoseData parse(const std::string& s) {
        PoseData pd;
        pd.valid = s.find("\"valid\": true")  != std::string::npos ||
                   s.find("\"valid\":true")   != std::string::npos;
        if (pd.valid) {
            pd.wristRight    = extractVec2(s, "wristRight");
            pd.wristLeft     = extractVec2(s, "wristLeft");
            pd.shoulderRight = extractVec2(s, "shoulderRight");
            pd.shoulderLeft  = extractVec2(s, "shoulderLeft");
            pd.ankleRight    = extractVec2(s, "ankleRight");
            pd.ankleLeft     = extractVec2(s, "ankleLeft");
        }
        float bx, by, br;
        pd.ballValid = extractVec3(s, "ball", bx, by, br);
        if (pd.ballValid) {
            pd.ballCenter = {bx, by};
            pd.ballRadius = br;
        }
        return pd;
    }
};