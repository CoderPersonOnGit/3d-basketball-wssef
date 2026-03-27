cmake_minimum_required(VERSION 3.20)
project(Basketball3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenGL ---
find_package(OpenGL REQUIRED)

# --- OpenCV Setup ---
set(OpenCV_DIR "C:/libs/opencv/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# --- Global Include Directories ---
include_directories(
    ${CMAKE_SOURCE_DIR}
    C:/libs/glew/glew-2.3.1/include
    C:/vcpkg/installed/x64-windows/include
    C:/libs/glm
    ${OpenCV_INCLUDE_DIRS}
)

# --- Library Paths ---
set(GLEW_LIBRARIES "C:/libs/glew/glew-2.3.1/lib/Release/x64/glew32.lib")
set(GLFW_LIBRARIES "C:/vcpkg/installed/x64-windows/lib/glfw3dll.lib")

# ------------------------------
# Executable: basketball3d
# ------------------------------
add_executable(basketball3d
    main.cpp
    physics.cpp
    renderer.cpp
    court.cpp
    shader.cpp
)

target_link_libraries(basketball3d PRIVATE
    OpenGL::GL
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OpenCV_LIBS}
)

# ------------------------------
# Executable: camera_app
# ------------------------------
add_executable(camera_app
    cameramain.cpp
    vision.cpp
    physics.cpp
    shader.cpp
)

target_link_libraries(camera_app PRIVATE
    OpenGL::GL
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OpenCV_LIBS}
)

# ------------------------------
# Windows-specific definitions
# ------------------------------
if(WIN32)
    target_compile_definitions(basketball3d PRIVATE NOMINMAX NO_MEDIAPIPE)
    target_compile_definitions(camera_app   PRIVATE NOMINMAX NO_MEDIAPIPE)
endif()

