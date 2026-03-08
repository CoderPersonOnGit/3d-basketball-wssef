cmake_minimum_required(VERSION 3.20)
project(Basketball3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(basketball3d
    src/main.cpp
    src/physics.cpp
    src/renderer.cpp
    src/court.cpp
    src/shader.cpp
)

target_include_directories(basketball3d PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(basketball3d PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    glm::glm
    ${OpenCV_LIBS}
)

add_executable(camera_app
    src/cameramain.cpp
    src/vision.cpp
    src/physics.cpp
    src/shader.cpp
)

target_include_directories(camera_app PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_app PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    glm::glm
    ${OpenCV_LIBS}
)

if(WIN32)
    target_compile_definitions(basketball3d PRIVATE NOMINMAX)
    target_compile_definitions(camera_app   PRIVATE NOMINMAX)
endif()