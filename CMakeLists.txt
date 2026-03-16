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
    main.cpp
    physics.cpp
    renderer.cpp
    court.cpp
    shader.cpp
)

target_include_directories(basketball3d PRIVATE
    ${CMAKE_SOURCE_DIR}
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
    cameramain.cpp
    vision.cpp
    physics.cpp
    shader.cpp
)

target_include_directories(camera_app PRIVATE
    ${CMAKE_SOURCE_DIR}
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
    target_compile_definitions(basketball3d PRIVATE NOMINMAX NO_MEDIAPIPE)
    target_compile_definitions(camera_app   PRIVATE NOMINMAX NO_MEDIAPIPE)
endif()