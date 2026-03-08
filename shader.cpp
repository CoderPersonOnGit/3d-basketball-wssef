#include "shader.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::string Shader::loadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[Shader] Cannot open: " << path << "\n";
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

GLuint Shader::compileShader(const std::string& src, GLenum type) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "[Shader] Compile error:\n" << log << "\n";
    }
    return s;
}

Shader::Shader(const std::string& vertPath, const std::string& fragPath) {
    std::string vs = loadFile(vertPath);
    std::string fs = loadFile(fragPath);
    GLuint v = compileShader(vs, GL_VERTEX_SHADER);
    GLuint f = compileShader(fs, GL_FRAGMENT_SHADER);
    ID = glCreateProgram();
    glAttachShader(ID, v);
    glAttachShader(ID, f);
    glLinkProgram(ID);
    GLint ok;
    glGetProgramiv(ID, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(ID, 512, nullptr, log);
        std::cerr << "[Shader] Link error:\n" << log << "\n";
    }
    glDeleteShader(v);
    glDeleteShader(f);
}