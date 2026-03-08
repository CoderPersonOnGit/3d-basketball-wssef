#pragma once
#include <GL/glew.h>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

class Shader {
public:
    GLuint ID;

    Shader() : ID(0) {}
    Shader(const std::string& vertPath, const std::string& fragPath);

    void use() const { glUseProgram(ID); }

    void setMat4(const std::string& name, const glm::mat4& m) const {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(m));
    }
    void setVec3(const std::string& name, const glm::vec3& v) const {
        glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(v));
    }
    void setVec4(const std::string& name, const glm::vec4& v) const {
        glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(v));
    }
    void setFloat(const std::string& name, float v) const {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), v);
    }
    void setInt(const std::string& name, int v) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), v);
    }

private:
    GLuint compileShader(const std::string& src, GLenum type);
    std::string loadFile(const std::string& path);
};