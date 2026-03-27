#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aAlpha;
layout(location = 2) in float aQuality;

out float vAlpha;
out float vQuality;

uniform mat4 view;
uniform mat4 projection;

void main() {
    vAlpha   = aAlpha;
    vQuality = aQuality;
    gl_Position  = projection * view * vec4(aPos, 1.0);
    gl_PointSize = 6.0 * aAlpha;
}