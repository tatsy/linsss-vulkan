#version 450
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec4 outPosScreen;

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 model;
    vec4 viewPos;
    vec4 lightPos;
} ubo;

void main() {
    gl_Position = ubo.projection * ubo.model * vec4(inPos, 1.0);
    outPos = inPos;
    outNormal = inNormal;
    outPosScreen = gl_Position;
}
