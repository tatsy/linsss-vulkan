#version 450

layout (location = 0) in vec3 inPos;

layout (location = 0) out vec3 outRayDir;

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 model;
    vec4 viewPos;
    vec4 lightPos;
} ubo;

out gl_PerVertex
{
    vec4 gl_Position;
};

float cubeSize = 100.0;

void main() {
    vec3 scaledPos = cubeSize * inPos;
    gl_Position = ubo.projection * ubo.model * vec4(scaledPos.xyz, 1.0);
    vec4 worldPos = vec4(scaledPos, 1.0);
    outRayDir = worldPos.xyz - ubo.viewPos.xyz;
}
