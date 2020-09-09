#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec3 outLightVec;
layout (location = 3) out vec3 outViewVec;

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


void main() {
    gl_Position = ubo.projection * ubo.model * vec4(inPos, 1.0);
    outUV = (gl_Position.xy / gl_Position.w) * 0.5 + 0.5;

    outNormal = mat3(inverse(transpose(ubo.model))) * inNormal;

    vec4 pos = ubo.model * vec4(inPos, 1.0);
    vec3 lPos = mat3(ubo.model) * ubo.lightPos.xyz;
    outLightVec = lPos - pos.xyz;
    outViewVec = ubo.viewPos.xyz - pos.xyz;
}
