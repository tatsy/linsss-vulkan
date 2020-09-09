#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 model;
    vec4 viewPos;
    vec4 lightPos;
    mat4 smModelViewProj;
} ubo;

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outPosCamSpace;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outNormal;
layout (location = 4) out vec3 outOrigNormal;
layout (location = 5) out vec3 outViewVec;
layout (location = 6) out vec3 outLightVec;
layout (location = 7) out vec4 outPosScreenSM;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    outPos = inPos;
    outUV = inUV;

    gl_Position = ubo.projection * ubo.model * vec4(inPos.xyz, 1.0);

    outNormal = mat3(inverse(transpose(ubo.model))) * inNormal;
    outOrigNormal = inNormal;

    vec4 pos = ubo.model * vec4(inPos, 1.0);
    outPosCamSpace = pos.xyz;
    vec3 lPos = mat3(ubo.model) * ubo.lightPos.xyz;
    outLightVec = lPos - pos.xyz;
    outViewVec = ubo.viewPos.xyz - pos.xyz;

    outPosScreenSM = ubo.smModelViewProj * vec4(inPos.xyz, 1.0);
}
