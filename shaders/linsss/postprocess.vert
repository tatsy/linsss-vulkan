#version 450

layout (location = 0) in vec3 inPos;

layout (location = 0) out vec4 outPos;
layout (location = 1) out vec2 outRcpFrame;

layout (binding = 0) uniform UBO
{
    int winWidth;
    int winHeight;
} ubo;

out gl_PerVertex
{
    vec4 gl_Position;
};

const float FXAA_SUBPIX_SHIFT = 1.0 / 4.0;

void main(void) {
    gl_Position = vec4(inPos, 1.0);

    vec2 inUV = inPos.xy * 0.5 + 0.5;
    vec2 rcpFrame = vec2(1.0 / float(ubo.winWidth), 1.0 / float(ubo.winHeight));
    outPos.xy = inUV;
    outPos.zw = inUV - (rcpFrame * (0.5 + FXAA_SUBPIX_SHIFT));
    outRcpFrame = rcpFrame;
}
