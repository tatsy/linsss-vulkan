#version 450

#include "utils.glsl"

layout (location = 0) in vec3 inRayDir;
layout (location = 0) out vec4 outFragColor;

layout (binding = 1) uniform UBO {
    vec4 sphereHarmCoefs[9];
    vec4 lightPower;
    int lightType;
} ubo;

layout (binding = 2) uniform sampler2D texEnvmap;

void main(void) {
    if (ubo.lightType == LIGHT_TYPE_POINT) discard;

    vec3 dir = normalize(inRayDir);
    float dx = clamp(dir.x, -1.0 + M_EPS, 1.0 - M_EPS);
    float dy = clamp(dir.y, -1.0 + M_EPS, 1.0 - M_EPS);
    float dz = clamp(dir.z, -1.0 + M_EPS, 1.0 - M_EPS);

    float phi = atan(dz, dx);
    float theta = acos(clamp(dy, -1.0, 1.0));
    float u = (phi + M_PI) / M_TWO_PI;
    float v = theta / M_PI;

    vec3 rgb = texture(texEnvmap, vec2(u, v)).xyz;
    outFragColor = sRGB(vec4(rgb, 1.0));
}
