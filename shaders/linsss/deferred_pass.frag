#version 450

#include "utils.glsl"

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inLightVec;
layout (location = 3) in vec3 inViewVec;

layout (location = 0) out vec4 outFragColor;

layout (binding = 3) uniform sampler2D sssTex;
layout (binding = 4) uniform sampler2D tsmTex;
layout (binding = 5) uniform sampler2D specTex;
layout (binding = 6) uniform sampler2D depthTex;

// eta = 1.5
// Fdr = -1.4399 / (eta * eta) + 0.7099 / eta + 0.6681 + 0.0636 * eta;
const float FRESNEL_DIFFUSE = 0.59681111;

void main() {

    vec3 n = normalize(inNormal);
    vec3 wi = normalize(inLightVec);
    vec3 wo = normalize(inViewVec);

    float cosThetaI = max(0.0, dot(n, wi));
    float cosThetaO = max(0.0, dot(n, wo));

    float Ft = 1.0 - fresnelDielectric(cosThetaO, 1.0, 1.5);
    float Fdr = 1.0 - FRESNEL_DIFFUSE;
    vec3 sssNear = texture(sssTex, inUV).rgb;
    vec3 sssFar = texture(tsmTex, inUV).rgb / texture(tsmTex, inUV).a;
    vec3 sss = (sssNear + sssFar) * Ft * Fdr * M_INV_PI;

    vec3 specular = texture(specTex, inUV).rgb;

    vec4 rgb = vec4(specular + sss, 1.0);
    outFragColor = sRGB(rgb);
}
