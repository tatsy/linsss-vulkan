#version 450

#include "utils.glsl"

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inPosCamSpace;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inNormal;
layout (location = 4) in vec3 inOrigNormal;
layout (location = 5) in vec3 inViewVec;
layout (location = 6) in vec3 inLightVec;
layout (location = 7) in vec4 inPosScreenSM;

layout (location = 0) out vec4 outFragColor;
layout (location = 1) out vec4 outSpecColor;
layout (location = 2) out vec4 outPos;
layout (location = 3) out vec4 outNormal;
layout (location = 4) out vec4 outDepth;

layout (binding = 1) uniform UBO {
    vec4 sphereHarmCoefs[9];
    vec4 lightPower;
    int lightType;
} ubo;

layout (binding = 2) uniform sampler2D texKs;
layout (binding = 3) uniform sampler2D texEnvmap;
layout (binding = 4) uniform sampler2D depthBuffer;

const int nPCFSamples = 16;
vec3 samples[] = vec3[32](
    vec3(-0.015809, -0.008987, 0.175437),
    vec3(0.664079, -0.286524, 0.151204),
    vec3(-0.221546, 0.289903, 0.863210),
    vec3(-0.018174, -0.024018, 0.668253),
    vec3(0.331269, 0.132583, 0.277102),
    vec3(0.663623, 0.075051, 0.106002),
    vec3(0.084538, 0.007773, 0.954540),
    vec3(-0.685963, -0.571579, 0.221340),
    vec3(-0.093575, -0.418713, 0.086912),
    vec3(0.373525, -0.070778, 0.411393),
    vec3(-0.146223, -0.194811, 0.128132),
    vec3(-0.282008, -0.481534, 0.598684),
    vec3(0.137529, -0.021423, 0.697192),
    vec3(-0.184849, -0.230509, 0.207357),
    vec3(0.057931, -0.261809, 0.514569),
    vec3(-0.626876, -0.142766, 0.015477),
    vec3(0.035749, -0.134544, 0.636266),
    vec3(0.140209, 0.705407, 0.542895),
    vec3(0.451050, 0.065316, 0.397382),
    vec3(-0.075430, -0.589476, 0.501260),
    vec3(-0.014807, -0.791854, 0.124690),
    vec3(0.066264, -0.024522, 0.987511),
    vec3(-0.553396, -0.513869, 0.344234),
    vec3(-0.031567, -0.014055, 0.061356),
    vec3(0.002009, 0.598487, 0.729929),
    vec3(-0.093261, -0.053454, 0.559946),
    vec3(0.000588, 0.011218, 0.983976),
    vec3(0.627568, 0.397864, 0.154045),
    vec3(0.009502, -0.000035, 0.066689),
    vec3(-0.273961, -0.341573, 0.043889),
    vec3(-0.090554, -0.178244, 0.063929),
    vec3(-0.017620, -0.016803, 0.095556)
);

float visibility(in vec3 n, in vec3 l, vec3 jitter) {
    float ndotl = max(0.0, dot(n, l));
    float bias = 0.005;

    float zValue = inPosScreenSM.z / inPosScreenSM.w;
    vec2 uv = (inPosScreenSM.xy / inPosScreenSM.w) * 0.5 + 0.5;
    float dValue = texture(depthBuffer, uv + jitter.xy).x;
    float vis = 1.0;
    if (dValue + bias < zValue) {
        vis = 0.0;
    }

    return vis;
}

float visibilityPCF(in vec3 n, in vec3 l) {
    float vis = 0.0;
    for (int i = 0; i < nPCFSamples; i++) {
        vec3 jitter = samples[i] * 0.005;
        vis += visibility(n, l, jitter);
    }
    return vis / float(nPCFSamples);
}

void main()
{
    vec3 n = normalize(inNormal);
    vec3 wi = normalize(inLightVec);
    vec3 wo = normalize(inViewVec);

    vec3 u = cross(abs(n.x) > 0.1 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0), n);
    vec3 v = cross(n, u);
    vec3 wiLocal = vec3(dot(u, wi), dot(v, wi), dot(n, wi));
    vec3 woLocal = vec3(dot(u, wo), dot(v, wo), dot(n, wo));

    vec2 screenUV = inPos.xy * 0.5 + 0.5;
    vec3 Ks = texture(texKs, screenUV).rgb * 2.0;

    float cosThetaI = wiLocal.z;
    float cosThetaV = woLocal.z;

    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    float alpha = 0.2;

    if (ubo.lightType == LIGHT_TYPE_POINT) {
        // Point lighting
        diffuse = cosThetaI * ubo.lightPower.rgb;
        float F = fresnelDielectric(cosThetaI, 1.0, 1.5);
        float fr = microfacetGGXBRDF(wiLocal, woLocal, vec2(alpha));
        specular = Ks * F * fr * cosThetaI * ubo.lightPower.rgb;

        // Shadow
        float vis = visibilityPCF(n, wi);
        diffuse *= vis;
        specular *= vis;
    } else {
        // Environment map
        vec3 no = normalize(inOrigNormal);
        diffuse = computeSH(vec4(no.x, -no.z, no.y, 1.0), ubo.sphereHarmCoefs).rgb;
        specular = vec3(0.0);
    }

    outFragColor = vec4(diffuse, 1.0);
    outSpecColor = vec4(specular, 1.0);
    outPos = vec4(inPos, 1.0);
    outNormal = vec4(inOrigNormal, 1.0);
    outDepth = vec4(vec3(-inPosCamSpace.z), 1.0);
}
