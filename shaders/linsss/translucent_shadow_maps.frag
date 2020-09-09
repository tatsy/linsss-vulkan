#version 450

#include "utils.glsl"

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inPosScreen;

layout (location = 0) out vec4 outFragColor;

layout (binding = 1) uniform UBOSSS {
    vec4 sigmas[8];
    float texOffsetX;
    float texOffsetY;
    float texScale;
    float irrScale;
} ubo_sss;

layout (binding = 2) uniform UBOTSM {
	mat4 mvpMat;
	mat4 smMvpMat;
	vec2 screenExtent;
	vec2 bssrdfExtent;
	vec2 seed;
	int numGauss;
	int ksize;
	float sigmaScale;
} ubo_tsm;

layout (binding = 3) uniform sampler2D accumTex;
layout (binding = 4) uniform sampler2D tsmIrrTex;
layout (binding = 5) uniform sampler2D tsmPosTex;
layout (binding = 6) uniform sampler2D tsmNormTex;
layout (binding = 7) uniform sampler3D bssrdfTex;

float eta = 1.5;

vec3 sigmas[32];

vec3 gauss(in float x, in vec3 s) {
	const vec3 invs = 1.0 / s;
	const vec3 xdivs = vec3(x) * invs;
    return M_INV_TWO_PI * exp(-0.5 * xdivs * xdivs) * invs * invs;
}

vec3 getGaussWeight(in vec2 pos, in int h) {
    vec2 uv = pos * 0.5 * ubo_sss.texScale + 0.5;
	uv.x += ubo_sss.texOffsetX;
	uv.y += ubo_sss.texOffsetY;
    float w = float(h + 0.5) / ubo_tsm.numGauss;
    return texture(bssrdfTex, vec3(uv, w)).xyz;
}

vec3 diffRef(in vec3 p0, in vec3 p1) {
	float r = length(p0 - p1);
	vec3 Px = vec3(0.0, 0.0, 0.0);
	vec3 Py = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < ubo_tsm.numGauss; i++) {
		const vec3 G =  gauss(r, sigmas[i]);
		Px += getGaussWeight(p0.xy, i) * G;
		Py += getGaussWeight(p1.xy, i) * G;
	}
    return sqrt(max(vec3(0.0), Px * Py));
}


const int TSM_SAMPLES = 8;
vec2 randState;

float rand() {
    float a = 12.9898;
    float b = 78.233;
    float c = 43758.5453;
    randState.x = fract(sin(float(dot(randState.xy - ubo_tsm.seed, vec2(a, b)))) * c);
    randState.y = fract(sin(float(dot(randState.xy - ubo_tsm.seed, vec2(a, b)))) * c);
    return randState.x;
}

void main() {
	float scale = max(ubo_tsm.bssrdfExtent.x, ubo_tsm.bssrdfExtent.y);
	for (int i = 0; i < ubo_tsm.numGauss; i++) {
		sigmas[i] = ubo_tsm.sigmaScale * ubo_sss.sigmas[i].xyz / scale;
	}

	vec4 posTsmSpace = ubo_tsm.smMvpMat * vec4(inPos, 1.0);
	vec2 st = (posTsmSpace.xy / posTsmSpace.w) * 0.5 + 0.5;

	vec2 screenUV = ((inPosScreen.xy / inPosScreen.w) * 0.5 + 0.5);
	randState = screenUV;

	vec3 rgb = vec3(0.0);
	float sumWgt = 0.0;
	for (int i = 0; i < TSM_SAMPLES; i++) {
		float r_max = 0.1;
		float xi1 = rand();
		float xi2 = rand();

	    vec2 texcoord;
		texcoord.x = st.x + r_max * xi1 * sin(2.0 * M_PI * xi2);
		texcoord.y = st.y + r_max * xi1 * cos(2.0 * M_PI * xi2);
		
		vec3 xi = texture(tsmPosTex, texcoord).xyz;
		vec3 xo = inPos;
		vec3 Rd = diffRef(xo, xi);
		vec3 irr = texture(tsmIrrTex, texcoord).rgb;
		vec3 Mo = irr * ubo_sss.irrScale * Rd;
		float wgt = xi1 * xi1;
		rgb += wgt * Mo;
		sumWgt += wgt;
	}

	vec4 accum = texture(accumTex, screenUV);
	outFragColor = vec4(accum.rgb + rgb / (sumWgt + M_EPS), accum.w + 1.0);
}