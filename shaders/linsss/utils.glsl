#ifndef GLSL_UTILS_GLSL
#define GLSL_UTILS_GLSL

#define M_PI 3.141592653589793
#define M_TWO_PI 6.283185307179586
#define M_INV_PI 0.3183098861837907
#define M_INV_TWO_PI 0.15915494309189535
#define M_INV_SQRT_TWO_PI 0.3989422804014327 
#define M_EPS 1.0e-6

// --------------------
// lights
// --------------------
#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_UFFIZI 1
#define LIGHT_TYPE_GRACE 2

float computeSHSub(in vec4 N, in vec4 coefs[9], in int colorIndex) {
    const float SH_C1 = 0.429043;
    const float SH_C2 = 0.511664;
    const float SH_C3 = 0.743125;
    const float SH_C4 = 0.886227;
    const float SH_C5 = 0.247708;

    const int SH_L_0_0  = 0;
	const int SH_L_1_M1 = 1;
	const int SH_L_1_0  = 2;
	const int SH_L_1_1  = 3;
	const int SH_L_2_M2 = 4;
	const int SH_L_2_M1 = 5;
	const int SH_L_2_0  = 6;
	const int SH_L_2_1  = 7;
	const int SH_L_2_2  = 8;

    mat4 m44 = mat4(
        SH_C1 * coefs[ SH_L_2_2 ][ colorIndex ],
		SH_C1 * coefs[ SH_L_2_M2 ][ colorIndex ],
		SH_C1 * coefs[ SH_L_2_1 ][ colorIndex ],
		SH_C2 * coefs[ SH_L_1_1 ][ colorIndex ],

		SH_C1 * coefs[ SH_L_2_M2 ][ colorIndex ],
		- SH_C1 * coefs[ SH_L_2_2 ][ colorIndex ],
		SH_C1 * coefs[ SH_L_2_M1 ][ colorIndex ],
		SH_C2 * coefs[ SH_L_1_M1 ][ colorIndex ],

		SH_C1 * coefs[ SH_L_2_1 ][ colorIndex ],
		SH_C1 * coefs[ SH_L_2_M1 ][ colorIndex ],
		SH_C3 * coefs[ SH_L_2_0 ][ colorIndex ],
		SH_C2 * coefs[ SH_L_1_0 ][ colorIndex ],

		SH_C2 * coefs[ SH_L_1_1 ][ colorIndex ],
		SH_C2 * coefs[ SH_L_1_M1 ][ colorIndex ],
		SH_C2 * coefs[ SH_L_1_0 ][ colorIndex ],
		SH_C4 * coefs[ SH_L_0_0 ][ colorIndex ] -
        SH_C5 * coefs[ SH_L_2_0 ][ colorIndex ]
    );

    vec4 tmp = m44 * N;
    return dot(N, tmp);
}

vec4 computeSH(in vec4 N, in vec4 coefs[9]) {
    float r = computeSHSub(N, coefs, 0);
    float g = computeSHSub(N, coefs, 1);
    float b = computeSHSub(N, coefs, 2);
    return vec4(r, g, b, 1.0);
}

// --------------------
// tonemap
// --------------------

vec3 sRGB(in vec3 color) {
    const float invGamma = 1.0 / 2.4;
    vec3 ret = clamp(color, 0.0, 1.0);
    ret[0] = ret[0] < 0.0031308 ? 12.92 * ret[0] : 1.055 * pow(ret[0], 1.0 / 2.4) - 0.055;
    ret[1] = ret[1] < 0.0031308 ? 12.92 * ret[1] : 1.055 * pow(ret[1], 1.0 / 2.4) - 0.055;
    ret[2] = ret[2] < 0.0031308 ? 12.92 * ret[2] : 1.055 * pow(ret[2], 1.0 / 2.4) - 0.055;
    return ret;
}

vec4 sRGB(in vec4 color) {
    return vec4(sRGB(color.xyz), color.w);
}

// --------------------
// BSDFs
// --------------------

vec3 fresnelConductor(float cosThetaI, vec3 eta, vec3 k) {
    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1.0 - cosThetaI2;
    float sinThetaI4 = sinThetaI2*sinThetaI2;
    vec3 eta2 = eta * eta;
    vec3 k2 = k * k;

    vec3 temp0 = eta2 - k2 - sinThetaI2;
    vec3 a2pb2 = sqrt(max(vec3(0.0), temp0 * temp0 + 4.0 * k2 * eta2));
    vec3 a = sqrt(max(vec3(0.0), (a2pb2 + temp0) * 0.5));

    vec3 temp1 = a2pb2 + vec3(cosThetaI2);
    vec3 temp2 = 2.0 * a * cosThetaI;
    vec3 Rs2 = (temp1 - temp2) / (temp1 + temp2);

    vec3 temp3 = a2pb2 * cosThetaI2 + vec3(sinThetaI4);
    vec3 temp4 = temp2 * sinThetaI2;
    vec3 Rp2 = Rs2 * (temp3 - temp4) / (temp3 + temp4);

    return 0.5 * (Rp2 + Rs2);
}

float fresnelDielectric(in float cosThetaI, in float etaI, in float etaT) {
    cosThetaI = clamp(cosThetaI, -1, 1);
    bool entering = cosThetaI > 0.0;
    if (!entering) {
        float temp = etaT;
        etaT = etaI;
        etaI = temp;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    if (sinThetaT >= 1) return 1.0;
    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) * 0.5;
}

float GGX(in vec3 wh, in vec2 alpha) {
    vec3 whShrink = vec3(wh.x / alpha.x, wh.y / alpha.y, wh.z);
    float l2 = dot(whShrink, whShrink);
    return 1.0 / (M_PI * (alpha.x * alpha.y) * l2 * l2);
}

float microfacetGGXBRDF(in vec3 wi, in vec3 wo, in vec2 alpha) {
    vec3 wh = normalize(wi + wo);
    float zi = abs(wi.z);
    float zo = abs(wo.z);
    vec3 wiStretch = vec3(wi.x * alpha.x, wi.y * alpha.y, wi.z);
    vec3 woStretch = vec3(wo.x * alpha.x, wo.y * alpha.y, wo.z);
    return GGX(wh, alpha) / (2.0 * (zo * length(wiStretch) + zi * length(woStretch)) + M_EPS);
}

vec3 sampleGGXVNDF(in vec3 ve, in vec2 alpha, in vec2 u) {
    // See "Sampling the GGX Distribution of Visible Normals" by E.Heitz in JCGT, 2018.
    vec3 vh = normalize(vec3(ve.x * alpha.x, ve.y * alpha.y, ve.z));

    float lensq = vh.x * vh.x + vh.y * vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(vh, T1);

    float r = sqrt(u.x);
    float phi = 2.0 * M_PI * u.y;

    float t1 = r * cos(float(phi));
    float t2 = r * sin(float(phi));
    float s = 0.5 * (1.0 + vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    vec3 nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * vh;
    vec3 ne = normalize(vec3(nh.x * alpha.x, nh.y * alpha.y, max(0.0, nh.z)));
    return ne;
}

float weightedGGXPDF(in vec3 wi, in vec3 wo, in vec3 wh, in vec2 alpha) {
    vec3 woStretch = vec3(wo.x * alpha.x, wo.y * alpha.y, wo.z);
    return 0.5 / (length(woStretch) + wo.z) * GGX(wh, alpha) * max(0.0, dot(wo, wh)) / max(M_EPS, dot(wi, wh));
}

#endif  // GLSL_UTILS_GLSL
