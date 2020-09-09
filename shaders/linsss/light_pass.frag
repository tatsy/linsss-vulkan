#version 450

#include "utils.glsl"

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec4 inPosScreen;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inLightPos;
layout (location = 4) in vec3 inLightPower;

layout (location = 0) out vec4 outFragColor;
layout (location = 1) out vec4 outPos;
layout (location = 2) out vec4 outNormal;

void main() 
{
	vec3 n = normalize(inNormal);
	vec3 l = normalize(inLightPos - inPos);

	outFragColor = vec4(inLightPower * vec3(max(0.0, dot(n, l))), 1.0);
	outPos = vec4(inPos, 1.0);
	outNormal = vec4(n, 1.0);
}
