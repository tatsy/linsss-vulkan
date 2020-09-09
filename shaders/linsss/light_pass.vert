#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 model;
	vec4 lightPos;
	vec4 lightPower;
} ubo;

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec4 outPosScreen;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outLightPos;
layout (location = 4) out vec3 outLightPower;

out gl_PerVertex 
{
    vec4 gl_Position;   
};

void main() 
{
	gl_Position = ubo.projection * ubo.model * vec4(inPos.xyz, 1.0);
	outPos = inPos;
	outPosScreen = gl_Position;
	outNormal = inNormal;
	outLightPos = ubo.lightPos.xyz;
	outLightPower = ubo.lightPower.rgb;
}
