#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 worldNormal;
layout(location = 1) in vec2 vUV;

layout(location = 0) out vec4 outColor;

void main()
{
	outColor.xyz = worldNormal * 0.5 + 0.5;
	outColor.w = 1.0;
}
