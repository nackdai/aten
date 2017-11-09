#version 450
precision highp float;
precision highp int;

in vec3 normal;
in vec2 uv;

layout(location = 0) out vec4 outColor;

void main()
{
	outColor.xyz = normal * 0.5 + 0.5;
	outColor.w = 1.0;
}
