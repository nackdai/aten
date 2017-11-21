#version 450
precision highp float;
precision highp int;

in vec3 varColor;

layout(location = 0) out vec4 outColor;

void main()
{
	outColor.xyz = varColor.xyz;
	outColor.w = 1;
}
