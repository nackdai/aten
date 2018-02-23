#version 450
precision highp float;
precision highp int;

uniform sampler2D s0;

in vec3 varNormal;
in vec2 varUV;
in vec4 varColor;

layout(location = 0) out vec4 outColor;

void main()
{
	outColor = texture2D(s0, varUV) * varColor;
}
