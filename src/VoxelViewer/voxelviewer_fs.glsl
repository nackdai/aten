#version 450
precision highp float;
precision highp int;

uniform vec3 color;
uniform vec3 normal;

layout(location = 0) out vec4 outColor;

void main()
{
	outColor = vec4(color, 1);
}
