#version 450
precision highp float;
precision highp int;

uniform sampler2D s0;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture2D(s0, uv) * color;
}
