#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec2 uv;

uniform sampler2D s0;    // albedo.
uniform vec4 color;

uniform bool hasAlbedo = false;

layout(location = 0) out vec4 outColor;

void main()
{
    if (hasAlbedo) {
        outColor = texture2D(s0, uv) * color;
    }
    else {
        // TODO
        outColor = color;
    }
}
