#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 uv;

uniform mat4 mtxL2W;
uniform mat4 mtxW2C;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;

void main()
{
    gl_Position = mtxW2C * mtxL2W * position;

    outNormal = normalize(mtxL2W * vec4(normal, 0)).xyz;
    outUV = uv.xy;
}
