#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;

uniform mat4 mtxW2C;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;

void main()
{
    vec4 pos = vec4(position.xyz, 1.0);
    vec3 nml = normal.xyz;
    vec2 uv = vec2(position.w, normal.w);

    gl_Position = mtxW2C * pos;

    outNormal = normalize(nml);
    outUV = uv.xy;
}
