#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 uv;

out vec3 worldNormal;
out vec2 vUV;
out vec4 prevWorldPos;

uniform mat4 mtx_L2W;
uniform mat4 mtx_prev_L2W;

void main()
{
    vec4 worldPos = mtx_L2W * position;
    gl_Position = worldPos;

    prevWorldPos = mtx_prev_L2W * position;

    worldNormal = normalize(mtx_L2W * vec4(normal, 0)).xyz;

    vUV = uv.xy;
}
