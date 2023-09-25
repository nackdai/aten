#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec4 prevPosition;

out vec3 worldNormal;
out vec2 vUV;
out vec4 prevWorldPos;

uniform mat4 mtx_L2W;
uniform mat4 mtx_prev_L2W;

void main()
{
    vec4 pos = vec4(position.xyz, 1.0);
    vec3 nml = normal.xyz;
    vec2 uv = vec2(position.w, normal.w);

    vec4 worldPos = mtx_L2W * pos;
    gl_Position = worldPos;

    prevWorldPos = mtx_prev_L2W * vec4(prevPosition.xyz, 1.0);

    worldNormal = normalize(mtx_L2W * vec4(nml, 0)).xyz;

    vUV = uv.xy;
}
