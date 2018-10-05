#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 uv;

out vec3 worldNormal;
out vec2 vUV;
out vec4 prevWorldPos;

uniform mat4 mtxL2W;
uniform mat4 mtxPrevL2W;

void main()
{
    vec4 worldPos = mtxL2W * position;
    gl_Position = worldPos;

    prevWorldPos = mtxPrevL2W * position;

    worldNormal = normalize(mtxL2W * vec4(normal, 0)).xyz;

    vUV = uv.xy;
}
