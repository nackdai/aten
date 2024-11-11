#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;

uniform mat4 mtx_L2W;
uniform mat4 mtx_W2C;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outPosition;

void main()
{
    gl_Position = mtx_W2C * mtx_L2W * position;

    outPosition = (mtx_L2W * position).xyz;
    outNormal = normalize(mtx_L2W * vec4(normal, 0)).xyz;
}
