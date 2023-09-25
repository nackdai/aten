#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 uv;

// Set by RasterizeRenderer.
uniform mat4 mtx_L2W;
uniform mat4 mtx_W2C;
uniform vec4 color;

layout(location = 0) out vec4 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec4 outClr;

void main()
{
    outWorldPos = mtx_L2W * position;
    outNormal = normalize(mtx_L2W * vec4(normal, 0)).xyz;
    outUV = uv.xy;
    outClr = color;

    gl_Position = mtx_W2C * mtx_L2W * position;
}
