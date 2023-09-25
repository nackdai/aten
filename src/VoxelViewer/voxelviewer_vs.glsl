#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;

uniform mat4 mtx_L2W;
uniform mat4 mtx_W2C;

void main()
{
    gl_Position = mtx_W2C * mtx_L2W * position;
}
