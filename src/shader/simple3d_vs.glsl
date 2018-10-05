#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 mtxL2W;
uniform mat4 mtxW2C;

out vec3 varColor;

void main()
{
    gl_Position = mtxW2C * mtxL2W * position;
    varColor = color;
}
