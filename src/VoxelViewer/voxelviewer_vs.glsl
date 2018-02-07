#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;

uniform mat4 mtxL2W;
uniform mat4 mtxW2C;

void main()
{
	gl_Position = mtxW2C * mtxL2W * position;
}
