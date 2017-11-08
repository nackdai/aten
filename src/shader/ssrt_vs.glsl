#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 uv;

layout(location = 0) out vec3 worldNormal;
layout(location = 1) out vec2 vUV;

uniform mat4 mtxL2W;
uniform mat4 mtxW2C;

void main()
{
	vec4 worldPos = mtxL2W * position;
	gl_Position = mtxW2C * worldPos;

	worldNormal = normalize(mtxL2W * vec4(normal, 0)).xyz;

	vUV = uv.xy;
}
