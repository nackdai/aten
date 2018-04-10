#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 uv;

// NOTE
// グローバルマトリクス計算時にルートに local to world マトリクスは乗算済み.
// そのため、シェーダでは計算する必要がないので、シェーダに渡されてこない.

uniform mat4 mtxW2C;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;

void main()
{
	gl_Position = mtxW2C * position;

	outNormal = normalize(normal);
	outUV = uv.xy;
}
