#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;

out vec3 worldNormal;
out vec2 vUV;
out vec4 prevWorldPos;

uniform mat4 mtxL2W;
uniform mat4 mtxPrevL2W;

void main()
{
	vec4 pos = vec4(position.xyz, 1.0);
	vec3 nml = normal.xyz;
	vec2 uv = vec2(position.w, normal.w);

	vec4 worldPos = mtxL2W * pos;
	gl_Position = worldPos;

	prevWorldPos = mtxPrevL2W * pos;

	worldNormal = normalize(mtxL2W * vec4(nml, 0)).xyz;

	vUV = uv.xy;
}
