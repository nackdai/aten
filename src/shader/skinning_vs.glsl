#version 450
precision highp float;
precision highp int;

attribute vec4 position;
attribute vec3 normal;
attribute vec2 uv;
attribute vec4 blendWeight;
attribute vec4 blendIndex;

uniform mat4 mtxJoint[48];
uniform mat4 mtxW2C;

out vec3 varNormal;
out vec2 varUV;

void main()
{
	varNormal = vec3(0);

	for (int i = 0; i < 4; i++) {
		int idx = int(blendIndex[i]);
		float weight = blendWeight[i];

		mat4 mtx = mtxJoint[idx];

		gl_Position += weight * mtx * position;
		varNormal += weight * mat3(mtx) * normal;
	}

	varNormal = normalize(varNormal);
	varUV = uv;
}
