#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec4 blendIndex;
layout(location = 5) in vec4 blendWeight;

// NOTE:
// グローバルマトリクス計算時にルートに local to world マトリクスは乗算済み.
// そのため、シェーダでは計算する必要がないので、シェーダに渡されてこない.

uniform mat4 mtxJoint[48];
uniform mat4 mtx_W2C;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec4 outColor;

void main()
{
    gl_Position = vec4(0);
    outNormal = vec3(0);

    for (int i = 0; i < 4; i++) {
        int idx = int(blendIndex[i]);
        float weight = blendWeight[i];

        mat4 mtx = mtxJoint[idx];

        gl_Position += weight * mtx * position;
        outNormal += weight * mat3(mtx) * normal;
    }

    gl_Position.w = 1;
    gl_Position = mtx_W2C * gl_Position;
    outNormal = normalize(outNormal);

    outUV = uv;
    outColor = color;
}
