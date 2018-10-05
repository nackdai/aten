#version 420
precision highp float;
precision highp int;

uniform sampler2D s0;    // rt buffer.
uniform sampler2D s1;    // normal map.
uniform sampler2D s2;    // position map.

// NOTE
// pow(2, iteration)
uniform int stepScale;

// NOTE
// pow(2, -iteration)
uniform float clrSigmaScale = 1.0;

uniform float clrSigma = 0.125;
uniform float nmlSigma = 0.125;
uniform float posSigma = 0.125;
uniform float threshold = 0.05125;

layout(location = 0) out vec4 oCoarse;
layout(location = 1) out vec4 oDetail;

// NOTE
// h = [1/16, 1/4, 3/8, 1/4, 1/16]
// H = h * h
const float kernel[25] = float[25](
    1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
    3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
    1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0);

// NOTE
// 5x5
const ivec2 offsets[25] = ivec2[25](
    ivec2(-2, -2), ivec2(-1, -2), ivec2(0, -2), ivec2(1, -2), ivec2(2, -2),
    ivec2(-2, -1), ivec2(-1, -1), ivec2(0, -2), ivec2(1, -1), ivec2(2, -1),
    ivec2(-2, 0), ivec2(-1, 0), ivec2(0, 0), ivec2(1, 0), ivec2(2, 0),
    ivec2(-2, 1), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1), ivec2(2, 1),
    ivec2(-2, 2), ivec2(-1, 2), ivec2(0, 2), ivec2(1, 2), ivec2(2, 2));

void main()
{
    vec4 sum = vec4(0.0);
    float weightSum = 0.0;

    ivec2 centerUV = ivec2(gl_FragCoord.xy);

    vec4 centerClr = texelFetch(s0, ivec2(gl_FragCoord.xy), 0);
    vec4 centerNml = texelFetch(s1, ivec2(gl_FragCoord.xy), 0);
    vec4 centerPos = texelFetch(s2, ivec2(gl_FragCoord.xy), 0);

    int centerMtrlId = int(centerPos.w);
    int centerMeshId = int(centerNml.w);

    if (centerMeshId < 0) {
        oCoarse = vec4(1, 1, 1, 1);
        return;
    }

    centerPos.w = 1.0f;
    centerNml.w = 0.0f;

    // NOTE
    // 5x5 = 25
    for (int i = 0; i < 25; i++) {
        ivec2 uv = centerUV + offsets[i] * stepScale;

        vec4 clr = texelFetch(s0, uv, 0);
        vec4 nml = texelFetch(s1, uv, 0);

        vec4 pos = texelFetch(s2, uv, 0);

        int mtrlId = int(pos.w);
        int meshId = int(nml.w);

        if (meshId < 0) {
            continue;
        }

        nml.w = 0.0f;
        pos.w = 1.0f;

        if (mtrlId != centerMtrlId || meshId != centerMeshId) {
            continue;
        }

        vec4 delta = clr - centerClr;
        float dist2 = dot(delta, delta);
        float w_rt = min(exp(-dist2 / (clrSigma * clrSigmaScale)), 1.0);

        delta = nml - centerNml;
        dist2 = dot(delta, delta);
        float w_n = min(exp(-dist2 / nmlSigma), 1.0);

        delta = pos - centerPos;
        dist2 = dot(delta, delta);
        float w_p = min(exp(-dist2 / posSigma), 1.0);

        float weight = w_rt * w_n * w_p;

        sum += clr * weight * kernel[i];
        weightSum += weight * kernel[i];
    }

    vec4 coarse = sum / weightSum;
    vec4 detail = centerClr - coarse;

    oCoarse = coarse;
    oDetail = max(vec4(0.0), abs(detail) - vec4(threshold)) * sign(detail);
}
