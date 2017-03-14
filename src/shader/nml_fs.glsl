#version 450
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;

uniform vec2 texel;

uniform float param_h;
uniform float sigma;

uniform bool revert;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

#define F	(2)
#define R	(6)

#define kKernel		(2 * F + 1)
#define kSupport	(2 * R + 1)
#define kHalfKernel (kKernel / 2)
#define kHalfSupport (kSupport / 2)

const vec4 powV = vec4(2.0, 2.0, 2.0, 2.0);

struct Template {
    vec4 a[kKernel * kKernel];
};

vec2 computeUV(in vec2 uv, int sx, int sy)
{
    vec2 smpluv = uv + ivec2(sx, sy) * texel;
    smpluv = clamp(smpluv, 0.0, 1.0);

    return smpluv;
}

void sampleArea(
    out Template tmpl,
    vec2 uv)
{
    int count = 0;

    for (int sx = -kHalfKernel; sx <= kHalfKernel; sx++) {
        for (int sy = -kHalfKernel; sy <= kHalfKernel; sy++) {
            vec2 smpluv = computeUV(uv, sx, sy);

            tmpl.a[count++] = texture2D(image, smpluv);
        }
    }
}

float computeDistanceSquared(in Template a, in Template b)
{
    vec4 sumV = vec4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < kKernel * kKernel; i++) {
        sumV += pow(a.a[i] - b.a[i], powV);
    }

    float sum = sumV.r;
    sum += sumV.g;
    sum += sumV.b;

	sum /= (kKernel * kKernel * 3);

    return sum;
}

void main()
{
    ivec2 xy = ivec2(gl_FragCoord.x, gl_FragCoord.y);

    vec2 uv = gl_FragCoord.xy * invScreen.xy;
	if (revert) {
		uv.x = 1.0 - uv.x;
		uv.y = 1.0 - uv.y;
	}

    // 注目領域.
    Template focus;
    sampleArea(focus, uv);

    vec4 sum = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    float sum_weight = 0.0f;

    for (int sx = -kHalfSupport; sx <= kHalfSupport; ++sx) {
        for (int sy = -kHalfSupport; sy <= kHalfSupport; ++sy) {
            // 相似度を調べる対象領域.
            Template target;
            vec2 tmpuv = computeUV(uv, sx, sy);
            sampleArea(target, tmpuv);

            // ノルム（相似度）計算.
            float dist = computeDistanceSquared(focus, target);

            // NOTE
            // Z(p) = sum(exp(-max(|v(p) - v(q)|^2 - 2σ^2, 0) / h^2))
            float arg = -max(dist - 2.0 * sigma * sigma, 0.0) / (param_h * param_h);

            float weight = exp(arg);

            vec4 pixel = texture2D(image, tmpuv);

            sum += weight * pixel;
            sum_weight += weight;
        }
    }

    vec4 color = sum / sum_weight;

	oColour = color;
	oColour.a = 1.0;
}
