#version 420
precision highp float;
precision highp int;

uniform sampler2D s0;	// current frame.
uniform sampler2D s1;	// previous frame.

uniform float blurSize = 0.2;

// output colour for the fragment
layout(location = 0) out highp vec4 oBuffer;
layout(location = 1) out highp vec4 oScreen;

vec3 RGB2YCbCr(vec3 rgb)
{
	vec3 RGB2Y = { 0.29900, 0.58700, 0.11400 };
	vec3 RGB2Cb = { -0.16874, -0.33126, 0.50000 };
	vec3 RGB2Cr = { 0.50000, -0.41869, -0.081 };

	return vec3(dot(rgb, RGB2Y), dot(rgb, RGB2Cb), dot(rgb, RGB2Cr));
}

vec3 YCbCr2RGB(vec3 ycc)
{
	vec3 YCbCr2R = { 1.0, 0.00000, 1.40200 };
	vec3 YCbCr2G = { 1.0,-0.34414, -0.71414 };
	vec3 YCbCr2B = { 1.0, 1.77200, 1.40200 };

	return vec3(dot(ycc, YCbCr2R), dot(ycc, YCbCr2G), dot(ycc, YCbCr2B));
}

vec3 clipAABB(
	vec3 aabb_min,
	vec3 aabb_max,
	vec3 q)
{
	vec3 center = 0.5 * (aabb_max + aabb_min);

	vec3 halfsize = 0.5 * (aabb_max - aabb_min) + 0.00000001f;

	// 中心からの相対位置.
	vec3 clip = q - center;

	// 相対位置の正規化.
	vec3 unit = clip / halfsize;

	vec3 abs_unit = abs(unit);

	float ma_unit = max(abs_unit.x, max(abs_unit.y, abs_unit.z));

	if (ma_unit > 1.0) {
		// クリップ位置.
		return center + clip / ma_unit;
	}
	else {
		// point inside aabb
		return q;
	}
}

vec4 sampleColor(sampler2D s, vec2 uv)
{
	vec4 clr = texture2D(s, uv);

	float lum = RGB2YCbCr(clr.xyz).x;

	clr.xyz = clr.xyz / (1 + lum);

	return clr;
}

void main()
{
	ivec2 texsize = textureSize(s0, 0);

	vec2 invScr = 1.0 / texsize.xy;;

	vec2 uv = gl_FragCoord.xy * invScr;

	vec2 du = vec2(invScr.x, 0.0);
	vec2 dv = vec2(0.0, invScr.y);

	vec4 ctl = sampleColor(s0, uv - dv - du);
	vec4 ctc = sampleColor(s0, uv - dv);
	vec4 ctr = sampleColor(s0, uv - dv + du);
	vec4 cml = sampleColor(s0, uv - du);
	vec4 cmc = sampleColor(s0, uv);
	vec4 cmr = sampleColor(s0, uv + du);
	vec4 cbl = sampleColor(s0, uv + dv - du);
	vec4 cbc = sampleColor(s0, uv + dv);
	vec4 cbr = sampleColor(s0, uv + dv + du);

	vec4 cmin = min(ctl, min(ctc, min(ctr, min(cml, min(cmc, min(cmr, min(cbl, min(cbc, cbr))))))));
	vec4 cmax = max(ctl, max(ctc, max(ctr, max(cml, max(cmc, max(cmr, max(cbl, max(cbc, cbr))))))));

	vec4 cavg = (ctl + ctc + ctr + cml + cmc + cmr + cbl + cbc + cbr) / 9.0;

	vec4 cmin5 = min(ctc, min(cml, min(cmc, min(cmr, cbc))));
	vec4 cmax5 = max(ctc, max(cml, max(cmc, max(cmr, cbc))));
	vec4 cavg5 = (ctc + cml + cmc + cmr + cbc) / 5.0;
	cmin = 0.5 * (cmin + cmin5);
	cmax = 0.5 * (cmax + cmax5);
	cavg = 0.5 * (cavg + cavg5);

	vec2 neighbor_offset[4] = {
		{  0, +1 },
		{  0, -1 },
		{ +1,  0 },
		{ -1,  0 },
	};

	const float cbcr_threshhold = 0.32;

	vec4 center_color = sampleColor(s0, uv);

	vec4 neighbor_sum = center_color;

	for (int i = 0; i < 4; i++) {
		vec4 neighbor = sampleColor(s1, uv + neighbor_offset[i] * invScr * blurSize);

		neighbor.xyz = clipAABB(cmin.xyz, cmax.xyz, neighbor.xyz);

#if 1
		vec3 color_diff = abs(neighbor.xyz - center_color.xyz);
		vec3 ycc = RGB2YCbCr(color_diff.xyz);		// 中心との差をYCbCrで見る
		float cbcr_len = length(color_diff.yz);

		// 色相成分が大きく異なる時、閾値に収まる範囲に色を補正して合成
		if (cbcr_threshhold < cbcr_len) {
			ycc = (cbcr_threshhold / cbcr_len) * ycc;
			neighbor.rgb = center_color.rgb + YCbCr2RGB(ycc);
		}
#else
		float lum0 = RGB2YCbCr(center_color.xyz).x;
		float lum1 = RGB2YCbCr(neighbor.xyz).x;

		float unbiased_diff = abs(lum0 - lum1) / max(lum0, max(lum1, 0.2));
		float unbiased_weight = 1.0 - unbiased_diff;
		float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
		float k_feedback = mix(0.2, 0.8, unbiased_weight_sqr);

		neighbor = mix(center_color, neighbor, k_feedback);
#endif

		neighbor_sum += neighbor;
	}

	neighbor_sum /= 5.0;

	float lum = RGB2YCbCr(neighbor_sum.xyz).x;
	neighbor_sum.xyz = neighbor_sum.xyz / (1 - lum);

	oBuffer = neighbor_sum;
	oBuffer.a = 1;

	oScreen = oBuffer;
}
