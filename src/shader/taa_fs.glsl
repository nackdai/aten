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

void main()
{
	ivec2 texsize = textureSize(s0, 0);

	vec2 invScr = 1.0 / texsize.xy;;

	vec2 uv = gl_FragCoord.xy * invScr;

	vec2 neighbor_offset[4] = {
		{  0, +1 },
		{  0, -1 },
		{ +1,  0 },
		{ -1,  0 },
	};

	const float cbcr_threshhold = 0.32;

	vec4 center_color = texture2D(s0, uv);

	vec4 neighbor_sum = center_color;

	for (int i = 0; i < 4; i++) {
		vec4 neighbor = texture2D(s1, uv + neighbor_offset[i] * invScr * blurSize);
		vec3 color_diff = abs(neighbor.xyz - center_color.xyz);
		vec3 ycc = RGB2YCbCr(color_diff.xyz);		// 中心との差をYCbCrで見る
		float cbcr_len = length(color_diff.yz);

		// 色相成分が大きく異なる時、閾値に収まる範囲に色を補正して合成
		if (cbcr_threshhold < cbcr_len) {
			ycc = (cbcr_threshhold / cbcr_len) * ycc;
			neighbor.rgb = center_color.rgb + YCbCr2RGB(ycc);
		}
		neighbor_sum += neighbor;
	}

	neighbor_sum /= 5.0;

	oBuffer = neighbor_sum;
	oBuffer.a = 1;

	oScreen = oBuffer;
}
