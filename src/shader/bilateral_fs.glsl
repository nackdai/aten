#version 450
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;

uniform vec2 texel;

uniform float sigmaS;
uniform float sigmaR;

uniform bool revert;

// For performace, I should use defined value.
uniform int radius;
//#define radius  (1)

#define buffersize  (10)
uniform float distW[(buffersize + 1) * (buffersize + 1)];

#define PI  3.14159265358979323846

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

vec2 computeUV(in vec2 uv, int sx, int sy)
{
	vec2 smpluv = uv + ivec2(sx, sy) * texel;
	smpluv = clamp(smpluv, 0.0, 1.0);

	return smpluv;
}

// 色距離の重み計算.
float kernelR(float cdist)
{
	float w = 1.0 / sqrt(2.0 * PI * sigmaR) * exp(-0.5 * (cdist * cdist) / (sigmaR * sigmaR));
	return w;
}

// ピクセル距離の重み計算.
float kernelS(int u, int v)
{
	int pos = v * (buffersize + 1) + u;
	float w = distW[pos];
	return w;
}

void main()
{
	ivec2 xy = ivec2(gl_FragCoord.x, gl_FragCoord.y);

	vec2 uv = gl_FragCoord.xy * invScreen.xy;
	if (revert) {
		uv.x = 1.0 - uv.x;
		uv.y = 1.0 - uv.y;
	}

	vec3 p = texture2D(image, uv).rgb;

	// 中心点.
	vec3 numer = vec3(1, 1, 1);
	vec3 denom = p;

	// 横方向.
	for (int u = 1; u <= radius; u++) {
		vec3 p0 = texture2D(image, computeUV(uv, -u, 0)).rgb;
		vec3 p1 = texture2D(image, computeUV(uv, +u, 0)).rgb;

		vec3 wr0 = vec3(
			kernelR(abs(p0.r - p.r)),
			kernelR(abs(p0.g - p.g)),
			kernelR(abs(p0.b - p.b)));
		vec3 wr1 = vec3(
			kernelR(abs(p1.r - p.r)),
			kernelR(abs(p1.g - p.g)),
			kernelR(abs(p1.b - p.b)));

		numer += kernelS(u, 0) * (wr0 + wr1);
		denom += kernelS(u, 0) * (wr0 * p0 + wr1 * p1);
	}

	// 縦方向.
	for (int v = 1; v <= radius; v++) {
		vec3 p0 = texture2D(image, computeUV(uv, 0, -v)).rgb;
		vec3 p1 = texture2D(image, computeUV(uv, 0, +v)).rgb;

		vec3 wr0 = vec3(
			kernelR(abs(p0.r - p.r)),
			kernelR(abs(p0.g - p.g)),
			kernelR(abs(p0.b - p.b)));
		vec3 wr1 = vec3(
			kernelR(abs(p1.r - p.r)),
			kernelR(abs(p1.g - p.g)),
			kernelR(abs(p1.b - p.b)));

		numer += kernelS(0, v) * (wr0 + wr1);
		denom += kernelS(0, v) * (wr0 * p0 + wr1 * p1);
	}

	for (int v = 1; v <= radius; v++) {
		for (int u = 1; u <= radius; u++) {
			vec3 p00 = texture2D(image, computeUV(uv, -u, -v)).rgb;
			vec3 p01 = texture2D(image, computeUV(uv, -u, +v)).rgb;
			vec3 p10 = texture2D(image, computeUV(uv, +u, -v)).rgb;
			vec3 p11 = texture2D(image, computeUV(uv, +u, +v)).rgb;

			vec3 wr00 = vec3(
				kernelR(abs(p00.r - p.r)),
				kernelR(abs(p00.g - p.g)),
				kernelR(abs(p00.b - p.b)));
			vec3 wr01 = vec3(
				kernelR(abs(p01.r - p.r)),
				kernelR(abs(p01.g - p.g)),
				kernelR(abs(p01.b - p.b)));
			vec3 wr10 = vec3(
				kernelR(abs(p10.r - p.r)),
				kernelR(abs(p10.g - p.g)),
				kernelR(abs(p10.b - p.b)));
			vec3 wr11 = vec3(
				kernelR(abs(p11.r - p.r)),
				kernelR(abs(p11.g - p.g)),
				kernelR(abs(p11.b - p.b)));

			numer += kernelS(u, v) * (wr00 + wr01 + wr10 + wr11);
			denom += kernelS(u, v) * (wr00 * p00 + wr01 * p01 + wr10 * p10 + wr11 * p11);
		}
	}

	vec3 color = denom / numer;

	oColour.rgb = color;
	oColour.a = 1.0;
}
