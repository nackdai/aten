#version 330
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;

uniform float coeff;
uniform float l_max;

vec3 RGB2Y   = vec3(+0.29900, +0.58700, +0.11400);
vec3 RGB2Cb  = vec3(-0.16874, -0.33126, +0.50000);
vec3 RGB2Cr  = vec3(+0.50000, -0.41869, -0.08131);
vec3 YCbCr2R = vec3(+1.00000, +0.00000, +1.40200);
vec3 YCbCr2G = vec3(+1.00000, -0.34414, -0.71414);
vec3 YCbCr2B = vec3(+1.00000, +1.77200, +0.00000);

void main()
{
    vec2 uv = gl_FragCoord.xy * invScreen.xy;
    //uv.y = 1.0 - uv.y;

	vec4 col = sqrt(texture2D(image, uv));

	float y = dot(RGB2Y, col.rgb);
	float cb = dot(RGB2Cb, col.rgb);
	float cr = dot(RGB2Cr, col.rgb);

	y = coeff * y;
	y = y * (1.0 + y / (l_max * l_max)) / (1.0 + y);

	vec3 ycbcr = vec3(y, cb, cr);

	float r = dot(YCbCr2R, ycbcr);
	float g = dot(YCbCr2G, ycbcr);
	float b = dot(YCbCr2B, ycbcr);

	gl_FragColor.rgb = clamp(vec3(r, g, b), 0, 1);
	gl_FragColor.a = 1;
}
