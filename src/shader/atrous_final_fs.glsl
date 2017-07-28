#version 420
precision highp float;
precision highp int;

uniform sampler2D s0;	// coarse.

// TODO
// texture array
uniform sampler2D s1;
uniform sampler2D s2;
uniform sampler2D s3;
uniform sampler2D s4;
uniform sampler2D s5;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

uniform float beta = 1.0;

void main()
{
#if 0
	vec4 data = texelFetch(s0, ivec2(gl_FragCoord.xy), 0);

	data += texelFetch(s1, ivec2(gl_FragCoord.xy), 0) * beta;
	data += texelFetch(s2, ivec2(gl_FragCoord.xy), 0) * beta;
	data += texelFetch(s3, ivec2(gl_FragCoord.xy), 0) * beta;
	data += texelFetch(s4, ivec2(gl_FragCoord.xy), 0) * beta;
	data += texelFetch(s5, ivec2(gl_FragCoord.xy), 0) * beta;

	data.a = 1.0;

	oColour = data;
#else
	oColour = texelFetch(s0, ivec2(gl_FragCoord.xy), 0);
	oColour += texelFetch(s1, ivec2(gl_FragCoord.xy), 0) * beta;
	oColour += texelFetch(s2, ivec2(gl_FragCoord.xy), 0) * beta;
	oColour += texelFetch(s3, ivec2(gl_FragCoord.xy), 0) * beta;
	oColour += texelFetch(s4, ivec2(gl_FragCoord.xy), 0) * beta;
	oColour += texelFetch(s5, ivec2(gl_FragCoord.xy), 0) * beta;
	oColour.a = 1.0;
#endif
}
