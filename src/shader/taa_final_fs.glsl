#version 420
precision highp float;
precision highp int;

uniform sampler2D s0;
uniform sampler2D s1;	// aov.

// output colour for the fragment
layout(location = 0) out highp vec4 oColor;

void main()
{
	ivec2 texsize = textureSize(s0, 0);

	vec2 uv = gl_FragCoord.xy / texsize.xy;

	oColor = texture2D(s1, uv);

	oColor = oColor * 0.5 + 0.5;

	oColor.a = 1;
}
