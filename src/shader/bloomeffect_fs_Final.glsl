#version 450

uniform sampler2D image;
uniform sampler2D bloomtex;

//in highp vec2 vTexCoord;

uniform highp vec2 texel;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

void main()
{
	highp vec2 uv = gl_FragCoord.xy * texel;

	oColour = texture2D(image, uv);
	
	highp vec4 bloom = texture2D(bloomtex, uv);

	oColour += bloom * 0.4;

	oColour.a = 1.0;
}
