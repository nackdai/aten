#version 450

uniform sampler2D image;

//in highp vec2 vTexCoord;

uniform highp vec2 dstTexel;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

void main()
{
    highp vec2 uv = gl_FragCoord.xy * dstTexel;
    oColour = texture2D(image, uv);
    oColour.a = 1.0;
}
