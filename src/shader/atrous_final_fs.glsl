#version 420
precision highp float;
precision highp int;

uniform sampler2D s0;    // coarse.

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

uniform float beta = 0.0;

//#define ENALBLE_SEPARATE_ALBEDO

#if 0
// TODO
// texture array
uniform sampler2D s1;
uniform sampler2D s2;
uniform sampler2D s3;
uniform sampler2D s4;
uniform sampler2D s5;

void main()
{
    oColour = texelFetch(s0, ivec2(gl_FragCoord.xy), 0);
    oColour += texelFetch(s1, ivec2(gl_FragCoord.xy), 0) * beta;
    oColour += texelFetch(s2, ivec2(gl_FragCoord.xy), 0) * beta;
    oColour += texelFetch(s3, ivec2(gl_FragCoord.xy), 0) * beta;
    oColour += texelFetch(s4, ivec2(gl_FragCoord.xy), 0) * beta;
    oColour += texelFetch(s5, ivec2(gl_FragCoord.xy), 0) * beta;
    oColour.a = 1.0;
}
#else
uniform sampler2D s1;    // albedo.

void main()
{
    oColour = texelFetch(s0, ivec2(gl_FragCoord.xy), 0);

#ifdef ENALBLE_SEPARATE_ALBEDO
    vec4 albedo = texelFetch(s1, ivec2(gl_FragCoord.xy), 0);

    oColour *= albedo;
#endif

    oColour.a = 1.0;
}
#endif
