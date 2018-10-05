#version 420
precision highp float;
precision highp int;

uniform sampler2D s0;

// output colour for the fragment
layout(location = 0) out highp vec4 oColor;

void main()
{
    ivec2 texsize = textureSize(s0, 0);

    vec2 uv = gl_FragCoord.xy / texsize.xy;

    oColor = texture2D(s0, uv);
    oColor.a = 1;
}
