#version 420
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;
uniform bool revert;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

void main()
{
    vec2 uv = gl_FragCoord.xy * invScreen.xy;
    if (revert) {
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
    }

    vec4 color = texture2D(image, uv);

    oColour.rgb = color.rgb;
    oColour.a = 1;
}
