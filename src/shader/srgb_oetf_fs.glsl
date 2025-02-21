// Tone mapping from Gran Turismo 7.

#version 450
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;
uniform bool revert = false;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

float SRGBOptoElectronicTransferFunction(float v)
{
    if (v <= 0.0031308f) {
        return v * 12.92f;
    }
    return 1.055f * pow(v, 1 / 2.4f) - 0.055f;
}

void main()
{
    vec2 uv = gl_FragCoord.xy * invScreen.xy;
    if (revert) {
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
    }

    vec4 col = texture2D(image, uv);

    col.r = SRGBOptoElectronicTransferFunction(col.r);
    col.g = SRGBOptoElectronicTransferFunction(col.g);
    col.b = SRGBOptoElectronicTransferFunction(col.b);

    oColour.rgb = col.rgb;
    oColour.a = 1;
}
