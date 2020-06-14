#version 450
precision highp float;
precision highp int;

uniform sampler2D image;

//in vec2 vTexCoord;

uniform vec2 srcTexel;
uniform vec2 dstTexel;

uniform float threshold;

uniform float adaptedLum = 0.2 + 0.00001;

// output colour for the fragment
layout(location = 0) out vec4 oColour;

// NOTE
// +---+---+---+---+  +---+---+---+---+
// |   |   |   |   |  |   |   |   |   |
// +---@---+---@---+  +---0---+---1---+
// |   |   |   |   |  |   |   |   |   |
// +---+---x---+---+  +---+---x---+---+
// |   |   |   |   |  |   |   |   |   |
// +---@-------@---+  +---2---+---3---+
// |   |   |   |   |  |   |   |   |   |
// +---+---+---+---+  +---+---+---+---+

const vec2 offset[4] = vec2[4]
(
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);

const vec3 RGB2Y = vec3(0.29891,  0.58661,  0.11448);
const vec3 YUV2G = vec3(1.00000, -0.50955, -0.19516);

// RGB -> YUV
vec3 RGBToYUV(vec3 vRGB)
{
    float fY = dot(RGB2Y, vRGB);
    return vec3(fY, vRGB.r - fY, vRGB.b - fY);
}

// YUV -> RGB
vec3 YUVToRGB(vec3 vYUV)
{
    float fG = dot(vYUV, YUV2G);
    return vec3(vYUV.x + vYUV.y, fG, vYUV.x + vYUV.z);
}

void main()
{
    oColour = vec4(0.0);

    for (int i = 0; i < 4; i++) {
        vec2 uv = gl_FragCoord.xy * dstTexel;
        uv += offset[i] * srcTexel;

        oColour += texture2D(image, uv);
    }

    oColour *= 0.25;
    oColour.a = 1.0;

    float fMiddleGrey = 0.18;

    // RGB -> YUV
    vec3 vYUV = RGBToYUV(oColour.rgb);

    float yy = max(vYUV.x - threshold, 0.0f);

    yy = yy * fMiddleGrey / (adaptedLum + 0.00001);

    // x' = (1 - exp(2 * x)) ^ 1.5
    float fY = pow(1.0f - exp(-yy * 2.0f), 1.5f);

    // compute how much Y changes.
    float fScale = fY / vYUV.x;

    vYUV.x = fY;
    vYUV.yz *= fScale;  // suitable for chnage ratio of Y.

    // YUV -> RGB
    oColour.rgb = YUVToRGB(vYUV);
}
