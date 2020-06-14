#version 450

uniform sampler2D image;

//in highp vec2 vTexCoord;

uniform highp vec2 srcTexel;
uniform highp vec2 dstTexel;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

// NOTE
// +---+---+---+---+---+---+---+---+---+---+---+---+---+
// | x | x | x | x | x | x | @ | x | x | x | x | x | x |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+
// |<----->|<----->|<----->|   |<----->|<----->|<----->|
// Handle every 2 texels.

// NOTE
// How to compute offset.
// +-------+-------+
// |       |       |
// |   2   |@  1   |
// |       |       |
// +-------+-------+
// The offset value is interpolated gauss weight value between 1 and 2.
// The offset value is offset from 1.
//   Offset = Weight2 / (Weight1 + Weight2)

const highp vec2 offset[7] = vec2[7]
(
    vec2(0.0, 0.0),
    vec2(1.4073334, 0.0),
    vec2(3.2942150, 0.0),
    vec2(5.2018132, 0.0),
    vec2(-1.4073334, 0.0),
    vec2(-3.2942150, 0.0),
    vec2(-5.2018132, 0.0)
);

// NOTE
// How to compute gauss weight.
// Pseud code.
//  float tmp[13] = computeGaussOneDirection13Points();
//  float weight[7];
//  weight[0] = tmp[0];
//  for (int i = 0; i < 7; i++) {
//      weight[i + 1] = tmp[i * 2 + 1] + tmp[i * 2 + 2];
//  }

const highp float weight[7] = float[7]
(
    0.59841347,
    0.89105409,
    0.27526283,
    0.032940224,
    0.89105409,
    0.27526283,
    0.032940224
);

void main()
{
    for (int i = 0; i < 7; i++) {
        highp vec2 uv = gl_FragCoord.xy * dstTexel.xy;
        uv += offset[i] * srcTexel.xy;

        oColour += weight[i] * texture2D(image, uv);
    }

    oColour.a = 1.0;
}
