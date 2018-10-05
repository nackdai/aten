#version 450

uniform sampler2D image;

//in highp vec2 vTexCoord;

uniform highp vec2 srcTexel;
uniform highp vec2 dstTexel;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

// NOTE
// Gauss Function
// f(x) = c * exp(-(x * x) / (2 * d * d))
// (c : constant / d * d : variance)
//
//     |
//     *
//    ***
// ]**o**->x
//    ***
//     *
//     |y

const highp vec2 offset[13] = vec2[13]
(
    vec2(0.0, -2.0),   // 0
    vec2(-1.0, -1.0),  // 1
    vec2(0.0, -1.0),   // 2
    vec2(1.0, -1.0),   // 3
    vec2(2.0,  0.0),   // 4
    vec2(1.0,  0.0),   // 5
    vec2(0.0,  0.0),   // 6
    vec2(1.0,  0.0),   // 7
    vec2(2.0,  0.0),   // 8
    vec2(1.0,  1.0),   // 9
    vec2(0.0,  1.0),   // 10
    vec2(1.0,  1.0),   // 11
    vec2(0.0,  2.0)    // 12

);

const highp float weight[13] = float[13]
(
    0.024882466,
    0.067637555,
    0.11151548,
    0.067637555,
    0.024882466,
    0.11151548,
    0.18385795,
    0.11151548,
    0.024882466,
    0.067637555,
    0.11151548,
    0.067637555,
    0.024882466
);

void main()
{
    for (int i = 0; i < 13; i++) {
        highp vec2 uv = gl_FragCoord.xy * dstTexel.xy;
        uv += offset[i] * srcTexel.xy;

        oColour += weight[i] * texture2D(image, uv);
    }

    oColour.a = 1.0;
}
