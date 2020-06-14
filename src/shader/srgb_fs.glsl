#version 330
precision highp float;
precision highp int;

uniform sampler2D s0;

uniform vec4 invScreen;

uniform bool revert;

layout(location = 0) out highp vec4 oColor;

void main()
{
    vec2 uv = gl_FragCoord.xy * invScreen.xy;
    if (revert) {
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
    }

    vec3 rgb = texture2D(s0, uv);

    // NOTE
    // http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
    vec3 sq1 = sqrt(rgb);
    vec3 sq2 = sqrt(sq1);
    vec3 sq3 = sqrt(sq2);
    vec3 srgb = 0.662002687 * sq1 + 0.684122060 * sq2 - 0.323583601 * sq3 - 0.0225411470 * c;

    oColor.rgb = srgb;
    oColor.a = 1.0;
}
