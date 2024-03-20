#version 420
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;
uniform bool revert;
uniform float scale;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

void main()
{
    vec2 uv = gl_FragCoord.xy * invScreen.xy;
    if (revert) {
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
    }

    // https://forums.unrealengine.com/t/how-to-convert-bump-map-to-normal-map/80675/6
    float height_pu = texture2D(image, uv + vec2(invScreen.x, 0)).r;
    float height_mu = texture2D(image, uv - vec2(invScreen.x, 0)).r;
    float height_pv = texture2D(image, uv + vec2(0, invScreen.y)).r;
    float height_mv = texture2D(image, uv - vec2(0, invScreen.y)).r;
    float dx = height_mu - height_pu;
    float dy = height_mv - height_pv;

    vec3 nml = normalize(vec3(dx, dy, 1.0 / scale));

    // [-1, 1] -> [0, 1]
    nml = (nml + 1) * 0.5;

    oColour.rgb = nml.rgb;
    oColour.a = 1;
}
