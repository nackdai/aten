// Tone mapping from Gran Turismo 7.

#version 330
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;
uniform bool revert = false;

// Where to end the range of toe = Where to start the range of linear.
uniform float end_of_toe = 0.22;

// To control contrast.
uniform float contrast_param = 1.0;

// Max monitor luminance. 100[nit] = 1.0
uniform float max_monitor_luminance = 1.0;

// The range of linear.
uniform float range_of_linear = 0.4;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

vec3 gt_tonemapper_toe(vec3 x)
{
    float b = 0.0;
    float c = 1.33;
    vec3 T = end_of_toe * pow(x / end_of_toe, vec3(c)) + b;
    return T;
}

vec3 gt_tonmapper_linear(vec3 x)
{
    vec3 L = end_of_toe + contrast_param * (x - end_of_toe);
    return L;
}

vec3 gt_tonemapper_shoulder(vec3 x, float l0)
{
    float S0 = end_of_toe + l0;
    float S1 = end_of_toe + contrast_param * l0;
    float C2 = contrast_param * max_monitor_luminance / (max_monitor_luminance - S1);
    vec3 S = max_monitor_luminance - (max_monitor_luminance - S1) * exp(-C2 * (x - S0) / max_monitor_luminance);
    return S;
}

vec3 RGBtoXYZ(vec3 rgb)
{
    mat3 mtx_RGBtoXYZ = mat3(
        2.7689, 1.7517, 1.1302,
        1.0000, 4.5907, 0.0601,
        0.0000, 0.0565, 5.6943
    );
    return mtx_RGBtoXYZ * rgb;
}

vec3 XYZtoRGB(vec3 xyz)
{
    mat3 mtx_XYZtoRGB = mat3(
        0.4185, -0.1587, -0.0814,
        -0.0912, 0.2524, 0.0154,
        0.0009, -0.0025, 0.1755
    );
    return mtx_XYZtoRGB * xyz;
}

void main()
{
    vec2 uv = gl_FragCoord.xy * invScreen.xy;
    if (revert) {
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
    }

    vec4 col = texture2D(image, uv);

    // Convert to XYZ.
    vec3 xyz = RGBtoXYZ(col.rgb);

#if 1
    float Y = xyz.y;
    float l0 = (max_monitor_luminance - end_of_toe) * range_of_linear / contrast_param;

    vec3 T = gt_tonemapper_toe(xyz);
    vec3 L = gt_tonmapper_linear(xyz);
    vec3 S = gt_tonemapper_shoulder(xyz, l0);

    vec3 w0 = 1 - smoothstep(0, end_of_toe, xyz);
    vec3 w2 = step(end_of_toe + l0, xyz);
    vec3 w1 = 1 - w0 - w2;

    float Y_dash = (T * w0 + L * w1 + S * w2).y;

    // NOTE:
    // Tone mapping should be applied to only Y (luminance).
    // In that case, we need to arrange XZ values with the converted rate of Y like:
    //   X' = X * (Y'/Y), Z' = Z * (Y' / Y)
    xyz.y = Y_dash;
    xyz.xz *= (Y_dash / Y);
#endif

    // Convert to RGB with the tone mapped luminance.
    col.rgb = XYZtoRGB(xyz);

    oColour.rgb = col.rgb;
    oColour.a = 1;
}
