#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 position;

uniform vec3 eye;

// Umbra to penumbra separation threshold.
uniform float shadow_level;

// Penumbra to highlights separation threshold.
uniform float midtone_level;

// Softness of tonal range transition, higher values produce softner results, lower values produce sharper transitions
uniform float diffuse_softness;

// Global diffuse term tint.
uniform vec3 diffuse_tint;

uniform vec3 shadow_color;
uniform vec3 midtone_color;
uniform vec3 highlight_color;

// Softness of transition from rim highlight to no highlights area.
uniform float rim_softness;

// Global rim reflection contribution.
uniform float rim_weight;

// Global rim reflection tint.
uniform vec3 rim_tint;

// Specular highlight to no highlights transition threshold.
uniform float glossy_level;

// Softness of transition from specular highlight to no highlights area.
uniform float glossy_softness;

uniform vec3 glossy_color;

// Global specular reflection tint.
uniform vec3 specular_tint;

layout(location = 0) out vec4 outColor;

#define pi 3.14159265358979323846

float luminance(vec3 v)
{
    float r = v.r;
    float g = v.g;
    float b = v.b;
    float ret = 0.212639F * r + 0.71517F * g + 0.0721926F * b;
    return ret;
}

vec3 RGBtoHSV(vec3 rgb)
{
    vec3 hsv;
    float minc = min(rgb.r, min(rgb.g, rgb.b));
    float maxc = max(rgb.r, max(rgb.g, rgb.b));
    hsv.z = maxc;
    float delta = maxc - minc;
    if (delta < 0.00001 || maxc < 0.00001) {
        hsv.x = 0;
        hsv.y = 0;
        return hsv;
    }
    hsv.y = delta / maxc;
    if (rgb.r >= maxc) {
        hsv.x = (rgb.g - rgb.b) / delta; // between yellow & magenta
    }
    else if (rgb.g >= maxc) {
        hsv.x = 2 + (rgb.b - rgb.r) / delta; // between cyan & yellow
    }
    else {
        hsv.x = 4 + (rgb.r - rgb.g) / delta; // between magenta & cyan
    }
    hsv.x *= 60; // to degrees
    if (hsv.x < 0) {
        hsv.x += 360;
    }
    return hsv;
}

vec3 HSVtoRGB(vec3 in_v)
{
    float hh, p, q, t, ff;
    int i;
    vec3 out_v;
    if (in_v.y <= 0.0) { // < is bogus, just shuts up warnings
        out_v = vec3(in_v.z, in_v.z, in_v.z); // achromatic (grey)
        return out_v;
    }
    hh = in_v.x;
    if (hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = int(hh);
    ff = hh - float(i);
    p = in_v.z * (1.0 - in_v.y);
    q = in_v.z * (1.0 - (in_v.y * ff));
    t = in_v.z * (1.0 - (in_v.y * (1.0 - ff)));
    switch (i) {
    case 0:
        out_v = vec3(in_v.z, t, p);
        break;
    case 1:
        out_v = vec3(q, in_v.z, p);
        break;
    case 2:
        out_v = vec3(p, in_v.z, t);
        break;
    case 3:
        out_v = vec3(p, q, in_v.z);
        break;
    case 4:
        out_v = vec3(t, p, in_v.z);
        break;
    case 5:
    default:
        out_v = vec3(in_v.z, p, q);
        break;
    }
    return out_v;
}

void main()
{
    vec4 color = vec4(1, 1, 1, 1);
    vec3 light = vec3(0, 0, 5);

    vec3 light_dir = normalize(light - position);
    vec3 n = normalize(normal);
    vec3 V = normalize(eye - position);

    vec3 base_diffuse_term = vec3(max(dot(n, light_dir), 0) / pi);

    vec3 diffuse_term = base_diffuse_term;
    {
        float diffuse_Y = clamp(luminance(diffuse_term), 0, 1);
        vec3 diffuse_hsv = RGBtoHSV(diffuse_term);
        vec3 diffuse_clr = HSVtoRGB(vec3(diffuse_hsv.r, diffuse_hsv.g, 1));

        // The same perceptual mapping considerations apply here.
        diffuse_Y = pow(diffuse_Y, 1.0 / 2.2);

        float shadows_level = min(shadow_level, midtone_level);
        float midtones_level = max(shadow_level, midtone_level);

        float umbra = smoothstep(
            shadows_level,
            min(1.0, shadows_level + diffuse_softness),
            diffuse_Y);

        float penumbra = smoothstep(
            midtones_level,
            min(1.0, midtones_level + diffuse_softness),
            diffuse_Y);

        diffuse_term = mix(
            shadow_color,
            midtone_color,
            umbra);

        diffuse_term = mix(
            diffuse_term,
            highlight_color,
            penumbra);
    }

    vec3 rim_term = base_diffuse_term;
    {
        // NOTE:
        // diffuse_Y is strong when the surface aims toward the light direction.
        // On the other hand, rim light appears when the surface doesn't aim toward the view direction.
        // Therefore, if rim light depends on diffuse_Y, the rim light appears only when the surface aims toward the light direction.
        // So, we don't use diffuse_Y and the relevant values from diffuse_Y.

        //float diffuse_Y = clamp(luminance(rim_term), 0.0, 1.0);
        //float mixing = smoothstep(0.0, 0.9, diffuse_Y);

        float facing = 1.0 - max(0.0, dot(n, V));

        //float rim_factor = mixing * pow(facing, max(2.0, sqrt(rim_softness) * 10));
        float rim_factor = pow(facing, max(2.0, sqrt(rim_softness) * 10));

        rim_factor = clamp(rim_factor, 0.0, 1.0);
        rim_factor = smoothstep(
            max(0.0, 0.5 - rim_softness),
            min(1.0, 0.5 + rim_softness),
            rim_factor);

        rim_factor = clamp(rim_factor, 0.0, 1.0);

        diffuse_term = mix(
            diffuse_term,
            rim_weight * rim_tint,
            rim_factor);
    }

    diffuse_term = diffuse_term * diffuse_tint;
    diffuse_term = max(vec3(0.0), diffuse_term);

    vec3 specular_term = vec3(clamp(dot(n, light_dir), 0, 1));
    {
        float specular_Y = clamp(luminance(specular_term), 0, 1);
        vec3 specular_hsv = RGBtoHSV(specular_term);
        vec3 specular_clr = HSVtoRGB(vec3(specular_hsv.r, specular_hsv.g, 1));

        // The same perceptual mapping considerations apply here.
        specular_Y = pow(specular_Y, 1.0 / 2.2);

        float highlight = smoothstep(
            glossy_level,
            min(1.0, glossy_level + glossy_softness),
            specular_Y);

        specular_term = mix(
            vec3(0),
            glossy_color,
            highlight);

        specular_term = specular_term * specular_tint;

        specular_term = max(vec3(0.0), specular_term);

        // Final blend with pure light color, with its HSV V set to 1.0
        specular_term = specular_term * specular_clr;
    }

    outColor = color * vec4(specular_term.rgb + diffuse_term.rgb, 1.0);
}
