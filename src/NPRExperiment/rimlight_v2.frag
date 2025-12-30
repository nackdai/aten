#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 position;

uniform vec3 eye;

uniform vec3 light_pos;

uniform float rim_offset;
uniform float front_rim_intensity;
uniform vec3 front_rim_color;
uniform float back_rim_offset;
uniform float back_rim_intensity;
uniform vec3 back_rim_color;

uniform float rim_width;
uniform float rim_intensity;
uniform float rim_strength;
uniform float rim_contrast;

layout(location = 0) out vec4 outColor;

void main()
{
    // NOTE:
    // Rim light experimental implementation from:
    // https://www.fab.com/ja/listings/00969cf5-848a-4d15-b1bc-f7c76ec126a2

    vec3 obj_color = vec3(0, 0, 0);

    vec3 L = normalize(light_pos - position);
    float NdotL = dot(normal, L);

    vec3 V = normalize(eye - position);
    float NdotV = dot(normal, V);

    float back_rim_t = clamp(NdotL + rim_offset + back_rim_offset, 0, 1);
    float back_rim_band = back_rim_t < 0.5 ? 1.0 : 0.0;
    vec3 back_rim = back_rim_color * back_rim_band * back_rim_intensity;

    float front_rim_t = clamp(NdotL + rim_offset, 0, 1);
    float front_rim_band = front_rim_t < 0.5 ? 0.0 : 1.0;
    vec3 front_rim = front_rim_color * front_rim_band * front_rim_intensity;

    vec3 rim_color = back_rim + front_rim;

    float width = 2.0 / max(rim_width, 0.0001);
    width = clamp(NdotV * width, 0, 1);
    width = 1 - width;
    width = pow(width, max(rim_intensity, 0));
    width *= rim_strength;
    width = max(width - rim_contrast, 0);

    rim_color = min(rim_color, vec3(width));

    outColor.xyz = rim_color;
    outColor.a = 1.0;
}
