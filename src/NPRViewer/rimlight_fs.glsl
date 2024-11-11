#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 position;

uniform vec3 eye;

uniform vec3 light_color;
uniform vec3 light_pos;

uniform float width;
uniform float softness;
uniform float spread;

layout(location = 0) out vec4 outColor;

// bezier curve with 2 control points
// A is the starting point, B, C are the control points, D is the destination
// t from 0 ~ 1
float bezier(float B0, float B1, float B2, float t)
{
    float P = (B0 - 2 * B1 + B2) * t * t + (-2 * B0 + 2 * B1) * t + B0;
    return P;
}

float bezier_smoothstep(float edge0, float edge1, float mid, float t, float s)
{
    if (t <= edge0) {
        return 0;
    }
    else if (t >= edge1) {
        return 1;
    }

    t = (t - edge0) / (edge1 - edge0);
    t *= s;

    float P = bezier(0, mid, 1, t);
    return P;
}

void main()
{
    vec3 obj_color = vec3(0, 0, 0);

    vec3 L = normalize(light_pos - position);
    float NdotL = dot(normal, L);

    vec3 V = normalize(eye - position);
    float NdotV = dot(normal, V);

    float rim = 0;

    // NOTE:
    // width is larger, rim light is thicker. If width is smaller, rim light is thinner.
    // As smoothstep, less than edge0 is zero.
    // In that case, if width is large, the result of smoothstep might be more zero.
    // It means, if width is larger, rim light is thinner. It's fully opposite what widthe means.
    // Therefore, width need to be invert as smoothstep edge0 argument.
    if (NdotV > 0) {
        rim = bezier_smoothstep(1.0 - width, 1.0, (1 - softness) * 0.5, 1 - NdotV, spread);
    }

    obj_color += rim * light_color;

    outColor.xyz = obj_color;
    outColor.w = 1;
}
