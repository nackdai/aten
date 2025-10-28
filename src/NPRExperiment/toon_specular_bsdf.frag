#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 position;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec3 binormal;

layout(location = 0) out vec4 outColor;

uniform vec3 eye;

uniform float StretchU;
uniform float StretchV;
uniform float Roughness;

#define RampInputEntryNum 3

uniform float ramp_inputs[RampInputEntryNum];
uniform vec3 ramp_colors[RampInputEntryNum];
uniform int ramp_interp_type;

#define PI 3.14159265358979323846

void FindRampIndex(in float t, out int left_idx, out int right_idx)
{
    left_idx = 0;
    right_idx = RampInputEntryNum - 1;

    if (t < ramp_inputs[left_idx]) {
        left_idx = -1;
        return;
    }

    if (t > ramp_inputs[right_idx]) {
        right_idx = -1;
        return;
    }

    while (right_idx - left_idx > 1) {
        int mid_idx = (left_idx + right_idx) / 2;
        if (t < ramp_inputs[mid_idx]) {
            right_idx = mid_idx;
        } else {
            left_idx = mid_idx;
        }
    }
}

vec3 EvalRamp1D(float t)
{
    t = clamp(t, 0.0, 1.0);

    int left_idx, right_idx;
    FindRampIndex(t, left_idx, right_idx);

    if (left_idx == -1) {
        return ramp_colors[0];
    }
    if (right_idx == -1) {
        return ramp_colors[RampInputEntryNum - 1];
    }

    float span_length = ramp_inputs[right_idx] - ramp_inputs[left_idx];
    if (span_length < 1e-6) {
        return ramp_colors[left_idx];
    }

    vec3 result = vec3(0.0);
    float weight = 0.0;

    t = (t - ramp_inputs[left_idx]) / span_length;

    if (ramp_interp_type == 0) {
        // None
        weight = 0.0;
    }
    else if (ramp_interp_type == 1) {
        // Linear
        weight = t;
    }
    else if (ramp_interp_type == 2) {
        // EXPONENTIAL UP.
        weight = t * t;
    }
    else if (ramp_interp_type == 3) {
        // EXPONENTIAL DOWN.
        weight = 1.0 - (1.0 - t) * (1.0 - t);
    }

    result = ramp_colors[left_idx] + weight * (ramp_colors[right_idx] - ramp_colors[left_idx]);
    return result;
}

// Rodrigues' rotation formula, assume axis is normalized
vec3 rotateVector(vec3 v, vec3 axis, float theta)
{
    float ct = cos(theta);
    float st = sin(theta);
    return ct * v + st * cross(axis, v) + dot(axis, v) * (1.0 - ct) * axis;
}

void main()
{
    float sEpsilon = 1e-4;

	vec3 N = normal;
	vec3 t = tangent;
	vec3 b = binormal;

    vec4 color = vec4(1, 1, 1, 1);
    vec3 light = vec3(0, 0, 5);

    vec3 wi = normalize(light - position);
    vec3 wo = normalize(eye - position);

    vec3 R = normalize(reflect(wi, N));

    float dot_u_l = dot(R, t);
    float dot_u_c = dot(wo, t);
    float dot_u = dot_u_l + dot_u_c;
    float rot_u = clamp(StretchU * dot_u, -0.5f, 0.5f);
    N =  rotateVector(N, b, rot_u);

    float dot_v_l = dot(R, b);
    float dot_v_c = dot(wo, b);
    float dot_v = dot_v_l + dot_v_c;
    float rot_v = (-StretchV * dot_v, -0.5f, 0.5f);
    N = rotateVector(N, t, rot_v);

    R = normalize(reflect(wi, N));
    float specAngle = max(0.0f, dot(-wo, R));
    specAngle = 1.0 - 2.0 * (acos(clamp(specAngle, -1.0f, 1.0f)) / PI);
    float exponent = (1.0f - Roughness) * 10.0;
    specAngle = pow(specAngle, exponent);

    if (specAngle <= 0.0f) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float cosThetaWi = max(dot(N, wi), 0.0f);

    vec3 ramp = EvalRamp1D(specAngle);

    vec3 albedo = clamp(ramp * color.rgb, vec3(0.0), vec3(1.0));

    vec3 final_color = albedo * cosThetaWi / PI;

    outColor = vec4(final_color, color.a);
}
