#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 position;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec3 binormal;

layout(location = 0) out vec4 outColor;

uniform vec3 eye;

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

	if (dot(N, wi) < 0) {
        // Make shading normal perpendicular to light direction
        // and nudge towards light with lerp
        N = normalize(mix(cross(wi, cross(N, wi)), wi, sEpsilon));
    }

    float cosThetaWi = max(dot(N, wi), 0.0f);

	float pdf = cosThetaWi / PI;

    float cosThetaWo = max(dot(N, wo), 0.0f);

    vec3 ramp = EvalRamp1D(cosThetaWi);

    vec3 albedo = clamp(ramp * color.rgb, vec3(0.0), vec3(1.0));
    float Gs = 1.0;

    float fresnel = pow(clamp(1.0 - cosThetaWo, 0.0, 1.0), 5.0);

    vec3 final_color = Gs * albedo * (1.0 - fresnel) * cosThetaWi / PI;

    outColor = vec4(final_color, color.a);
}
