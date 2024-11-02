#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 position;

uniform vec4 color;
uniform float translation_dt;
uniform float translation_db;
uniform float scale_t;
uniform float split_t;
uniform float split_b;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 light = vec3(0, 0, 5);
    vec3 eye = vec3(0, 0, 5);

    vec3 wi = normalize(light - position);
    vec3 wo = normalize(eye - position);
    vec3 wh = normalize(wi + wo);

    vec3 t;
    if (abs(normal.z) > 0) {
        float k = sqrt(normal.y * normal.y + normal.z * normal.z);
        t.x = 0;
        t.y = -normal.z / k;
        t.z = normal.y / k;
    }
    else {
        float k = sqrt(normal.x * normal.x + normal.y * normal.y);
        t.x = normal.y / k;
        t.y = -normal.x / k;
        t.z = 0;
    }
    t = normalize(t);

    vec3 b = cross(normal, t);
    t = cross(b, normal);

    // Translaction.
    vec3 h = wh + translation_dt * t;
    h = normalize(h);

    // Direction scale.
    h = h - scale_t * dot(h, t) * t;
    h = normalize(h);

    // Split.
    h = h - split_t * sign(dot(h, t)) * t;
    h = normalize(h);

    float c = pow(clamp(dot(normal, h), 0, 1), 10);

    outColor = color * (1 - c);
}
