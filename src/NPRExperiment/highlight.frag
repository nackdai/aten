#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 position;

uniform vec4 color;
uniform float translation_dt;
uniform float translation_db;
uniform float scale_t;
uniform float scale_b;
uniform float split_t;
uniform float split_b;
uniform float square_sharp;
uniform float square_magnitude;

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
    vec3 h = wh + translation_dt * t + translation_db * b;
    h = normalize(h);

    // Direction scale.
    h = h - scale_t * dot(h, t) * t - scale_b * dot(h, b) * b;
    h = normalize(h);

    // Split.
    h = h - split_t * sign(dot(h, t)) * t - split_b * sign(dot(h, b)) * b;
    h = normalize(h);

    // Square.
    float sqrnorm_t = sin(pow(acos(dot(h, t)), square_sharp));
    float sqrnorm_b = sin(pow(acos(dot(h, b)), square_sharp));
    h = h - square_magnitude * (sqrnorm_t * dot(h, t) * t + sqrnorm_b * dot(h, b) * b);
    h = normalize(h);

    float c = pow(clamp(dot(normal, h), 0, 1), 10);

    outColor = color * (1 - c);
}
